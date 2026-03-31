from sentence_transformers import SentenceTransformer
import torch
import os
import mteb
from mteb.cache import ResultCache
import logging
import json
import torchsort
import json
from dotenv import load_dotenv

from huggingface_hub import login

from utils.config import PROJECT_ROOT

load_dotenv()


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)


try:
    if hf_token := os.getenv("HF_TOKEN"):
        login(token=hf_token)
except Exception as e:
    logger.warning(f"Could not log in to Hugging Face Hub: {e}")


def find_checkpoint_lowest_val_loss(trained_path: str) -> tuple[float, int]:
    """Find the checkpoint with the lowest validation loss."""
    last_checkpoint = max([
        int(d.split("checkpoint-")[-1])
        for d in os.listdir(trained_path)
        if d.startswith("checkpoint-")
    ])

    with open(
        os.path.join(
            trained_path, 
            f"checkpoint-{last_checkpoint}", 
            "trainer_state.json"
        )
    ) as f:
        trainer_state = json.load(f)
    
    lowest_val = 100000
    best_checkpoint = None
    for info_dict in trainer_state["log_history"]:
        if not "eval_loss" in info_dict:
            continue
        if info_dict["eval_loss"] < lowest_val:
            lowest_val = info_dict["eval_loss"]
            best_checkpoint = info_dict["step"]

    return lowest_val, best_checkpoint


def compute_positional_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    weighted: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute weighted pairwise distance preservation loss.

    L = (1 / n) * sum_{i<j} w_ij * (d_low(x_i, x_j) - d_high(x_i, x_j))^2
    where w_ij = 1 / (d_high(x_i, x_j) + eps)^m
    """
    low_dim_dist = torch.cdist(low_dim_embeddings, low_dim_embeddings, p=2)
    high_dim_dist = torch.cdist(high_dim_embeddings, high_dim_embeddings, p=2)

    n = low_dim_dist.size(0)
    triu_indices = torch.triu_indices(
        n, n, offset=1, device=low_dim_embeddings.device
    )

    low_d = low_dim_dist[triu_indices[0], triu_indices[1]]
    high_d = high_dim_dist[triu_indices[0], triu_indices[1]]

    if not weighted:
        loss = (low_d - high_d).pow(2).mean()
    else:
        weights = 1.0 / (high_d + eps)
        loss = (weights * (low_d - high_d).pow(2)).mean()

    return loss


def compute_angular_loss(
    low_dim_embeddings: torch.Tensor, 
    high_dim_embeddings: torch.Tensor,
    weighted: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute pairwise cosine similarity preservation loss."""
    # Normalize embeddings to unit vectors
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)
    
    # Compute similarity matrices
    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())
    
    # Use triu_indices for better memory efficiency
    n = low_dim_sim.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)
    
    low_dim_sim_upper = low_dim_sim[triu_indices[0], triu_indices[1]]
    high_dim_sim_upper = high_dim_sim[triu_indices[0], triu_indices[1]]

    if not weighted:
        return (low_dim_sim_upper - high_dim_sim_upper).pow(2).mean()
    
    weights = 1.0 / (1.0 - high_dim_sim_upper + eps)
    return (weights * (low_dim_sim_upper - high_dim_sim_upper).pow(2)).mean()


# --- Pearson's correlation ---
def pearsonr(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighted: bool = False,
    **kw
):
    n = target.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=target.device)
    pred = pred[triu_indices[0], triu_indices[1]]
    target = target[triu_indices[0], triu_indices[1]]
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)

    if weighted:
        weights = target / target.sum(dim=1, keepdim=True)
        
        w_sum = weights.sum(dim=1, keepdim=True)
        pred_mean = (weights * pred).sum(dim=1, keepdim=True) / w_sum
        target_mean = (weights * target).sum(dim=1, keepdim=True) / w_sum
        
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        
        cov = (weights * pred_centered * target_centered).sum(dim=1)
        pred_var = (weights * pred_centered ** 2).sum(dim=1)
        target_var = (weights * target_centered ** 2).sum(dim=1)
        
        eps = 1e-8
        correlation = cov / (torch.sqrt(pred_var * target_var) + eps)
        return correlation.mean()
    else:
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return (pred * target).sum()


def compute_pearson_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    training: bool = False,
    weighted: bool = False
) -> torch.Tensor:
    """Compute Pearson correlation loss between low and high dimensional embeddings."""
    # Normalize embeddings
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)
    
    # Compute similarity matrices
    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())

    with torch.set_grad_enabled(training):
        return 1.0 - pearsonr(low_dim_sim, high_dim_sim, weighted=weighted)


# --- Spearman's rank ---
def spearmanr_differentiable(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighted: bool = False,
    **kw
):
    n = target.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=target.device)
    pred = pred[triu_indices[0], triu_indices[1]]
    target = target[triu_indices[0], triu_indices[1]]
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)

    pred_ranks = torchsort.soft_rank(pred, **kw)
    target_ranks = torchsort.soft_rank(target, **kw)
    
    if weighted:
        # Normalize soft ranks to use as weights
        weights = target_ranks / target_ranks.sum(dim=1, keepdim=True)
        
        # Weighted mean centering
        w_sum = weights.sum(dim=1, keepdim=True)
        pred_mean = (weights * pred_ranks).sum(dim=1, keepdim=True) / w_sum
        target_mean = (weights * target_ranks).sum(dim=1, keepdim=True) / w_sum
        
        pred_centered = pred_ranks - pred_mean
        target_centered = target_ranks - target_mean
        
        # Weighted covariance and variances
        cov = (weights * pred_centered * target_centered).sum(dim=1)
        pred_var = (weights * pred_centered ** 2).sum(dim=1)
        target_var = (weights * target_centered ** 2).sum(dim=1)
        
        eps = 1e-8
        correlation = cov / (torch.sqrt(pred_var * target_var) + eps)
        return correlation.mean()
    else:
        pred_ranks = pred_ranks - pred_ranks.mean()
        pred_ranks = pred_ranks / pred_ranks.norm()
        target_ranks = target_ranks - target_ranks.mean()
        target_ranks = target_ranks / target_ranks.norm()
        return (pred_ranks * target_ranks).sum()


def spearmanr_differentiable_local(
    pred: torch.Tensor, 
    target: torch.Tensor,
    weighted: bool = False,
    **kw
) -> torch.Tensor:
    """Differentiable Spearman correlation weighted by target rank position."""
    n = pred.shape[0]
    m = n - 1
    
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=pred.device)
    pred_offdiag = pred[mask].reshape(n, m)
    target_offdiag = target[mask].reshape(n, m)
    
    # Get soft ranks of target similarities (continuous approximations)
    target_soft_ranks = torchsort.soft_rank(target_offdiag, **kw)
    
    if weighted:
        # Normalize soft ranks to use as weights
        weights = target_soft_ranks / target_soft_ranks.sum(dim=1, keepdim=True)
    else:
        weights = torch.ones_like(target_soft_ranks)

    # Get ranks of predictions
    pred_ranks = torchsort.soft_rank(pred_offdiag, **kw)
    
    # Weighted mean centering
    w_sum = weights.sum(dim=1, keepdim=True)
    pred_mean = (weights * pred_ranks).sum(dim=1, keepdim=True) / w_sum
    target_mean = (weights * target_soft_ranks).sum(dim=1, keepdim=True) / w_sum
    
    pred_centered = pred_ranks - pred_mean
    target_centered = target_soft_ranks - target_mean
    
    # Weighted covariance and variances
    cov = (weights * pred_centered * target_centered).sum(dim=1)
    pred_var = (weights * pred_centered ** 2).sum(dim=1)
    target_var = (weights * target_centered ** 2).sum(dim=1)
    
    eps = 1e-8
    row_correlations = cov / (torch.sqrt(pred_var * target_var) + eps)
    
    return row_correlations.mean()


def compute_spearman_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    local_or_global: str = "local",
    training: bool = False,
    weighted: bool = False,
) -> torch.Tensor | float:
    """Compute Spearman correlation loss between low and high dimensional embeddings."""
    # Normalize embeddings
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)
    
    # Compute similarity matrices
    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())

    with torch.set_grad_enabled(training):
        if local_or_global == "local":
            return 1.0 - spearmanr_differentiable_local(low_dim_sim, high_dim_sim, weighted=weighted)
        else:
            return 1.0 - spearmanr_differentiable(low_dim_sim, high_dim_sim, weighted=weighted)


# --- Eval functions ---
@torch.no_grad()
def eval_intrinsic(
    projection: torch.nn.Module,
    backbone_model_path: str,
    dataset_name = "sentence-paraphrases",
    checkpoint: str | None = None,
    cache_path: str | None = None,
    model_name: str | None = None,
    spearman_test_batch_size: int | None = 20000,
    spearman_local_or_global: str = "local"
):
    results = {
        "task_name": "IntrinsicEvaluation",
        "checkpoint": checkpoint,
        "spearman_loss": None,
        "spearman_loss_weighted": None,
        "angular_loss": None,
        "angular_loss_weighted": None,
        "positional_loss": None,
        "positional_loss_weighted": None
    }

    test_embeddings_path = os.path.join(
        PROJECT_ROOT,
        "storage",
        "precalculated_embeddings",
        dataset_name.split("/")[-1],
        backbone_model_path.replace("/", "__"),
        "test_embeddings.pt"
    )
    
    if not os.path.exists(test_embeddings_path):
        raise FileNotFoundError(
            f"Precalculated embeddings not found at {test_embeddings_path}"
        )
    
    high_dim_embeddings: torch.Tensor = torch.load(test_embeddings_path).to("cuda")
    low_dim_embeddings = projection(high_dim_embeddings)

    spearman_loss = compute_spearman_loss(
        low_dim_embeddings[:spearman_test_batch_size], 
        high_dim_embeddings[:spearman_test_batch_size], 
        training=False,
        weighted=False,
        local_or_global=spearman_local_or_global
    )
    results["spearman_loss"] = spearman_loss.item()

    spearman_loss_weighted = compute_spearman_loss(
        low_dim_embeddings[:spearman_test_batch_size], 
        high_dim_embeddings[:spearman_test_batch_size], 
        training=False,
        weighted=True,
        local_or_global=spearman_local_or_global
    )
    results["spearman_loss_weighted"] = spearman_loss_weighted.item()

    angular_loss = compute_angular_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=False
    )
    results["angular_loss"] = angular_loss.item()
    
    angular_loss_weighted = compute_angular_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=True
    )
    results["angular_loss_weighted"] = angular_loss_weighted.item()

    positional_loss = compute_positional_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=False
    )
    results["positional_loss"] = positional_loss.item()
    
    positional_loss_weighted = compute_positional_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=True
    )
    results["positional_loss_weighted"] = positional_loss_weighted.item()

    logger.info(
        f"Intrinsic results for {model_name}, checkpoint {checkpoint}: "
        f"{json.dumps(results, indent=2)}"
    )
    if not cache_path:
        return results
    if not model_name:
        raise ValueError("model_name must be provided to save intrinsic test results")

    # Save in the same directory structure as MTEB results
    intrinsic_results_path = os.path.join(
        cache_path, 
        "results", 
        model_name.replace("/", "__"),
        "no_revision_available"
    )
    os.makedirs(intrinsic_results_path, exist_ok=True)
    save_path = os.path.join(intrinsic_results_path, "intrinsic.json")
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved at: {save_path}")

    return results
    

# --- MTEB evaluation ---

def evaluate_sts(
    model: SentenceTransformer, 
    cache_path: str,
    tasks_list: list[str] = [
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICK-R"
    ],
    languages: list[str] | None = None,
    batch_size: int = 4096,
    fast_mode: bool = False,
    overwrite_cache: bool = False
) -> float:
    """Evaluates a SentenceTransformer model on the STSBenchmark English dataset.

    Returns average Spearman correlation score.
    """
    if fast_mode:
        logger.info("Using fast mode for STS")
        tasks_list = ["STSBenchmark"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model, 
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing",
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)

def evaluate_retrieval(
    model: SentenceTransformer,
    cache_path: str,
    tasks_list: list[str] = [
        "ArguAna",

        # "MIRACLRetrievalHardNegatives",
        # "QuoraRetrievalHardNegatives",
        # "HotpotQAHardNegatives",
        # "DBPediaHardNegatives",
        # "NQHardNegatives",
        # "MSMARCOHardNegatives",

        "QuoraRetrieval",
        "HotpotQA",
        "DBPedia",
        "NQ",
        "MSMARCO"

    ],
    languages: list[str] | None = None,
    batch_size: int = 6,
    fast_mode: bool = False,
    overwrite_cache: bool = False
) -> float:
    """Evaluates a SentenceTransformer model on retrieval task.

    Returns average NDCG@10 score.
    """
    if fast_mode:
        logger.info("Using fast mode for retrieval")
        tasks_list = ["ArguAna"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model, 
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing"
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)
 

def evaluate_classification(
    model: SentenceTransformer,
    cache_path: str,
    tasks_list: list[str] = [
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "ImdbClassification",  # This has .v2 version
        "ToxicConversationsClassification",  # This has .v2 version
        "AmazonPolarityClassification",  # This has .v2 version    
    ],
    languages: list[str] | None = None,
    batch_size: int = 16,
    fast_mode: bool = False,
    overwrite_cache: bool = False
) -> float:
    """Evaluates a SentenceTransformer model on classification task.

    Returns accuracy score.
    """
    if fast_mode:
        logger.info("Using fast mode for classification")
        tasks_list = ["AmazonCounterfactualClassification"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model,
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing"
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)


def evaluate_clustering(
    model: SentenceTransformer,
    cache_path: str,
    tasks_list: list[str] = [
        "ArxivClusteringS2S",  # This has newer version (ArXivHierarchicalClusteringS2S)
        "RedditClustering",  # This has .v2 version
        "StackExchangeClustering"  # This has .v2 version
    ],
    languages: list[str] | None = None,
    batch_size: int = 20,
    fast_mode: bool = False,
    overwrite_cache: bool = False
):
    """Evaluates a SentenceTransformer model on clustering task.

    Returns average V-measure score.
    """
    if fast_mode:
        logger.info("Using fast mode for clustering")
        tasks_list = ["RedditClustering"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model, 
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing"
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)


def evaluate_mteb(
    model: SentenceTransformer,
    cache_path: str,
    sts_batch_size: int = 4096,
    retrieval_batch_size: int = 6,
    classification_batch_size: int = 16,
    clustering_batch_size: int = 20,
    skip_sts: bool = False,
    skip_retrieval: bool = False,
    skip_classification: bool = False,
    skip_clustering: bool = False,
    fast_mode: bool = False,
    overwrite_cache: bool = False
):
    """Evaluates a SentenceTransformer model on MTEB benchmark."""
    if not skip_sts:
        sts_score = evaluate_sts(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=sts_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Spearman correlation on STS test set: {sts_score:.4f}")
    
    if not skip_retrieval:
        retrieval_score = evaluate_retrieval(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=retrieval_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Retrieval NDCG@10 score: {retrieval_score:.4f}")

    if not skip_classification:
        classification_score = evaluate_classification(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=classification_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Classification accuracy score: {classification_score:.4f}")

    if not skip_clustering:
        clustering_score = evaluate_clustering(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=clustering_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Clustering V-measure score: {clustering_score:.4f}")
