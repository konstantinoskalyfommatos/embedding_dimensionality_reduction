from sentence_transformers import SentenceTransformer
import torch
import os
import mteb
from mteb.cache import ResultCache
import logging
import json
import torchsort
from ignite.engine import Engine
from ignite.metrics.regression import SpearmanRankCorrelation

from utils.config import PROJECT_ROOT


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)


def compute_positional_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    weight_exponent: float = 1.0,
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

    if weight_exponent == 0:
        return (low_d - high_d).pow(2).mean()

    weights = 1.0 / (high_d + eps).pow(weight_exponent)
    loss = weights * (low_d - high_d).pow(2)

    return loss.mean()


def compute_angular_loss(
    low_dim_embeddings: torch.Tensor, 
    high_dim_embeddings: torch.Tensor,
    weight_exponent: int = 0,
    eps: float = 1e-8,
    loss_factor: int = 1  # NOTE: Try out 25
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

    if weight_exponent == 0:
        return (low_dim_sim_upper - high_dim_sim_upper).pow(2).mean() * loss_factor
    
    weights = 1.0 / (1.0 - high_dim_sim_upper + eps).pow(weight_exponent)
    return (weights * (low_dim_sim_upper - high_dim_sim_upper).pow(2)).mean() * loss_factor


# --- Spearman's rank ---
def spearmanr_differentiable(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    **kw
) -> torch.Tensor:
    """Differentiable Spearman correlation weighted by target rank position."""
    n = pred.shape[0]
    
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=pred.device)
    pred_offdiag = pred[mask].reshape(n, n-1)
    target_offdiag = target[mask].reshape(n, n-1)
    
    # Get ranks of target similarities
    target_ranks = torchsort.soft_rank(target_offdiag, **kw)
    
    # NOTE: We are using rank position as weights
    
    # Get ranks of predictions (regular ascending order)
    pred_ranks = torchsort.soft_rank(pred_offdiag, **kw)
    
    # Weighted mean centering
    w_sum = target_ranks.sum(dim=1, keepdim=True)
    pred_mean = (target_ranks * pred_ranks).sum(dim=1, keepdim=True) / w_sum
    target_mean = (target_ranks * target_ranks).sum(dim=1, keepdim=True) / w_sum
    
    pred_centered = pred_ranks - pred_mean
    target_centered = target_ranks - target_mean
    
    # Weighted covariance and variances
    cov = (target_ranks * pred_centered * target_centered).sum(dim=1)
    pred_var = (target_ranks * pred_centered ** 2).sum(dim=1)
    target_var = (target_ranks * target_centered ** 2).sum(dim=1)
    
    row_correlations = cov / (torch.sqrt(pred_var * target_var))
    
    return row_correlations.mean()


def evaluation_step(engine, batch):
    return batch

def spearmanr_non_differentiable(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute average Spearman correlation across rows of similarity matrices.
    
    Args:
        pred: 2D tensor of shape (n, n) - predicted similarity matrix
        target: 2D tensor of shape (n, n) - target similarity matrix
    
    Returns:
        Average Spearman correlation across all rows
    """
    n = pred.size(0)
    correlations = []
    
    default_evaluator = Engine(evaluation_step)
    metric = SpearmanRankCorrelation(device="cuda")
    
    for i in range(n):
        metric.reset()
        metric.attach(default_evaluator, 'spearman_corr')
        corr = default_evaluator.run([[pred[i], target[i]]]).metrics["spearman_corr"]
        correlations.append(corr)
    
    return sum(correlations) / len(correlations)


def compute_spearman_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    training: bool = False
) -> torch.Tensor | float:
    """Compute Spearman correlation loss between low and high dimensional embeddings."""
    # Normalize embeddings
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)
    
    # Compute similarity matrices
    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())

    n = low_dim_embeddings.size(0)

    if training:
        # Use differentiable torchsort during training (slower but supports backprop)
        # triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)
        # low_flat = low_dim_sim[triu_indices[0], triu_indices[1]]
        # high_flat = high_dim_sim[triu_indices[0], triu_indices[1]]
        # low_flat = low_flat.unsqueeze(0)
        # high_flat = high_flat.unsqueeze(0)
        return 1.0 - spearmanr_differentiable(low_dim_sim, high_dim_sim)
    with torch.no_grad():
        return 1.0 - spearmanr_differentiable(low_dim_sim, high_dim_sim)


# --- Eval functions ---


def eval_intrinsic(
    projection: torch.nn.Module,
    backbone_model_path: str,
    dataset_name = "sentence-paraphrases",
    positional_or_angular: str = "positional",
    weight_exponent: int = 0,
    checkpoint: str | None = None,
    cache_path: str | None = None,
    model_name: str | None = None,
):
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
    high_dim_embeddings = high_dim_embeddings[:15000]

    with torch.inference_mode():
        low_dim_embeddings = projection(high_dim_embeddings)

        return compute_spearman_loss(low_dim_embeddings, high_dim_embeddings, training=False)
    # if positional_or_angular == "angular":
    #     loss = compute_angular_loss(
    #         low_dim_embeddings=low_dim_embeddings,
    #         high_dim_embeddings=high_dim_embeddings,
    #         weight_exponent=weight_exponent
    #     )
    # else:
    #     loss = compute_positional_loss(
    #     low_dim_embeddings=low_dim_embeddings,
    #     high_dim_embeddings=high_dim_embeddings,
    #     weight_exponent=weight_exponent
    # )
        
    if not cache_path:
        return loss.item()
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
    
    # Only save if loss is lower than previous or file doesn't exist
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            previous_results = json.load(f)
        previous_loss = previous_results["scores"]["test"][0]["main_score"]
        if loss.item() >= previous_loss:
            return loss.item()
    
    # To be compatible with MTEB results format
    results = {
        "task_name": "IntrinsicEvaluation",
        "checkpoint": checkpoint,
        "scores": {
            "test": [
                {"main_score": loss.item()}
            ]
        }
    }
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved at: {save_path}")

    return loss.item()
    

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
        # "MIRACLRetrievalHardNegatives",
        "ArguAna",
        "QuoraRetrievalHardNegatives",
        "HotpotQAHardNegatives",
        "DBPediaHardNegatives",
        "NQHardNegatives",
        "MSMARCOHardNegatives",
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
