from sentence_transformers import SentenceTransformer
import torch
import os
import mteb
from mteb.cache import ResultCache
import logging

from utils.config import PROJECT_ROOT


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_positional_loss(
    low_dim_embeddings: torch.Tensor, 
    high_dim_embeddings: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise distance preservation loss."""
    low_dim_dist = torch.cdist(low_dim_embeddings, low_dim_embeddings, p=2)
    high_dim_dist = torch.cdist(high_dim_embeddings, high_dim_embeddings, p=2)
    
    # Use triu_indices for better memory efficiency
    n = low_dim_dist.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)
    
    low_dim_dist_upper = low_dim_dist[triu_indices[0], triu_indices[1]]
    high_dim_dist_upper = high_dim_dist[triu_indices[0], triu_indices[1]]
    
    return torch.nn.functional.mse_loss(
        low_dim_dist_upper, 
        high_dim_dist_upper, 
        reduction="mean"
    )


def compute_angular_loss(
    low_dim_embeddings: torch.Tensor, 
    high_dim_embeddings: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise cosine similarity preservation loss."""
    # Compute similarity matrices
    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())
    
    # Use triu_indices for better memory efficiency
    n = low_dim_sim.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)
    
    low_dim_sim_upper = low_dim_sim[triu_indices[0], triu_indices[1]]
    high_dim_sim_upper = high_dim_sim[triu_indices[0], triu_indices[1]]

    return torch.nn.functional.mse_loss(
        low_dim_sim_upper, 
        high_dim_sim_upper, 
        reduction="mean"
    ) * 100


# --- Eval functions ---


def eval_intrinsic(
    projection: torch.nn.Module,
    backbone_model_path: str,
    dataset_name = "sentence-paraphrases",
    positional_or_angular: str = "positional"
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
    high_dim_embeddings: torch.Tensor = torch.load(test_embeddings_path)
    high_dim_embeddings = high_dim_embeddings.to("cuda")

    low_dim_embeddings = projection(high_dim_embeddings)

    if positional_or_angular == "angular":
        return compute_angular_loss(
            low_dim_embeddings=low_dim_embeddings,
            high_dim_embeddings=high_dim_embeddings
        )
    return compute_positional_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings
    )


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
    # This benchmark has issues with specifying language
    # if languages and "MIRACLRetrievalHardNegatives" in tasks_list:
    #     tasks_list.remove("MIRACLRetrievalHardNegatives")

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
        "AmazonPolarityClassification",  # This has .v2 version
        "AmazonReviewsClassification",
        "ImdbClassification",  # This has .v2 version
        "ToxicConversationsClassification",  # This has .v2 version      
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
