from sentence_transformers import SentenceTransformer

import torch
import mteb

import logging

from utils.config import PROJECT_ROOT

# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_sts(
    model: SentenceTransformer, 
    model_name: str,
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
    batch_size: int = 2048
) -> float:
    """Evaluates a SentenceTransformer model on the STSBenchmark English dataset.

    Returns average Spearman correlation score.
    """
    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}",
        encode_kwargs={'batch_size': batch_size},
        overwrite_results=True
    )
    scores = [result.scores["test"][0]["main_score"] for result in results]
    return sum(scores) / len(scores)

def evaluate_retrieval(
    model: SentenceTransformer, 
    model_name: str,
    tasks_list: list[str] = [
        # "MIRACLRetrievalHardNegatives",
        "QuoraRetrievalHardNegatives",
        "HotpotQAHardNegatives",
        "DBPediaHardNegatives",
        "NQHardNegatives",
        "MSMARCOHardNegatives",
        "ArguAna"    
    ],
    languages: list[str] | None = None,
    batch_size: int = 4
) -> float:
    """Evaluates a SentenceTransformer model on retrieval task.

    Returns average NDCG@10 score.
    """
    # This benchmark has issues with specifying language
    # if languages and "MIRACLRetrievalHardNegatives" in tasks_list:
    #     tasks_list.remove("MIRACLRetrievalHardNegatives")

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}",
        encode_kwargs={'batch_size': batch_size},
        overwrite_results=True
    )
    scores = [result.scores["test"][0]["main_score"] for result in results]
    return sum(scores) / len(scores)


def evaluate_classification(
    model: SentenceTransformer, 
    model_name: str,
    tasks_list: list[str] = [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",  # This has .v2 version
        "AmazonReviewsClassification",
        "ImdbClassification",  # This has .v2 version
        "ToxicConversationsClassification",  # This has .v2 version      
    ],
    languages: list[str] | None = None,
    batch_size: int = 16
) -> float:
    """Evaluates a SentenceTransformer model on classification task.

    Returns accuracy score.
    """
    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}",
        encode_kwargs={'batch_size': batch_size},
        overwrite_results=True
    )
    scores = [result.scores["test"][0]["main_score"] for result in results]
    return sum(scores) / len(scores)


def evaluate_clustering(
    model: SentenceTransformer,
    model_name: str,
    tasks_list: list[str] = [
        "ArxivClusteringS2S",  # This has newer version (ArXivHierarchicalClusteringS2S)
        "RedditClustering",  # This has .v2 version
        "StackExchangeClustering"  # This has .v2 version
    ],
    languages: list[str] | None = None,
    batch_size: int = 16
):
    """Evaluates a SentenceTransformer model on clustering task.

    Returns average V-measure score.
    """
    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}",
        encode_kwargs={'batch_size': batch_size},
        overwrite_results=True
    )
    scores = [result.scores["test"][0]["main_score"] for result in results]
    return sum(scores) / len(scores)
