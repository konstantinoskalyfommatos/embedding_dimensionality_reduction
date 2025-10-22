from sentence_transformers import SentenceTransformer

import torch
import mteb

import logging

from utils.config import PROJECT_ROOT

# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: Test other datasets
def evaluate_sts(
    model: SentenceTransformer, 
    model_name: str,
    tasks_list: list[str] = ["STSBenchmark"],
    batch_size: int = 2048
) -> float:
    """Evaluates a SentenceTransformer model on the STSBenchmark English dataset.

    Returns Spearman correlation score.
    """
    tasks = mteb.get_tasks(tasks=tasks_list)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}/sts",
        encode_kwargs={'batch_size': batch_size},
        overwrite_results=True
    )
    return results[0].scores["test"][0]["main_score"]


def evaluate_retrieval(
    model: SentenceTransformer, 
    model_name: str,
    tasks_list: list[str] = ["ArguAna"],
    batch_size: int = 4
) -> float:
    """Evaluates a SentenceTransformer model on retrieval task.

    Returns NDCG@10 score.
    """
    tasks = mteb.get_tasks(tasks=tasks_list)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}/retrieval",
        encode_kwargs={'batch_size': batch_size},
        overwrite_results=True
    )
    return results[0].scores["test"][0]["main_score"]


def evaluate_classification(
    model: SentenceTransformer, 
    model_name: str,
    tasks_list: list[str] = ["ToxicConversationsClassification.v2"],
    batch_size: int = 32
) -> float:
    """Evaluates a SentenceTransformer model on classification task.

    Returns accuracy score.
    """
    tasks = mteb.get_tasks(tasks=tasks_list, languages=["en"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}/classification",
        encode_kwargs={'batch_size': batch_size},
        overwrite_results=True
    )
    return results[0].scores["test"][0]["main_score"]
