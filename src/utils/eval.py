from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import torch
import mteb
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
)
import logging

from utils.config import PROJECT_ROOT

# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_sts(
    model: SentenceTransformer, 
    model_name: str,
    split: str = "test",
    batch_size: int = 2048
) -> float:
    """Evaluates a SentenceTransformer model on the STSBenchmark English dataset.

    Returns Spearman correlation score.
    """
    tasks = mteb.get_tasks(tasks=["STSBenchmark"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model, 
        output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}",
        batch_size=batch_size
    )
    
    task_result = results[0]
    scores = task_result.get("scores", {})
    split_scores = scores.get(split, {})
    cos_sim_scores = split_scores.get("cos_sim", {})
    spearman = cos_sim_scores.get("spearman", 0.0)

    return float(spearman)


def evaluate_retrieval(
    model: SentenceTransformer, 
    model_name: str,
    split: str = "test"
) -> float:
    """Evaluates a SentenceTransformer model on retrieval task.

    Returns NDCG@10 score.
    """
    tasks = mteb.get_tasks(tasks=["ArguAna"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"{PROJECT_ROOT}/storage/results/{model_name}", batch_size=4)
    
    task_result = results[0]
    scores = task_result.get("scores", {})
    split_scores = scores.get(split, {})
    ndcg_at_10 = split_scores.get("ndcg_at_10", 0.0)
    
    return float(ndcg_at_10)
