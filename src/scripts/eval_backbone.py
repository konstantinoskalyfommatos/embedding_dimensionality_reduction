from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import torch
import logging
import os
from utils.config import EVALUATION_RESULTS_PATH

from utils.eval import evaluate_sts, evaluate_retrieval, evaluate_classification, evaluate_clustering


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a distilled SentenceTransformer model on Benchmarks")
    parser.add_argument(
        "--backbone_model", type=str, 
        help="Name or path of the backbone SentenceTransformer model",
        default="Alibaba-NLP/gte-multilingual-base"
    )
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--sts_batch_size", type=int, default=2048, help="Batch size for STS evaluation")
    parser.add_argument("--classification_batch_size", type=int, default=6, help="Batch size for classification evaluation")
    parser.add_argument("--retrieval_batch_size", type=int, default=20, help="Batch size for retrieval evaluation")
    parser.add_argument("--clustering_batch_size", type=int, default=16, help="Batch size for clustering evaluation")

    args = parser.parse_args()

    model = SentenceTransformer(args.backbone_model, device="cuda", trust_remote_code=True)
    model.eval()

    # Evaluate the model
    cache_path = os.path.join(EVALUATION_RESULTS_PATH, "backbone", args.backbone_model.replace("/", "__"))
    if not args.skip_sts:
        sts_score = evaluate_sts(
            model=model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.sts_batch_size
        )
        logger.info(f"Final Spearman correlation on STS test set: {sts_score:.4f}")

    if not args.skip_classification:
        classification_score = evaluate_classification(
            model=model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.classification_batch_size
        )
        logger.info(f"Final classification results: {classification_score}")

    if not args.skip_retrieval:
        retrieval_score = evaluate_retrieval(
            model=model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.retrieval_batch_size
        )
        logger.info(f"Final retrieval results: {retrieval_score}")

    if not args.skip_clustering:
        clustering_score = evaluate_clustering(
            model=model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.clustering_batch_size
        )
        logger.info(f"Final clustering results: {clustering_score}")
