from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging
import sys

from utils.config import EVALUATION_RESULTS_PATH
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import evaluate_sts, evaluate_retrieval, evaluate_classification, evaluate_clustering, eval_intrinsic


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a distilled SentenceTransformer model on STSBenchmark")
    parser.add_argument(
        "--backbone_model", type=str, 
        help="Name or path of the backbone SentenceTransformer model",
        default="Alibaba-NLP/gte-multilingual-base"
    )
    parser.add_argument(
        "--backbone_model_output_dim",
        default=768
    )
    parser.add_argument("--target_dim", type=int, default=32, help="Target dimension of the distilled embeddings")
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--sts_batch_size", type=int, default=2048, help="Batch size for STS evaluation")
    parser.add_argument("--classification_batch_size", type=int, default=6, help="Batch size for classification evaluation")
    parser.add_argument("--retrieval_batch_size", type=int, default=20, help="Batch size for retrieval evaluation")
    parser.add_argument("--clustering_batch_size", type=int, default=16, help="Batch size for clustering evaluation")
    parser.add_argument("--eval_intrinsic", action="store_true", help="Evaluate only on the intrinsic test set")
    parser.add_argument("--positional_or_angular", type=str, default="positional", help="Whether to use positional or angular loss for intrinsic evaluation")

    args = parser.parse_args()
    logger.info(f"Args: {args}")
    
    # Select first args.target_dim indices
    indices = torch.arange(args.target_dim)

    # Create selection matrix
    M = torch.zeros(args.backbone_model_output_dim, args.target_dim)
    M[indices, torch.arange(args.target_dim)] = 1.0

    projection_head = nn.Linear(args.backbone_model_output_dim, args.target_dim, bias=False)
    projection_head.weight = nn.Parameter(M.t())

    print(projection_head)

    if args.eval_intrinsic:
        projection_head.eval()
        loss = eval_intrinsic(
            projection=projection_head,
            backbone_model_path=args.backbone_model,
            positional_or_angular=args.positional_or_angular
        )
        logger.info(f"Intrinsic test loss: {loss}")
        sys.exit(0)

    model_name = os.path.join(
        f"{args.backbone_model}"
        f"_distilled_{args.target_dim}"
        "_truncation"
    )

    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name
    )

    custom_model.eval()

    # Evaluate the model
    cache_path = os.path.join(
        EVALUATION_RESULTS_PATH,
        "truncation",
        args.backbone_model.replace("/", "__"),
    )

    if not args.skip_sts:
        sts_score = evaluate_sts(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.sts_batch_size
        )
        logger.info(f"Final Spearman correlation on STS test set: {sts_score:.4f}")

    if not args.skip_retrieval:
        retrieval_score = evaluate_retrieval(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.retrieval_batch_size
        )
        logger.info(f"Final retrieval results: {retrieval_score}")

    if not args.skip_clustering:
        clustering_score = evaluate_clustering(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.clustering_batch_size
        )
        logger.info(f"Final clustering results: {clustering_score}")

    if not args.skip_classification:
        classification_score = evaluate_classification(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.classification_batch_size
        )
        logger.info(f"Final classification results: {classification_score}")
        