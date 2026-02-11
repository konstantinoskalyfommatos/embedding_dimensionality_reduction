from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging
import json
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
    parser.add_argument("--backbone_model", type=str, default="jinaai/jina-embeddings-v2-small-en", help="Name or path of the backbone SentenceTransformer model")
    parser.add_argument("--backbone_model_output_dim", default=512, type=int)

    parser.add_argument("--target_dim", type=int, default=32, help="Target dimension of the distilled embeddings")
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--sts_batch_size", type=int, default=2048, help="Batch size for STS evaluation")
    parser.add_argument("--classification_batch_size", type=int, default=6, help="Batch size for classification evaluation")
    parser.add_argument("--retrieval_batch_size", type=int, default=48, help="Batch size for retrieval evaluation")
    parser.add_argument("--clustering_batch_size", type=int, default=16, help="Batch size for clustering evaluation")
    parser.add_argument("--normalize_vector_before_projecting", action="store_true")

    parser.add_argument("--eval_intrinsic", action="store_true", help="Evaluate only on the intrinsic test set")
    parser.add_argument("--weight_exponent", type=int, default=0, help="Exponent to raise inverse distances to, in the loss function for intrinsic evaluation")
    parser.add_argument("--positional_or_angular", type=str, default="angular", help="Whether to use positional or angular loss for intrinsic evaluation")

    args = parser.parse_args()
    logger.info(f"Args: {args}")
    
    projection_head = nn.Linear(args.backbone_model_output_dim, args.target_dim).to("cuda")

    print(projection_head)

    # Evaluate the model
    cache_path = os.path.join(
        EVALUATION_RESULTS_PATH,
        "random_projection",
        args.backbone_model.replace("/", "__"),
    )

    model_name = os.path.join(
        f"{args.backbone_model}"
        f"_distilled_{args.target_dim}"
        "_random_projection"
    )

    if args.eval_intrinsic:
        projection_head.eval()
        loss = eval_intrinsic(
            projection=projection_head,
            backbone_model_path=args.backbone_model,
            positional_or_angular=args.positional_or_angular,
            weight_exponent=args.weight_exponent,
            cache_path=cache_path,
            model_name=model_name
        )
        logger.info(f"Intrinsic test loss: {loss}")
        sys.exit(0)

    model_name = os.path.join(
        f"{args.backbone_model}"
        f"_distilled_{args.target_dim}"
        "_random_projection"
    )

    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name,
        normalize_vector_before_projecting=args.normalize_vector_before_projecting
    )

    custom_model.eval()


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
        