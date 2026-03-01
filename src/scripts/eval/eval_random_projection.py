from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging
import sys

from utils.config import EVALUATION_RESULTS_PATH
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import eval_intrinsic, evaluate_mteb


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
    parser.add_argument("--retrieval_batch_size", type=int, default=20, help="Batch size for retrieval evaluation")
    parser.add_argument("--clustering_batch_size", type=int, default=16, help="Batch size for clustering evaluation")

    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite MTEB evaluation cache results")
    parser.add_argument("--spearman_test_batch_size", type=int, default=5000, help="Batch size for intrinsic Spearman evaluation")

    parser.add_argument("--intrinsic", action="store_true", help="Evaluate only on the intrinsic test set")

    args = parser.parse_args()
    logger.info(f"Args: {args}")
    
    projection_head = nn.Linear(
        args.backbone_model_output_dim, 
        args.target_dim,
        bias=False
    ).to("cuda")
    nn.init.normal_(projection_head.weight, mean=0, std=1)

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

    if args.intrinsic:
        logger.info("Evaluating random projection on intrinsic test set")
        projection_head.eval()
        eval_intrinsic(
            projection=projection_head,
            backbone_model_path=args.backbone_model,
            checkpoint=None,
            cache_path=cache_path,
            model_name=model_name,
            spearman_test_batch_size=args.spearman_test_batch_size
        )
        sys.exit(0)

    logger.info("Evaluating random projection on MTEB benchmark")
    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name,
    )

    evaluate_mteb(
        model=custom_model,
        cache_path=cache_path,
        model_name=model_name,
        sts_batch_size=args.sts_batch_size,
        classification_batch_size=args.classification_batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        clustering_batch_size=args.clustering_batch_size,
        skip_sts=args.skip_sts,
        skip_classification=args.skip_classification,
        skip_retrieval=args.skip_retrieval,
        skip_clustering=args.skip_clustering,
        fast_mode=args.fast_mode,
        overwrite_cache=args.overwrite_cache,
    )
        