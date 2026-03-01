from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging

from utils.config import TRAINED_MODELS_PATH, EVALUATION_RESULTS_PATH
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import evaluate_mteb, eval_intrinsic


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a distilled SentenceTransformer model on MTEB and intrinsic metrics")
    parser.add_argument("--backbone_model", type=str, default="jinaai/jina-embeddings-v2-small-en", help="Name or path of the backbone SentenceTransformer model")
    parser.add_argument("--backbone_model_output_dim", default=512, type=int)
    parser.add_argument("--target_dim", type=int, default=32, help="Target dimension of the distilled embeddings")
    parser.add_argument("--train_batch_size", type=int, help="Batch size used for training", default=20000)
    parser.add_argument("--positional_loss_factor", type=float, default=1, help="factor for positional vs similarity loss")
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")

    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite MTEB evaluation cache results")
    parser.add_argument("--fast_mode", action="store_true")

    parser.add_argument("--intrinsic", action="store_true", help="Intrinsic only")

    parser.add_argument("--sts_batch_size", type=int, default=2048, help="Batch size for STS evaluation")
    parser.add_argument("--retrieval_batch_size", type=int, default=6, help="Batch size for retrieval evaluation")
    parser.add_argument("--classification_batch_size", type=int, default=20, help="Batch size for classification evaluation")
    parser.add_argument("--clustering_batch_size", type=int, default=16, help="Batch size for clustering evaluation")
    parser.add_argument("--spearman_test_batch_size", type=int, default=5000, help="Batch size for intrinsic Spearman evaluation")

    parser.add_argument("--custom_suffix", type=str, default=None, help="Was added to the normal model name")
    parser.add_argument("--spearman", action="store_true", help="Differentiable Spearman correlation loss")

    args = parser.parse_args()
    logger.info(f"Args: {args}")

    projection_head = nn.Sequential(
        nn.Linear(args.backbone_model_output_dim, args.target_dim),
        nn.ReLU(),
    ).to("cuda")

    model_name = (
        f"{args.backbone_model}"
        f"_distilled_{args.target_dim}"
        f"_batch_{args.train_batch_size}"
        f"{'_poslossfactor_' + str(args.positional_loss_factor) if not args.spearman else ''}"
        f"{'_' + args.custom_suffix if args.custom_suffix else ''}"
    )

    trained_path = os.path.join(
        TRAINED_MODELS_PATH,
        args.backbone_model.replace("/", "__"),
        model_name.replace("/", "__"),
    )

    cache_path = os.path.join(
        EVALUATION_RESULTS_PATH,
        "trained_models",
        args.backbone_model.replace("/", "__"),
    )

    best_model_path = os.path.join(trained_path, "best_model.pt")
    logger.info(f"Loading best model from: {best_model_path}")
    projection_head.load_state_dict(torch.load(best_model_path, map_location="cuda"))
    projection_head.eval()

    if args.intrinsic:
        # Intrinsic evaluation
        logger.info("Evaluating intrinsic metrics on test set")
        eval_intrinsic(
            projection=projection_head,
            backbone_model_path=args.backbone_model,
            cache_path=cache_path,
            model_name=model_name,
            spearman_test_batch_size=args.spearman_test_batch_size,
        )
        exit()

    # MTEB evaluation
    logger.info("Evaluating on MTEB benchmark")
    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name,
    )
    custom_model.eval()

    evaluate_mteb(
        model=custom_model,
        cache_path=cache_path,
        sts_batch_size=args.sts_batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        classification_batch_size=args.classification_batch_size,
        clustering_batch_size=args.clustering_batch_size,
        skip_sts=args.skip_sts,
        skip_retrieval=args.skip_retrieval,
        skip_classification=args.skip_classification,
        skip_clustering=args.skip_clustering,
        fast_mode=args.fast_mode,
        overwrite_cache=args.overwrite_cache,
    )
