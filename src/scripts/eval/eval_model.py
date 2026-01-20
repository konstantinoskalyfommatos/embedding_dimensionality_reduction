from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging
import sys
from safetensors.torch import load_file

from utils.config import TRAINED_MODELS_PATH, EVALUATION_RESULTS_PATH
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import (
    evaluate_sts, evaluate_retrieval, 
    evaluate_classification, evaluate_clustering,
    eval_intrinsic
)


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
        default=768,
        type=int
    )
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to use")
    parser.add_argument("--target_dim", type=int, default=32, help="Target dimension of the distilled embeddings")
    parser.add_argument("--positional_loss_factor", type=float, default=1.0, help="Weight for positional loss used during training")
    parser.add_argument("--train_batch_size", type=int, help="Batch size used for training", default=20000)
    parser.add_argument("--model_saved_path", type=str, required=False, default=None, help="Custom path where the model was saved at")
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite MTEB evaluation cache results")
    parser.add_argument("--normalize_vector_before_projecting", action="store_true")
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--sts_batch_size", type=int, default=2048, help="Batch size for STS evaluation")
    parser.add_argument("--retrieval_batch_size", type=int, default=48, help="Batch size for retrieval evaluation")
    parser.add_argument("--classification_batch_size", type=int, default=20, help="Batch size for classification evaluation")
    parser.add_argument("--clustering_batch_size", type=int, default=16, help="Batch size for clustering evaluation")
    
    parser.add_argument("--eval_intrinsic", action="store_true", help="Evaluate only on the intrinsic test set")
    parser.add_argument("--positional_or_angular", type=str, default="positional", help="Whether to use positional or angular loss for intrinsic evaluation")
    
    args = parser.parse_args()
    logger.info(f"Args: {args}")

    projection_head = nn.Sequential(
        nn.Linear(args.backbone_model_output_dim, args.target_dim),
        nn.ReLU(),
    ).to("cuda")

    print(projection_head)

    if args.model_saved_path:
        trained_path = args.model_saved_path
        # For MTEB
        if not args.model_saved_path.endswith("/"):
            args.model_saved_path += "/"
    else:
        trained_path = os.path.join(
            TRAINED_MODELS_PATH,
            args.backbone_model.replace("/", "__"),
            f"{args.backbone_model.replace("/", "__")}"
            f"_distilled_{args.target_dim}"
            f"_batch_{args.train_batch_size}"
            f"_poslossfactor_{float(args.positional_loss_factor)}"
        )

    if args.eval_intrinsic:
        projection_head.eval()
        sorted_checkpoints = sorted(
            os.listdir(trained_path),
            key=lambda x: int(x.split("checkpoint-")[-1])
        )
        best_checkpoint = sorted_checkpoints[0]
        best_loss = float('inf')
        for checkpoint in sorted_checkpoints:
            # Load trained weights
            projection_head.load_state_dict(
                load_file(
                    os.path.join(
                        trained_path, 
                        str(checkpoint), 
                        "model.safetensors"
                    )
                )
            )

            with torch.no_grad():
                loss = eval_intrinsic(
                    projection=projection_head,
                    backbone_model_path=args.backbone_model
                )
            if loss < best_loss:
                best_loss = loss
                best_checkpoint = checkpoint
            logger.info(f"Intrinsic test loss at {checkpoint}: {loss}")
            torch.cuda.empty_cache()
        logger.info(f"Best checkpoint: {best_checkpoint} with loss: {best_loss}")
        sys.exit(0)

    if args.checkpoint:
        checkpoint_to_use = f"checkpoint-{args.checkpoint}"
    else:
        last_checkpoint = max([int(c.split("checkpoint-")[-1]) for c in os.listdir(trained_path)])
        checkpoint_to_use = f"checkpoint-{last_checkpoint}"
    logger.info(f"Using checkpoint: {checkpoint_to_use}")

    # NOTE: MTEB expect the model name to be in the format company/model_name
    model_name = trained_path.split("/checkpoint")[0].split("/")[-1].replace("__", "/")

    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name,
        normalize_vector_before_projecting=args.normalize_vector_before_projecting
    )

    custom_model.load_checkpoint(
        os.path.join(
            trained_path, 
            str(checkpoint_to_use), 
            "model.safetensors"
        )
    )
    custom_model.eval()

    # Evaluate the model
    if "checkpoint" in trained_path:
        trained_path = trained_path.split("/checkpoint")[0]

    cache_path = os.path.join(
        EVALUATION_RESULTS_PATH,
        "trained_models",
        trained_path.split("models/")[-1],
    )

    if not args.skip_sts:
        sts_score = evaluate_sts(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.sts_batch_size,
            overwrite_cache=args.overwrite_cache
        )
        logger.info(f"Final Spearman correlation on STS test set: {sts_score:.4f}")

    if not args.skip_retrieval:
        retrieval_score = evaluate_retrieval(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.retrieval_batch_size,
            overwrite_cache=args.overwrite_cache
        )
        logger.info(f"Final retrieval results: {retrieval_score}")

    if not args.skip_classification:
        classification_score = evaluate_classification(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.classification_batch_size,
            overwrite_cache=args.overwrite_cache
        )
        logger.info(f"Final classification results: {classification_score}")

    if not args.skip_clustering:
        clustering_score = evaluate_clustering(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode,
            batch_size=args.clustering_batch_size,
            overwrite_cache=args.overwrite_cache
        )
        logger.info(f"Final clustering results: {clustering_score}")
