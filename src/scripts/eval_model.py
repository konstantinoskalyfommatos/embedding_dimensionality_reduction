from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging
import json

from utils.config import TRAINED_MODELS_PATH, EVALUATION_RESULTS_PATH
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import evaluate_sts, evaluate_retrieval, evaluate_classification, evaluate_clustering


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_best_safetensors(model_dir: str):
    """Returns the best model checkpoint safetensors based on eval_loss."""
    if "checkpoint" in model_dir:
        last_checkpoint = model_dir
    else:
        last_checkpoint = os.path.join(
            model_dir,
            sorted(os.listdir(model_dir))[-1]
        )

    last_state_filepath = os.path.join(last_checkpoint, "trainer_state.json")
    with open(last_state_filepath) as f:
        last_state = json.load(f)

    best_metric = last_state["best_metric"]
    best_model_checkpoint = last_state["best_model_checkpoint"]

    logger.info(
        f"Loading safetensors from checkpoint: {best_model_checkpoint} "
        f"with best metric: {best_metric}"
    )
    return os.path.join(
            best_model_checkpoint,
            "model.safetensors"
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a distilled SentenceTransformer model on STSBenchmark")
    parser.add_argument(
        "--backbone_model", type=str, 
        help="Name or path of the backbone SentenceTransformer model",
        default="jinaai/jina-embeddings-v2-small-en"
    )
    parser.add_argument(
        "--backbone_model_output_dim",
        default=512
    )
    parser.add_argument("--target_dim", type=int, default=32, help="Target dimension of the distilled embeddings")
    parser.add_argument("--positional_loss_factor", type=float, default=1.0, help="Weight for positional loss used during training")
    parser.add_argument("--train_batch_size", type=int, help="Batch size used for training")
    parser.add_argument("--model_saved_path", type=str, required=False, default=None, help="Custom path where the model was saved at")
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")
    parser.add_argument("--normalize_vector_before_projecting", action="store_true")
    parser.add_argument("--fast_mode", action="store_true")

    args = parser.parse_args()
    logger.info(f"Args: {args}")

    projection_head = nn.Sequential(
        nn.Linear(args.backbone_model_output_dim, args.target_dim),
        nn.ReLU(),
    )

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

    best_safetenors_path = get_best_safetensors(trained_path)

    # NOTE: MTEB expect the model name to be in the format company/model_name
    model_name = trained_path.split("/checkpoint")[0].split("/")[-1].replace("__", "/")

    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name,
        normalize_vector_before_projecting=args.normalize_vector_before_projecting
    )
    custom_model.load_checkpoint(best_safetenors_path)

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
            fast_mode=args.fast_mode
        )
        logger.info(f"Final Spearman correlation on STS test set: {sts_score:.4f}")

    if not args.skip_retrieval:
        retrieval_score = evaluate_retrieval(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode
        )
        logger.info(f"Final retrieval results: {retrieval_score}")

    if not args.skip_clustering:
        clustering_score = evaluate_clustering(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode
        )
        logger.info(f"Final clustering results: {clustering_score}")

    if not args.skip_classification:
        classification_score = evaluate_classification(
            model=custom_model,
            cache_path=cache_path,
            fast_mode=args.fast_mode
        )
        logger.info(f"Final classification results: {classification_score}")
        