from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging
import json

from utils.config import PROJECT_ROOT
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import evaluate_sts, evaluate_retrieval, evaluate_classification, evaluate_clustering


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_model(model_dir: str, custom_model: DistilledSentenceTransformer):
    """Load the best model checkpoint based on eval_loss."""
    
    last_checkpoint = sorted(os.listdir(model_dir))[-1]
    last_state_filepath = os.path.join(model_dir, last_checkpoint, "trainer_state.json")
    with open(last_state_filepath) as f:
        last_state = json.load(f)

    best_metric = last_state["best_metric"]
    best_model_checkpoint = last_state["best_model_checkpoint"]

    logger.info(
        f"Loading safetensors from checkpoint: {best_model_checkpoint} "
        f"with best metric: {best_metric}"
    )
    custom_model.load_checkpoint(
        os.path.join(
            best_model_checkpoint,
            "model.safetensors"
        )
    )
    return custom_model


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
    parser.add_argument("--use_random_projection", action="store_true", help="Use random projection head")
    parser.add_argument("--model_base_path", type=str, default="storage/models",
                       help="Path where the model was saved at. Will be appended to project root")
    args = parser.parse_args()
    logger.info(f"Args: {args}")
    
    if args.use_random_projection:
        projection_head = nn.Linear(512, args.target_dim)

    else:
        # projection_head = nn.Sequential(
        #     nn.Linear(args.backbone_model_output_dim, args.backbone_model_output_dim * 4),
        #     nn.ReLU(),
        #     nn.Linear(args.backbone_model_output_dim * 4, args.target_dim)
        # )
        projection_head = nn.Sequential(
            nn.Linear(args.backbone_model_output_dim, args.target_dim),
            nn.ReLU(),
        )

    print(projection_head)

    if args.use_random_projection:
        model_name = os.path.join(
            f"{args.backbone_model}"
            f"_distilled_{args.target_dim}"
            "_random"
        )
    else:
        trained_path = os.path.join(
            PROJECT_ROOT, 
            args.model_base_path,
            f"{args.backbone_model.replace("/", "__")}"
            f"_distilled_{args.target_dim}"
            f"_batch_{args.train_batch_size}"
            f"_poslossfactor_{float(args.positional_loss_factor)}"
        )
        # NOTE: MTEB expect the model name to be in the format company/model_name
        model_name = trained_path.split("storage/models/")[-1].replace("__", "/")

    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name
    )
    if not args.use_random_projection:
        custom_model = load_best_model(trained_path, custom_model)

    custom_model.eval()

    # Evaluate the model
    sts_score = evaluate_sts(custom_model)
    logger.info(f"Final Spearman correlation on STS test set: {sts_score:.4f}")

    classification_score = evaluate_classification(custom_model)
    logger.info(f"Final classification results: {classification_score}")

    retrieval_score = evaluate_retrieval(custom_model)
    logger.info(f"Final retrieval results: {retrieval_score}")

    clustering_score = evaluate_clustering(custom_model)
    logger.info(f"Final clustering results: {clustering_score}")
