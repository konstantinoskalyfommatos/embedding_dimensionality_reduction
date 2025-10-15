from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging

from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import evaluate_sts, evaluate_retrieval


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a distilled SentenceTransformer model on STSBenchmark")
    parser.add_argument(
        "--backbone_model_path", type=str, 
        help="Name or path of the backbone SentenceTransformer model",
        default="jinaai/jina-embeddings-v2-small-en"
    )
    parser.add_argument(
        "--trained_model_base_path", type=str, 
        help="Path to the distilled model",
        default="storage/models/jina-embeddings-v2-small-en_distilled_"
    )
    parser.add_argument("--target_dim", type=int, default=32, help="Target dimension of the distilled embeddings")
    parser.add_argument("--use_random_projection", action="store_true", help="Use random projection head")

    args = parser.parse_args()

    logger.info(f"Args: {args}")

    model_path = None

    try:
        path = f"{args.trained_model_base_path}{args.target_dim}"
        last_path = sorted(os.listdir(path), key=lambda x: int(x.split('-')[-1]))[-1]
        model_path = os.path.join(path, last_path, "model.safetensors")
        logger.info(f"Loading model from {model_path}")
    except FileNotFoundError:
        pass

    if args.use_random_projection:
        projection_head = nn.Linear(512, args.target_dim)

    else:
        match args.target_dim:
            case 32:
                projection_head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.GELU(),
                    nn.Linear(128, 32),
                )
            case 16:
                projection_head = nn.Sequential(
                    nn.Linear(512, 64),
                    nn.GELU(),
                    nn.Linear(64, 16),
                )
            case 3:
                projection_head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.GELU(),
                    nn.Linear(128, 3),
                )
            case _:
                projection_head = nn.Linear(512, args.target_dim)

    print(projection_head)

    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model_path,
        projection=projection_head,
        output_dim=args.target_dim,
    )
    if model_path:
        custom_model.load_checkpoint(model_path)

    custom_model.eval()

    if args.use_random_projection:
        model_name = custom_model.model_name + "_random"
    else:
        model_name = custom_model.model_name

    # Evaluate the model
    sts_score = evaluate_sts(custom_model, model_name=model_name, batch_size=2048)
    logger.info(f"Final Spearman correlation on STS test set: {sts_score:.4f}")

    # retrieval_score = evaluate_retrieval(custom_model, model_name=model_name, batch_size=4)
    # logger.info(f"Final retrieval results: {retrieval_score}")
