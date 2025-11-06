from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import torch
import logging

from utils.eval import evaluate_sts, evaluate_retrieval, evaluate_classification, evaluate_clustering


# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a distilled SentenceTransformer model on Benchmarks")
    parser.add_argument(
        "--backbone_model_path", type=str, 
        help="Name or path of the backbone SentenceTransformer model",
        default="jinaai/jina-embeddings-v2-small-en"
    )

    args = parser.parse_args()

    model = SentenceTransformer(args.backbone_model_path, device="cuda", trust_remote_code=True)
    model.eval()

    # Evaluate the model
    sts_score = evaluate_sts(model, model_name=f"{args.backbone_model_path.replace('/', '-')}", languages=None)
    logger.info(f"Final Spearman correlation on STS test set: {sts_score:.4f}")

    retrieval_score = evaluate_retrieval(model, model_name=f"{args.backbone_model_path.replace('/', '-')}")
    logger.info(f"Final retrieval results: {retrieval_score}")

    classification_score = evaluate_classification(model, model_name=f"{args.backbone_model_path.replace('/', '-')}")
    logger.info(f"Final classification results: {classification_score}")

    clustering_score = evaluate_clustering(model, model_name=f"{args.backbone_model_path.replace('/', '-')}")
    logger.info(f"Final clustering results: {clustering_score}")
