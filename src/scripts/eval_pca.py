from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging

from utils.config import EVALUATION_RESULTS_PATH, PROJECT_ROOT
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import evaluate_sts, evaluate_retrieval, evaluate_classification, evaluate_clustering


torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAProjection(nn.Module):
    def __init__(self, mean_vector, projection_matrix):
        super().__init__()
        self.register_buffer('mean_vector', mean_vector)
        self.register_buffer('projection_matrix', projection_matrix)
    
    def forward(self, x):
        return (x - self.mean_vector) @ self.projection_matrix.T
    

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
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")
    parser.add_argument("--fast_mode", action="store_true")

    args = parser.parse_args()
    logger.info(f"Args: {args}")

    pca_matrix_path = os.path.join(
        PROJECT_ROOT,
        "storage",
        "pca",
        args.backbone_model.replace("/", "__"),
        str(args.target_dim),
    )
    
    pca_file = [f for f in os.listdir(pca_matrix_path) if f.endswith(".pt")][0]
    pca_filepath = os.path.join(pca_matrix_path, pca_file)
    
    pca_state = torch.load(pca_filepath)
    projection_matrix = pca_state["components"]  # Shape: (target_dim, backbone_dim)
    mean_vector = pca_state["mean"]  # Shape: (backbone_dim,)
    
    projection_head = PCAProjection(mean_vector, projection_matrix)
    print(projection_head)

    model_name = os.path.join(
        f"{args.backbone_model}"
        f"_distilled_{args.target_dim}"
        "_pca"
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
        "pca_projection",
        args.backbone_model.replace("/", "__"),
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
