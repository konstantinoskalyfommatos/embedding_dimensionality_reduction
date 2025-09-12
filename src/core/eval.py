from sentence_transformers import util
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from argparse import ArgumentParser
import torch.nn as nn
import os
import torch
import json
import logging

from core.distilled_sentence_transformer import DistilledSentenceTransformer

# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_sts(
    model: SentenceTransformer, 
    split: str = "test",
    batch_size: int = 2048
) -> float:
    """
    Evaluate a SentenceTransformer model on the STSBenchmark English dataset.
    """
    # Load STS benchmark dataset from HuggingFace
    dataset = load_dataset("stsb_multi_mt", name="en", split=split)

    # Extract sentence pairs and labels
    sentences1 = list(dataset["sentence1"])
    sentences2 = list(dataset["sentence2"])
    labels = np.array(dataset["similarity_score"]).astype(float)

    with torch.no_grad():
        # Encode sentences
        embeddings1 = model.encode(sentences1, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy().diagonal()

    # Spearman correlation
    spearman_corr, _ = spearmanr(labels, cosine_scores)

    return spearman_corr


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

    args = parser.parse_args()

    model_path = None

    try:
        path = f"{args.trained_model_base_path}{args.target_dim}"
        last_path = sorted(os.listdir(path))[0]
        model_path = os.path.join(path, last_path, "model.safetensors")
        logger.info(f"Loading model from {model_path}")
    except FileNotFoundError:
        pass

    match args.target_dim:
        case 32:
            # projection_head = nn.Sequential(
            #     nn.Linear(512, 256),
            #     nn.GELU(),
            #     nn.Linear(256, 128),
            #     nn.GELU(),
            #     nn.Linear(128, 64),
            #     nn.GELU(),
            #     nn.Linear(64, 32),
            # )
            projection_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, 32),
            )
            # projection_head = nn.Sequential(
            #     nn.Linear(512, 32),
            # )
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

    custom_model = DistilledSentenceTransformer(
        model_name_or_path=args.backbone_model_path,
        projection=projection_head,
        output_dim=args.target_dim,
        freeze_backbone=True
    )
    if model_path:
        custom_model.load_checkpoint(model_path)

    # custom_model = SentenceTransformer(args.backbone_model_path, device="cuda", trust_remote_code=True)

    custom_model.eval()

    # Evaluate the model
    score = evaluate_sts(custom_model, split="test", batch_size=2048)
    logger.info(f"Final Spearman correlation on STS test set: {score:.4f}")