import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()

from src.utils.custom_datasets import TokenizedDataset, EmbeddingsDataset
from utils.datasets_info import get_dataset_max_length


def calculate_embeddings(
    model: SentenceTransformer, 
    dataloader: DataLoader, 
    device="cuda"
):
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            embeddings = model(
                {
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask
                }
            )["sentence_embedding"]
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def get_precalculated_embeddings_dataset(
    dataset_path: str, 
    model_name: str,
    split: str, 
):
    output_path = os.path.join(
        os.getenv("PROJECT_ROOT"),
        "storage",
        "precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "_"),
        f"{split}_embeddings.pt"
    )
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Precalculated embeddings not found at {output_path}")

    embeddings = torch.load(output_path)
    return EmbeddingsDataset(embeddings)
