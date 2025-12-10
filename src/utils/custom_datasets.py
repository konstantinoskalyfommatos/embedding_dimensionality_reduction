from torch.utils.data import Dataset
import torch
import os

from utils.config import PROJECT_ROOT


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class MonteCarloEmbeddingsDataset(Dataset):
    """Dataset for Monte Carlo inference.
    
    Will be used after Monte Carlo inference, where pairs of \
    similar embeddings will be created for each sentence input.
    """
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings) // 2
    
    def __getitem__(self, idx):
        return (
            self.embeddings[2 * idx],
            self.embeddings[2 * idx + 1],
        )


def get_precalculated_embeddings_dataset(
    dataset_path: str, 
    model_name: str,
    split: str, 
):
    output_path = os.path.join(
        PROJECT_ROOT,
        "storage",
        "precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "_"),
        f"{split}_embeddings.pt"
    )
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Precalculated embeddings not found at {output_path}")

    embeddings = torch.load(output_path)
    return MonteCarloEmbeddingsDataset(embeddings)
