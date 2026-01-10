from torch.utils.data import Dataset
import torch
import os
import numpy as np

from utils.config import PROJECT_ROOT


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class RandomVectorsDataset(Dataset):
    def __init__(
        self, 
        dim: int, 
        len: int,
        dtype: torch.dtype,
        low: float = 0,
        high: float = 1,
    ):
        np.random.seed(42)

        self.dim = dim
        self.len = len
        vectors = torch.from_numpy(
            np.random.uniform(
                low=low, 
                high=high, 
                size=(len, dim)
            )
        ).to(dtype)
        random_matrix = torch.randn(dim, dim, dtype=dtype)
        q, r = torch.linalg.qr(random_matrix)

        # Rotate vectors
        self.vectors = torch.matmul(vectors, q)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.vectors[idx]
        

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
        model_name.replace("/", "__"),
        f"{split}_embeddings.pt"
    )
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Precalculated embeddings not found at {output_path}")

    embeddings = torch.load(output_path)
    return EmbeddingsDataset(embeddings)
