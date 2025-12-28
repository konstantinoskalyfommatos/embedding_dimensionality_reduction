"""Fits and saves a PCA projection using a subset of embeddings."""

import os
import argparse
import numpy as np
import torch

from sklearn.decomposition import PCA

from utils.custom_datasets import get_precalculated_embeddings_dataset
from utils.config import PROJECT_ROOT


def fit_pca_and_save(
    dataset_path: str,
    model_name: str,
    split: str,
    target_dim: int,
    max_samples: int,
    output_path: str,
    seed: int = 42,
):
    """Fits PCA and saves components + mean as a .pt file."""

    assert max_samples > 0, "max_samples must be positive"
    os.makedirs(output_path, exist_ok=True)

    dataset = get_precalculated_embeddings_dataset(
        dataset_path=dataset_path,
        model_name=model_name,
        split=split,
    )

    n_samples = min(max_samples, len(dataset))
    print(f"Loading {n_samples} embeddings")

    embeddings = np.stack(
        [dataset[i].cpu().numpy() for i in range(n_samples)],
        axis=0,
    )

    print(f"Fitting PCA on matrix of shape {embeddings.shape}")

    pca = PCA(
        n_components=target_dim,
        svd_solver="auto",
        random_state=seed,
    )
    pca.fit(embeddings)

    pca_state = {
        "components": torch.tensor(pca.components_, dtype=torch.float32),  # (k, d)
        "mean": torch.tensor(pca.mean_, dtype=torch.float32),               # (d,)
        "explained_variance_ratio": torch.tensor(
            pca.explained_variance_ratio_, dtype=torch.float32
        ),
    }

    output_file = os.path.join(
        output_path,
        f"pca_{target_dim}_samples_{n_samples}.pt",
    )

    torch.save(pca_state, output_file)

    print(f"PCA saved to: {output_file}")
    print(f"Components shape: {pca_state['components'].shape}")
    print(f"Mean shape: {pca_state['mean'].shape}")
    print(
        f"Explained variance ratio sum: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fit PCA and save projection")

    parser.add_argument("--dataset_path", type=str, default="allenai/c4")
    parser.add_argument(
        "--model_name",
        type=str,
        default="jinaai/jina-embeddings-v2-small-en",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--target_dim", type=int, default=32)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=300_000,
        help="Number of embeddings to use for PCA",
    )

    args = parser.parse_args()

    fit_pca_and_save(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        split=args.split,
        target_dim=args.target_dim,
        max_samples=args.max_samples,
        output_path=os.path.join(
            PROJECT_ROOT,
            "storage",
            "pca",
            args.model_name.replace("/", "__"),
            str(args.target_dim),
        ),
    )


if __name__ == "__main__":
    main()
