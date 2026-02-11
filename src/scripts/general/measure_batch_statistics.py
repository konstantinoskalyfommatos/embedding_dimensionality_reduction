import torch
import os
import argparse

from utils.config import PROJECT_ROOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v2-small-en", help="Others: Alibaba-NLP/gte-multilingual-base")

    args = parser.parse_args()
    
    test_tensor = os.path.join(
        PROJECT_ROOT, 
        "storage", 
        "precalculated_embeddings", 
        "sentence-paraphrases", 
        args.model_name.replace("/", "__"), 
        "test_embeddings.pt"
    )

    # Calculate mean L2 norm and standard deviation of pairwise differences
    embeddings = torch.load(test_tensor).to("cpu")
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    # Exclude diagonal (self-pairs)
    mask = ~torch.eye(pairwise_distances.size(0), dtype=torch.bool, device=pairwise_distances.device)
    pairwise_distances_no_diag = pairwise_distances[mask]
    mean_distance = pairwise_distances_no_diag.mean().item()
    std_distance = pairwise_distances_no_diag.std().item()

    print(f"Test - Mean pairwise distance: {mean_distance:.4f}")
    print(f"Test - Std of pairwise distances: {std_distance:.4f}")

    # Calculate the mean cosinesimilarity and standard deviation of pairwise cosine similarities
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    pairwise_cosine_similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
    # Exclude diagonal (self-pairs)
    mask = ~torch.eye(pairwise_cosine_similarities.size(0), dtype=torch.bool, device=pairwise_cosine_similarities.device)
    pairwise_cosine_similarities_no_diag = pairwise_cosine_similarities[mask]
    mean_cosine_similarity = pairwise_cosine_similarities_no_diag.mean().item()
    std_cosine_similarity = pairwise_cosine_similarities_no_diag.std().item()

    print(f"Test - Mean pairwise cosine similarity: {mean_cosine_similarity:.4f}")
    print(f"Test - Std of pairwise cosine similarities: {std_cosine_similarity:.4f}")


    train_tensor = os.path.join(
        PROJECT_ROOT, 
        "storage", 
        "precalculated_embeddings", 
        "c4", 
        args.model_name.replace("/", "__"), 
        "train_embeddings.pt"
    )
    # Use the first 20000 embeddings
    embeddings = torch.load(train_tensor).to("cpu")[:20000]

    # Calculate mean L2 norm and standard deviation of pairwise differences
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    # Exclude diagonal (self-pairs)
    mask = ~torch.eye(pairwise_distances.size(0), dtype=torch.bool, device=pairwise_distances.device)
    pairwise_distances_no_diag = pairwise_distances[mask]
    mean_distance = pairwise_distances_no_diag.mean().item()
    std_distance = pairwise_distances_no_diag.std().item()

    print(f"Mean pairwise distance: {mean_distance:.4f}")
    print(f"Std of pairwise distances: {std_distance:.4f}")

    # Calculate the mean cosinesimilarity and standard deviation of pairwise cosine similarities
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    pairwise_cosine_similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
    # Exclude diagonal (self-pairs)
    mask = ~torch.eye(pairwise_cosine_similarities.size(0), dtype=torch.bool, device=pairwise_cosine_similarities.device)
    pairwise_cosine_similarities_no_diag = pairwise_cosine_similarities[mask]
    mean_cosine_similarity = pairwise_cosine_similarities_no_diag.mean().item()
    std_cosine_similarity = pairwise_cosine_similarities_no_diag.std().item()

    print(f"Mean pairwise cosine similarity: {mean_cosine_similarity:.4f}")
    print(f"Std of pairwise cosine similarities: {std_cosine_similarity:.4f}")

    # Find the smallest pairwise distance and the largest pairwise cosine similarity in the train set
    min_distance = pairwise_distances_no_diag.min().item()
    print(f"Minimum pairwise distance: {min_distance:.4f}")

    max_cosine_similarity = pairwise_cosine_similarities_no_diag.max().item()
    print(f"Maximum pairwise cosine similarity: {max_cosine_similarity:.4f}")
