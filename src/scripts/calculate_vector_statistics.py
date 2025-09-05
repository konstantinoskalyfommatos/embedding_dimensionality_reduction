import torch
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_PATH = os.getenv("PROJECT_ROOT")
embeddings_path = os.path.join(
    ROOT_PATH, 
    "storage/precalculated_embeddings/c4/jinaai_jina-embeddings-v2-small-en/train_embeddings.pt"
)

# Load the embeddings
embeddings = torch.load(embeddings_path)

print(f"Total number of vectors: {embeddings.shape[0]}")

# Ensure embeddings is a tensor
if not isinstance(embeddings, torch.Tensor):
    raise ValueError("Loaded embeddings are not a torch.Tensor.")

# Check for negative values in any dimension for each vector
has_negative = (embeddings < 0).all(dim=1)

# Calculate the average number of negative values per vector
average_negatives_per_vector = (embeddings < 0).sum(dim=1).float().mean().item()
print(f"Average number of negative values per vector: {average_negatives_per_vector}")

# Count how many vectors have at least one negative value
num_vectors_with_negative = has_negative.sum().item()

print(f"Number of vectors with all negative values: {num_vectors_with_negative}")