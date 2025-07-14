from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer
import torch
from src.scripts.distil_jina_v2 import DistilledModel
import os
from utils.datasets.datasets_info import get_dataset_max_length


# TODO: Define proper validation methods


# Load the backbone model first
backbone_model = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
backbone_model.max_seq_length = get_dataset_max_length("cl-nagoya/wikisplit-pp", tokenizer)

LOW_DIM_SIZE = 256
distilled_model = DistilledModel(backbone_model, low_dim_size=LOW_DIM_SIZE)

# Check if trained model exists and load it
distilled_model_path = "models/distilled_jina_v2.pth"
if os.path.exists(distilled_model_path):
    print("Loading trained model...")
    distilled_model.load_state_dict(torch.load(distilled_model_path, map_location="cuda"))
    model_status = "trained"
else:
    print("No trained model found. Using untrained model...")
    model_status = "untrained"

print("--- Testing Zero Vector Mapping ---")
with torch.no_grad():
    # Create zero vector and test projection
    zero_vector = torch.zeros((1, distilled_model.backbone_dim), device="cuda")
    projected_zero = distilled_model.projection_net(zero_vector)
    zero_norm = torch.norm(projected_zero).item()
    
    print(f"Zero vector norm after projection: {zero_norm:.8f}")

print()

# Set to evaluation mode
distilled_model.eval()

sentences = [
    "Today is a beautiful day.",
    "The current day is very beautiful.",
    "The weather is terrible today.",
    "I do not want to eat",
    "The capital of France is Paris.",
]

# Use the distilled model's forward method
with torch.no_grad():
    embeddings = distilled_model(sentences)

print(f"Model status: {model_status}")
print(f"Number of sentences: {len(sentences)}")
print(f"Embedding dimension: {embeddings.shape[1]}")

print("\n--- Pairwise Cosine Similarities ---")
with torch.no_grad():
    backbone_embeddings = backbone_model.encode(sentences, convert_to_tensor=True)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            # Calculate similarities for both models
            distilled_similarity = cos_sim(embeddings[i], embeddings[j])
            distilled_distance = torch.norm(embeddings[i] - embeddings[j]).item()
            backbone_similarity = cos_sim(backbone_embeddings[i], backbone_embeddings[j])
            backbone_distance = torch.norm(backbone_embeddings[i] - backbone_embeddings[j]).item()
            
            print(f"Sentence {i+1}: \"{sentences[i]}\"")
            print(f"Sentence {j+1}: \"{sentences[j]}\"")
            print(f"  Distilled model cosine similarity: {distilled_similarity.item():.4f}")
            print(f"  Backbone model cosine similarity: {backbone_similarity.item():.4f}")
            print(f"  Distilled model euclidean distance: {distilled_distance:.4f}")
            print(f"  Backbone model euclidean distance: {backbone_distance:.4f}")
            print()
    
print(f"Backbone embedding dimension: {backbone_embeddings.shape[1]}")
# Print the norm of the vectors for both models
print("\n--- Norm of Embeddings ---")
with torch.no_grad():
    distilled_norms = torch.norm(embeddings, dim=1)
    backbone_norms = torch.norm(backbone_embeddings, dim=1)
    
    for i in range(len(sentences)):
        print(f"  Distilled model norm: {distilled_norms[i].item():.4f}")
        print(f"  Backbone model norm: {backbone_norms[i].item():.4f}")
        print()
        