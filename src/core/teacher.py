import torch
import numpy as np
from utils.embed_functions import embed_points_isometric


class Teacher:
    def __init__(self, backbone):
        self.backbone = backbone
        self.backbone.eval()

    def get_targets(self, sentences):
        """Get target embeddings using isometric embedding"""
        with torch.no_grad():
            sentence_embeddings = self.backbone.encode(sentences, convert_to_tensor=True)
            embeddings_np = sentence_embeddings.cpu().numpy()
            zero_vector = np.zeros((1, embeddings_np.shape[1]))
            embeddings_np = np.vstack([zero_vector, embeddings_np])

            target_embeddings_np = embed_points_isometric(embeddings_np)
            target_embeddings = torch.from_numpy(target_embeddings_np).float().to(sentence_embeddings.device)
            target_embeddings = target_embeddings.clone().detach()

            return target_embeddings
        