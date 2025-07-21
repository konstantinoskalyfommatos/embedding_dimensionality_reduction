import torch
import numpy as np
from utils.embed_functions import embed_points_isometric


class Teacher:
    def __init__(self, backbone, device='cuda'):
        self.backbone = backbone
        self.device = device
        self.backbone.to(self.device)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

    def get_targets(self, input_ids, attention_mask):
        """Get target embeddings using isometric embedding"""
        with torch.no_grad():
            backbone_output = self.backbone(
                {
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask
                }
            )
            embeddings = backbone_output["sentence_embedding"]
            embeddings_np = embeddings.cpu().numpy()

            zero_vector = np.zeros((1, embeddings_np.shape[1]))
            
            embeddings_np = np.vstack([zero_vector, embeddings_np])

            target_embeddings_np = embed_points_isometric(embeddings_np)
            target_embeddings = torch.from_numpy(target_embeddings_np).float().to(embeddings.device)
            target_embeddings = target_embeddings.clone().detach()

            return target_embeddings
        