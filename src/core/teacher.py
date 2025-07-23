import torch
from utils.embed_functions import embed_points_isometric


class Teacher:
    def __init__(self, backbone, device="cuda", use_backbone: bool = True):
        if use_backbone:
            self.backbone = backbone
            self.device = device
            self.backbone.to(self.device)

            for param in self.backbone.parameters():
                param.requires_grad = False

            self.backbone.eval()

    def get_targets(self, input_ids, attention_mask):
        """Get target embeddings using isometric embedding"""
        assert hasattr(self, 'backbone'), "Teacher backbone is not set."
        
        with torch.no_grad():
            backbone_output = self.backbone(
                {
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask
                }
            )
            embeddings = backbone_output["sentence_embedding"]
            zero_vector = torch.zeros((1, embeddings.shape[1]), device=embeddings.device)
            embeddings = torch.cat([zero_vector, embeddings], dim=0)

            target_embeddings = embed_points_isometric(embeddings)
            target_embeddings = target_embeddings.clone().detach()

            return target_embeddings
        
    def get_targets_from_precalculated_embeddings(self, embeddings, keep_zero_vector=True):
        """Get target embeddings from pre-calculated embeddings"""
        with torch.no_grad():
            zero_vector = torch.zeros((1, embeddings.shape[1]), device=embeddings.device)
            embeddings = torch.cat([zero_vector, embeddings], dim=0)

            target_embeddings = embed_points_isometric(embeddings)
            target_embeddings = target_embeddings.clone().detach()

            if not keep_zero_vector:
                target_embeddings = target_embeddings[1:]

            return target_embeddings
        