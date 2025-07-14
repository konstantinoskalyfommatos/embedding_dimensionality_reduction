"""Contains the class that will be used to distill models."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cos_sim
from utils.embed_functions import embed_points_isometric


class DistilledJina(nn.Module):
    def __init__(
        self, 
        backbone_model,
        low_dim_size: int,
        hidden_size=368,
        backbone_dim=512,
        finetune_backbone=False,
        device="cuda"
    ):
        super(DistilledJina, self).__init__()
        self.student_backbone = backbone_model
        self.teacher_backbone = copy.deepcopy(backbone_model)
        self.low_dim_size = low_dim_size
        self.backbone_dim = backbone_dim
        self.finetune_backbone = finetune_backbone
        
        self.projection_net = nn.Sequential(
            nn.Linear(backbone_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, low_dim_size)
        )
        
        self.device = device
        self.to(self.device)

        if not self.finetune_backbone:
            for param in self.student_backbone.parameters():
                param.requires_grad = False

        self.teacher_backbone.eval()

    def forward(self, sentences):
        sentence_embeddings = self.student_backbone.encode(sentences, convert_to_tensor=True)
        zero_vector = torch.zeros((1, self.backbone_dim), device=sentence_embeddings.device)
        sentence_embeddings = torch.cat([zero_vector, sentence_embeddings], dim=0)

        if not self.finetune_backbone:
            sentence_embeddings = sentence_embeddings.clone().detach().requires_grad_(True)
            
        low_dim_embeddings = self.projection_net(sentence_embeddings)
        
        if self.training:
            return low_dim_embeddings
        else:
            return low_dim_embeddings[1:] 

    def get_targets(self, sentences):
        """Get target embeddings using isometric embedding"""
        with torch.no_grad():
            sentence_embeddings = self.teacher_backbone.encode(sentences, convert_to_tensor=True)
            embeddings_np = sentence_embeddings.cpu().numpy()
            zero_vector = np.zeros((1, embeddings_np.shape[1]))
            embeddings_np = np.vstack([zero_vector, embeddings_np])

            target_embeddings_np = embed_points_isometric(embeddings_np)
            target_embeddings = torch.from_numpy(target_embeddings_np).float().to(sentence_embeddings.device)
            target_embeddings = target_embeddings.clone().detach()

            return target_embeddings
            
    def compute_loss(self, sentences, alpha=0.1):
        student_predictions = self.forward(sentences)
        isometric_targets = self.get_targets(sentences)

        # Positioning
        loss_position = nn.MSELoss()(student_predictions, isometric_targets)

        # Cosine similarity
        student_sim_matrix = cos_sim(student_predictions, student_predictions)
        target_sim_matrix = cos_sim(isometric_targets, isometric_targets)

        loss_similarity = nn.MSELoss()(student_sim_matrix, target_sim_matrix)

        total_loss = alpha * loss_position + (1 - alpha) * loss_similarity
        
        return total_loss