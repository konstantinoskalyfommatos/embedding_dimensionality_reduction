import torch
from torch.nn import functional as F
import torch.nn as nn


class Student(nn.Module):
    def __init__(
        self, 
        backbone, 
        projection_net, 
        freeze_backbone=True,
        device='cuda'
    ):
        super(Student, self).__init__()
        self.backbone = backbone
        self.projection_net = projection_net
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.device = device
        self.to(self.device)

    def forward(self, input_ids, attention_mask, keep_zero_vector=True):
        backbone_output = self.backbone(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )
        embeddings = backbone_output["sentence_embedding"]

        zero_vector = torch.zeros(
            (1, embeddings.shape[1]), 
            device=embeddings.device
        )
        embeddings = torch.cat([zero_vector, embeddings], dim=0)

        if self.freeze_backbone:
            embeddings = embeddings.clone().detach().requires_grad_(True)
            
        low_dim_embeddings = self.projection_net(embeddings)
        
        if self.training or keep_zero_vector:
            return low_dim_embeddings
        else:
            return low_dim_embeddings[1:]
        
    def forward_precalculated_embeddings(
        self, 
        embeddings, 
        keep_zero_vector=True
    ):
        """Forward pass using pre-calculated embeddings.
        
        Maps the embeddings to the low-dimensional space using the 
        projection network.
        """
        assert self.freeze_backbone, (
            "Cannot use pre-calculated embeddings when the student's backbone "
            "is being finetuned, as the embeddings will change during training."
        )

        zero_vector = torch.zeros(
            (1, embeddings.shape[1]), 
            device=embeddings.device
        )
        embeddings = torch.cat([zero_vector, embeddings], dim=0)
        
        low_dim_embeddings = self.projection_net(embeddings)

        if self.training or keep_zero_vector:
            return low_dim_embeddings
          
        return low_dim_embeddings[1:]
    
    def _is_zero_vector_in_embeddings(self, embeddings):
        """Check if the first vector in the embeddings is a zero vector."""
        return (
            embeddings.shape[0] > 0 and 
            torch.allclose(embeddings[0], torch.zeros_like(embeddings[0]))
        )

    def _compute_positional_loss(self, student_vectors, teacher_vectors):
        """Computes the pairwise distance preservation loss."""
        student_dist = torch.cdist(student_vectors, student_vectors, p=2)
        teacher_dist = torch.cdist(teacher_vectors, teacher_vectors, p=2)

        mask = torch.triu(torch.ones_like(student_dist), diagonal=1).bool()
            
        return nn.MSELoss()(student_dist[mask], teacher_dist[mask])
    
    def _compute_similarity_loss(self, student_vectors, teacher_vectors):
        """Computes the pairwise cosine similarity preservation loss"""
        student_vectors = F.normalize(student_vectors, dim=1)
        teacher_vectors = F.normalize(teacher_vectors, dim=1)

        student_sim_matrix = torch.matmul(student_vectors, student_vectors.T)
        teacher_sim_matrix = torch.matmul(teacher_vectors, teacher_vectors.T)

        # Mask to select only the upper triangle, excluding the diagonal
        mask = torch.triu(torch.ones_like(student_sim_matrix), diagonal=1).bool()

        return nn.MSELoss()(student_sim_matrix[mask], teacher_sim_matrix[mask])

    def compute_loss_fixed_weight(
        self, 
        student_vectors, 
        teacher_vectors, 
        positional_loss_factor=0.3
    ):
        similarity_loss = 0.0
        positional_loss = 0.0
        if positional_loss_factor > 0:
            positional_loss = self._compute_positional_loss(
                student_vectors, 
                teacher_vectors
            )
        if positional_loss_factor < 1:
            # Remove zero vector if it exists
            if self._is_zero_vector_in_embeddings(teacher_vectors):
                teacher_vectors = teacher_vectors[1:]
                student_vectors = student_vectors[1:]

            similarity_loss = self._compute_similarity_loss(
                student_vectors, 
                teacher_vectors
            )

        return (
            positional_loss_factor * positional_loss + 
            (1 - positional_loss_factor) * similarity_loss
        )
    
    def compute_loss_adaptive_weight(self, student_vectors, teacher_vectors):
        positional_loss = self._compute_positional_loss(
            student_vectors, 
            teacher_vectors
        )

        if self._is_zero_vector_in_embeddings(teacher_vectors):
            teacher_vectors = teacher_vectors[1:]
            student_vectors = student_vectors[1:]
        similarity_loss = self._compute_similarity_loss(
            student_vectors, 
            teacher_vectors
        )

        pos_weight = 1.0 / (positional_loss.detach() + 1e-8)
        sim_weight = 1.0 / (similarity_loss.detach() + 1e-8)
        
        total_weight = pos_weight + sim_weight
        positional_loss_factor_adaptive = pos_weight / total_weight
        
        return (
            positional_loss_factor_adaptive * positional_loss + 
            (1 - positional_loss_factor_adaptive) * similarity_loss
        )
    