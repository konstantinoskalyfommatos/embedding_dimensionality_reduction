import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cos_sim
from core.teacher import Teacher


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

        zero_vector = torch.zeros((1, embeddings.shape[1]), device=embeddings.device)
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
            "Cannot use pre-calculated embeddings when the student's backbone is being finetuned, "
            "as the embeddings will change during training."
        )

        zero_vector = torch.zeros((1, embeddings.shape[1]), device=embeddings.device)
        embeddings = torch.cat([zero_vector, embeddings], dim=0)
        
        if self.training or keep_zero_vector:
            return self.projection_net(embeddings)
        else:
            with torch.no_grad():
                low_dim_embeddings = self.projection_net(embeddings)
            return low_dim_embeddings[1:]

    def compute_loss(self, student_predictions, teacher_targets, alpha=0.3):
        # Positioning
        loss_position = nn.MSELoss()(student_predictions, teacher_targets)

        # Cosine similarity
        student_sim_matrix = cos_sim(student_predictions, student_predictions)
        target_sim_matrix = cos_sim(teacher_targets, teacher_targets)

        loss_similarity = nn.MSELoss()(student_sim_matrix, target_sim_matrix)

        total_loss = alpha * loss_position + (1 - alpha) * loss_similarity
        return total_loss