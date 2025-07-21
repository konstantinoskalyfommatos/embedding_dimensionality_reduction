import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cos_sim
from core.teacher import Teacher


class Student(nn.Module):
    def __init__(
        self, 
        backbone, 
        projection_net, 
        finetune_backbone=False,
        device='cuda'
    ):
        super(Student, self).__init__()
        self.backbone = backbone
        self.projection_net = projection_net
        self.finetune_backbone = finetune_backbone

        if not self.finetune_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.device = device
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        backbone_output = self.backbone(
            {
                "input_ids": input_ids, 
                "attention_mask": attention_mask
            }
        )
        embeddings = backbone_output["sentence_embedding"]

        zero_vector = torch.zeros((1, embeddings.shape[1]), device=embeddings.device)
        embeddings = torch.cat([zero_vector, embeddings], dim=0)
        if not self.finetune_backbone:
            embeddings = embeddings.clone().detach().requires_grad_(True)
        return self.projection_net(embeddings)

    def compute_loss(self, input_ids, attention_mask, teacher: Teacher, alpha=0.1):
        student_predictions = self.forward(input_ids, attention_mask)
        isometric_targets = teacher.get_targets(input_ids, attention_mask)

        # Positioning
        loss_position = nn.MSELoss()(student_predictions, isometric_targets)

        # Cosine similarity
        student_sim_matrix = cos_sim(student_predictions, student_predictions)
        target_sim_matrix = cos_sim(isometric_targets, isometric_targets)

        loss_similarity = nn.MSELoss()(student_sim_matrix, target_sim_matrix)

        total_loss = alpha * loss_position + (1 - alpha) * loss_similarity
        return total_loss