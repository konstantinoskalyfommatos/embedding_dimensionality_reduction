import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def eval_intrinsic(
    student,
    teacher,
    val_loader: DataLoader,
    device: str = "cuda",
    alpha: float = 0.5,  # Weight for similarity loss
) -> float:
    """Returns the average combined loss (distance + cosine similarity).
    
    The loss is computed as a weighted sum of the MSE between the cosine similarity matrices
    and the MSE between the distance matrices of the student and teacher embeddings.
    """
    student.eval()
    # NOTE: Teacher is always in eval mode

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            student_emb = student(input_ids, attention_mask)
            teacher_emb = teacher.get_targets(input_ids, attention_mask)[1:]  # Skip the zero vector

            student_dist = torch.cdist(student_emb, student_emb, p=2)
            teacher_dist = torch.cdist(teacher_emb, teacher_emb, p=2)
            dist_loss = F.mse_loss(student_dist, teacher_dist)

            student_emb_norm = F.normalize(student_emb, p=2, dim=1)
            teacher_emb_norm = F.normalize(teacher_emb, p=2, dim=1)
            student_sim_matrix = torch.matmul(student_emb_norm, student_emb_norm.T)
            teacher_sim_matrix = torch.matmul(teacher_emb_norm, teacher_emb_norm.T)
            sim_loss = F.mse_loss(student_sim_matrix, teacher_sim_matrix)

            # Combined loss
            loss = alpha * sim_loss + (1 - alpha) * dist_loss

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss
