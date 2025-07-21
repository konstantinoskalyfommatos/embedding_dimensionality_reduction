import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def eval_intrinsic(
    student,
    teacher,
    val_loader: DataLoader,
    device: str = "cuda",
) -> float:
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

            # Add zero vector to student embeddings
            zero_vector = torch.zeros((1, student_emb.shape[1]), device=student_emb.device)
            student_emb = torch.cat([zero_vector, student_emb], dim=0)

            teacher_emb = teacher.get_targets(input_ids, attention_mask)

            student_emb = F.normalize(student_emb, p=2, dim=1)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=1)

            cosine_sim = F.cosine_similarity(student_emb, teacher_emb, dim=1)
            loss = (1 - cosine_sim).mean()

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss
