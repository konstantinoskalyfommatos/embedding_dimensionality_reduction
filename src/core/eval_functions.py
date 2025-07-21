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
    teacher.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            student_emb = student(batch)
            teacher_emb = teacher.backbone(batch)

            student_emb = F.normalize(student_emb, p=2, dim=1)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=1)

            cosine_sim = F.cosine_similarity(student_emb, teacher_emb, dim=1)
            loss = (1 - cosine_sim).mean()

            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss
