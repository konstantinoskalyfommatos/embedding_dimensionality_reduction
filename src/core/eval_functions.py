import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def eval_intrinsic(
    student,
    student_val_loader: DataLoader,
    teacher_val_loader: DataLoader,
    device: str = "cuda",
    alpha: float = 0.1, 
    use_precalculated_student_embeddings: bool = False,
) -> float:
    """Returns the average combined loss (distance + cosine similarity).
    
    The loss is computed as a weighted sum of the MSE between the cosine similarity matrices
    and the MSE between the distance matrices of the student and teacher embeddings.
    Only unique pairs (i < j) are considered to avoid redundant computations.
    """
    student.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for student_batch, teacher_batch in zip(student_val_loader, teacher_val_loader):
            if use_precalculated_student_embeddings:
                assert student.freeze_backbone, (
                    "Cannot use pre-calculated embeddings when the student's backbone is being finetuned, "
                    "as the embeddings will change during training."
                )
                student_emb = student.forward_precalculated_embeddings(
                    student_batch.to(device),
                    keep_zero_vector=False
                )
            else:       
                input_ids, attention_mask = student_batch
                student_emb = student(
                    input_ids.to(device),
                    attention_mask.to(device),
                    keep_zero_vector=False
                )

            teacher_emb = teacher_batch.to(device)

            n = student_emb.shape[0]
            # Get all unique pairs (i < j)
            idx = torch.triu_indices(n, n, offset=1, device=student_emb.device)
            idx_i, idx_j = idx[0], idx[1]

            # Cosine similarity for all unique pairs
            student_sim = F.cosine_similarity(student_emb[idx_i], student_emb[idx_j], dim=1)
            teacher_sim = F.cosine_similarity(teacher_emb[idx_i], teacher_emb[idx_j], dim=1)
            sim_loss = F.mse_loss(student_sim, teacher_sim)

            # Euclidean distance for all unique pairs
            student_dist = torch.norm(student_emb[idx_i] - student_emb[idx_j], dim=1)
            teacher_dist = torch.norm(teacher_emb[idx_i] - teacher_emb[idx_j], dim=1)
            dist_loss = F.mse_loss(student_dist, teacher_dist)

            # Combined loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            loss = alpha * sim_loss + (1 - alpha) * dist_loss

            batch_size = input_ids.size(0) if not use_precalculated_student_embeddings else student_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def eval_extrinsic(
    student,
    teacher,
    student_val_loader: DataLoader,
    teacher_val_loader: DataLoader,
    device: str = "cuda",
    alpha: float = 0.1,
    use_precalculated_student_embeddings: bool = False,
) -> float:
    """
    Evaluate the student model by comparing its mapped embeddings to the teacher's mapped embeddings.
    The loss is the same as used in Student.compute_loss.
    """
    student.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for student_batch, teacher_batch in zip(student_val_loader, teacher_val_loader):
            if use_precalculated_student_embeddings:
                assert student.freeze_backbone, (
                    "Cannot use pre-calculated embeddings when the student's backbone is being finetuned, "
                    "as the embeddings will change during training."
                )
                student_predictions = student.forward_precalculated_embeddings(
                    student_batch.to(device)
                )
            else:
                input_ids, attention_mask = student_batch
                student_predictions = student(
                    input_ids.to(device),
                    attention_mask.to(device)
                )

            teacher_targets = teacher.get_targets_from_precalculated_embeddings(
                teacher_batch.to(device)
            )

            loss = student.compute_loss(student_predictions, teacher_targets, alpha=alpha)

            batch_size = input_ids.size(0) if not use_precalculated_student_embeddings else student_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss
