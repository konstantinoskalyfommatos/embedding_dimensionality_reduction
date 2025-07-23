import torch
from torch.utils.data import DataLoader


def eval_intrinsic_original_vs_projected_space(
    student,
    student_val_loader: DataLoader,
    teacher_val_loader: DataLoader,
    device: str = "cuda",
    positional_loss_factor: float = 0.5, 
    use_precalculated_student_embeddings: bool = False,
) -> float:
    """Evaluation on high and low-dimensional spaces.
    
    Evaluates the student model by comparing its embeddings'
    cosine similarities and pairwise distances to the teacher's
    embeddings, where the latter are in the high-dimensional space.
    """
    student.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for student_batch, teacher_batch in zip(
            student_val_loader, 
            teacher_val_loader
        ):
            if use_precalculated_student_embeddings:
                assert student.freeze_backbone, (
                    "Cannot use pre-calculated embeddings when the student's "
                    "backbone is being finetuned, as the embeddings will "
                    "change during training."
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

            loss = student.compute_loss_fixed_weight(
                student_emb,
                teacher_emb,
                positional_loss_factor=positional_loss_factor
            )

            if not use_precalculated_student_embeddings:
                batch_size = input_ids.size(0)
            else:
                batch_size = student_batch.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def eval_intrinsic_projected_space(
    student,
    teacher,
    student_val_loader: DataLoader,
    teacher_val_loader: DataLoader,
    device: str = "cuda",
    positional_loss_factor: float = 0.1,
    use_precalculated_student_embeddings: bool = False,
) -> float:
    """Evaluation on the same low-dimensional space.
    
    Evaluates the student model by comparing its mapped embeddings
    to the teacher's mapped embeddings.
    """
    student.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for student_batch, teacher_batch in zip(
            student_val_loader, 
            teacher_val_loader
        ):
            if use_precalculated_student_embeddings:
                assert student.freeze_backbone, (
                    "Cannot use pre-calculated embeddings when the student's "
                    "backbone is being finetuned, as the embeddings will "
                    "change during training."
                )
                student_predictions = student.forward_precalculated_embeddings(
                    student_batch.to(device),
                    keep_zero_vector=True
                )
            else:
                input_ids, attention_mask = student_batch
                student_predictions = student(
                    input_ids.to(device),
                    attention_mask.to(device),
                    keep_zero_vector=True
                )

            teacher_targets = teacher.get_targets_from_precalculated_embeddings(
                teacher_batch.to(device),
                keep_zero_vector=True
            )

            loss = student.compute_loss_fixed_weight(
                student_predictions, 
                teacher_targets, 
                positional_loss_factor=positional_loss_factor
            )

            if not use_precalculated_student_embeddings:
                batch_size = input_ids.size(0)
            else:
                batch_size = student_batch.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss
