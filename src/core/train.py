import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


def train(
    student,
    teacher,
    student_train_loader: DataLoader,
    student_val_loader: DataLoader,
    teacher_train_loader: DataLoader,
    teacher_val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    validation_fn,
    model_path: str,
    epochs: int = 10,
    early_stopping_patience: int = 3,
    print_every: int = 1,
    use_precalculated_student_embeddings: bool = False,
    warmup_validation_epochs: int = 3,
):
    print("Training student model")
    student.train()

    lowest_validation_loss = float('inf')
    epochs_no_improve = 0

    for i, epoch in enumerate(range(epochs)):
        total_loss = 0.0

        batch_bar = tqdm(
            total=min(len(student_train_loader), len(teacher_train_loader)),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        )

        for batch_idx, (student_batch, teacher_batch) in enumerate(
            zip(student_train_loader, teacher_train_loader)
        ):
            optimizer.zero_grad()

            if use_precalculated_student_embeddings:
                assert student.freeze_backbone, (
                    "Cannot use pre-calculated embeddings when the student's "
                    "backbone is being finetuned, as the embeddings will "
                    "change during training."
                )
                x = student_batch.to(student.device)
            else:
                input_ids, attention_mask = student_batch
                x = (input_ids.to(student.device), attention_mask.to(student.device))

            student_emb = student(
                x,
                keep_zero_vector=True,
                use_precalculated_embeddings=use_precalculated_student_embeddings
            )

            teacher_emb = teacher.get_targets_from_precalculated_embeddings(
                teacher_batch.to(student.device)
            )

            loss = student.compute_loss(
                student_emb,
                teacher_emb,
                positional_loss_factor=1.0
            )
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            update_every = max(1, len(student_train_loader) // 5)
            if (batch_idx + 1) % update_every == 0 or (batch_idx + 1) == batch_bar.total:
                batch_bar.update(update_every if (batch_idx + 1) % update_every == 0 else batch_bar.total % update_every)

        avg_train_loss = total_loss / len(student_train_loader)

        scheduler.step()

        # Only run validation after warmup_epochs
        if epoch >= warmup_validation_epochs:
            current_validation_loss = validation_fn(
                student=student, 
                student_val_loader=student_val_loader,
                teacher_val_loader=teacher_val_loader,
                device=student.device,
                use_precalculated_student_embeddings=use_precalculated_student_embeddings,
            )

            if (i + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {current_validation_loss:.4f}"
                )

            # Save best model
            if current_validation_loss < lowest_validation_loss:
                lowest_validation_loss = current_validation_loss
                epochs_no_improve = 0
                torch.save(student.state_dict(), model_path)
                print(f"New best model (Metric: {lowest_validation_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}!")
                    break
        else:
            if (i + 1) % print_every == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f} (warmup)")

    if os.path.exists(model_path):
        student.load_state_dict(torch.load(model_path))
    return student
