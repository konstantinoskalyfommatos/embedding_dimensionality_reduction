import torch
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


def train(
    student,
    teacher,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    validation_fn,
    model_path: str,
    epochs: int = 10,
    early_stopping_patience: int = 3,
    print_every: int = 1
):
    print(f"Training student model with {len(train_loader)} batches per epoch.")
    student.train()

    best_model_path = model_path.replace('.pth', '_best.pth')
    best_metric = None
    epochs_no_improve = 0

    for i, epoch in enumerate(range(epochs)):
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids, attention_mask = batch
            input_ids = input_ids.to(student.device)
            attention_mask = attention_mask.to(student.device)
            loss = student.compute_loss(input_ids, attention_mask, teacher)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (i + 1) % print_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        scheduler.step()

        current_metric = validation_fn(
            student=student, 
            teacher=teacher,
            val_loader=val_loader
        )

        # Save best model
        if best_metric is None or current_metric > best_metric:
            best_metric = current_metric
            epochs_no_improve = 0
            torch.save(student.state_dict(), best_model_path)
            print(f"New best model (Metric: {best_metric:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}!")
                break

    # Load best model before returning
    if os.path.exists(best_model_path):
        student.load_state_dict(torch.load(best_model_path))
    return student
