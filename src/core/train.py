"""
Training utilities for distilled sentence transformers using the Sentence Transformers framework.
"""

import torch
from torch.utils.data import Dataset
from typing import Any
import logging

from transformers import TrainingArguments
from transformers import Trainer
import torch.nn as nn

from src.utils.embed_functions import embed_points_isometric

logger = logging.getLogger(__name__)
    

def collate_embeddings(features):
    """
    Collate a list of tensors or dicts into a batch dict with key 'input'.
    """
    if isinstance(features[0], dict):
        return {"input": torch.stack([f["input"] for f in features], dim=0)}
    # features are tensors
    return {"input": torch.stack(features, dim=0)}


class SimilarityTrainer(Trainer):
    def __init__(
        self,
        *args,
        positional_loss_factor: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.positional_loss_factor = positional_loss_factor
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """
        Compute the combined loss for distillation.
        
        Teacher embeddings lie in the low-dimensional space.
        """
        # Accept both dict and tensor batches
        if isinstance(inputs, dict):
            batch = inputs["input"]
            if batch is None:
                raise ValueError("No input tensor found in batch. Expected key 'input'.")
        else:
            batch = inputs

        # inputs are the high dimensional vectors
        high_dim_zero_vector = torch.zeros((1, batch.shape[1]), device=batch.device)
        high_dim_vectors_with_zero = torch.cat([high_dim_zero_vector, batch], dim=0)
        teacher_vectors = embed_points_isometric(high_dim_vectors_with_zero)

        # Get student embeddings
        student_vectors = model(batch)
        low_dim_zero = torch.zeros((1, student_vectors.shape[1]), device=student_vectors.device)
        student_embeddings_with_zero = torch.cat(
            [low_dim_zero, student_vectors], 
            dim=0
        )

        # Compute losses
        similarity_loss = 0.0
        positional_loss = 0.0
        
        if self.positional_loss_factor > 0:
            positional_loss = self._compute_positional_loss(
                student_embeddings_with_zero, 
                teacher_vectors
            )

        if self.positional_loss_factor < 1:
            # Remove zero vector for similarity loss
            student_no_zero = student_embeddings_with_zero[1:]
            teacher_no_zero = teacher_vectors[1:]

            similarity_loss = self._compute_similarity_loss(
                student_no_zero, 
                teacher_no_zero
            )
        
        return (
            self.positional_loss_factor * positional_loss + 
            (1 - self.positional_loss_factor) * similarity_loss
        )
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Returns the intrinsic evaluation loss on the evaluation dataset."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if eval_dataset is None:
            return {}
            
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for features in eval_dataloader:
                batch = features["input"] if isinstance(features, dict) else features

                teacher_vectors = batch
                high_dim_zero = torch.zeros(
                    (1, teacher_vectors.shape[1]), 
                    device=teacher_vectors.device
                )
                teacher_vectors_with_zero = torch.cat(
                    [high_dim_zero, teacher_vectors], 
                    dim=0
                )

                student_vectors = model(batch)
                low_dim_zero = torch.zeros(
                    (1, student_vectors.shape[1]), 
                    device=student_vectors.device
                )
                student_vectors_with_zero = torch.cat(
                    [low_dim_zero, student_vectors], 
                    dim=0
                )

                similarity_loss = 0.0
                positional_loss = 0.0
                if self.positional_loss_factor > 0:
                    positional_loss = self._compute_positional_loss(
                        student_vectors_with_zero, 
                        teacher_vectors_with_zero
                    )
                if self.positional_loss_factor < 1:
                    similarity_loss = self._compute_similarity_loss(
                        student_vectors_with_zero[1:], 
                        teacher_vectors_with_zero[1:]
                    )
                
                loss = (
                    self.positional_loss_factor * positional_loss + 
                    (1 - self.positional_loss_factor) * similarity_loss
                )

                batch_size = batch.shape[0]
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        
        # Log metrics
        self.log(metrics)
        
        return metrics
    
    def _compute_positional_loss(
        self, 
        student_vectors: torch.Tensor, 
        teacher_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distance preservation loss."""
        student_dist = torch.cdist(student_vectors, student_vectors, p=2)
        teacher_dist = torch.cdist(teacher_vectors, teacher_vectors, p=2)
        
        mask = torch.triu(torch.ones_like(student_dist), diagonal=1).bool()
        return nn.MSELoss()(student_dist[mask], teacher_dist[mask])
    
    def _compute_similarity_loss(
        self, 
        student_vectors: torch.Tensor, 
        teacher_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity preservation loss."""
        student_vectors = nn.functional.normalize(student_vectors, dim=1)
        teacher_vectors = nn.functional.normalize(teacher_vectors, dim=1)
        
        student_sim_matrix = torch.matmul(student_vectors, student_vectors.T)
        teacher_sim_matrix = torch.matmul(teacher_vectors, teacher_vectors.T)
        
        mask = torch.triu(torch.ones_like(student_sim_matrix), diagonal=1).bool()
        return nn.MSELoss()(student_sim_matrix[mask], teacher_sim_matrix[mask])


def train_distilled_model(
    student_model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_batch_size: int,
    val_batch_size: int,
    epochs: int = 10,
    warmup_steps: int = 1000,
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
    optimizer_params: dict[str, Any] = {'lr': 1e-4},
    scheduler: str = 'WarmupLinear',
    weight_decay: float = 0.01,
    evaluation_steps: int = 1000,
    output_path: str = None,
    positional_loss_factor: float = 1.0,
) -> nn.Module:
    # Create training arguments
    args = TrainingArguments(
        output_dir=output_path or "./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_strategy="steps" if evaluation_steps > 0 and val_dataset is not None else "no",
        eval_steps=evaluation_steps if val_dataset is not None else None,
        logging_dir="./logs",
        logging_strategy="epoch",
        # Model saving configurations - match eval strategy
        save_strategy="steps" if evaluation_steps > 0 and val_dataset is not None else "epoch",
        save_steps=evaluation_steps if val_dataset is not None else None,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset is not None else False,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,  # Lower loss is better
        dataloader_drop_last=True,
    )

    # Create optimizer
    optimizer = optimizer_class(student_model.parameters(), **optimizer_params)
    
    # Create scheduler based on string parameter
    total_steps = len(train_dataset) // train_batch_size * epochs
    if scheduler == 'WarmupLinear':
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        lr_scheduler = None

    # Initialize custom trainer
    trainer = SimilarityTrainer(
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        positional_loss_factor=positional_loss_factor,
        optimizers=(optimizer, lr_scheduler),
        data_collator=collate_embeddings,  # New
    )

    trainer.train()

    return student_model