"""
Training utilities for distilled sentence transformers using the Sentence Transformers framework.
"""

import torch
from torch.utils.data import Dataset
from typing import Any
import logging

from transformers import TrainingArguments, EarlyStoppingCallback
from transformers import Trainer
import torch.nn as nn

from utils.embed_functions import embed_points_isometric
from core.distilled_sentence_transformer import DistilledSentenceTransformer
from core.eval import evaluate_sts

torch.manual_seed(42)

logger = logging.getLogger(__name__)
    

def collate_embeddings(features):
    """
    Collate a list of tensors or dicts into a batch dict with key 'input'.
    """
    if isinstance(features[0], dict):
        return {"input": torch.stack([f["input"] for f in features], dim=0)}
    return {"input": torch.stack(features, dim=0)}


class SimilarityTrainer(Trainer):
    def __init__(
        self,
        *args,
        target_dim: int,
        backbone_model_path: str,
        positional_loss_factor: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.positional_loss_factor = positional_loss_factor
        self.target_dim = target_dim

        self.distilled_sentece_transformer = DistilledSentenceTransformer(
            model_name_or_path=backbone_model_path,
            projection=self.model,
            output_dim=target_dim,
            device=self.args.device
        )
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """Compute the combined loss for distillation."""
        if isinstance(inputs, dict):
            high_dim_embeddings = inputs["input"]
            if high_dim_embeddings is None:
                raise ValueError("No input tensor found in batch. Expected key 'input'.")
        else:
            high_dim_embeddings = inputs

        low_dim_embeddings = model(high_dim_embeddings)

        # Compute losses
        similarity_loss = 0.0
        positional_loss = 0.0
        
        if self.positional_loss_factor > 0:
            positional_loss = self._compute_positional_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings
            )
            positional_loss.requires_grad_(True)

        if self.positional_loss_factor < 1:
            similarity_loss = self._compute_similarity_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings
            )
            similarity_loss.requires_grad_(True)
        
        return (
            self.positional_loss_factor * positional_loss + 
            (1 - self.positional_loss_factor) * similarity_loss
        )
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = self._evaluate_intrinsic(eval_dataset, metric_key_prefix)
        self.log(metrics)
        return metrics

    def _evaluate_intrinsic(self, eval_dataset=None, metric_key_prefix="eval"):
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
                high_dim_embeddings = features["input"] if isinstance(features, dict) else features

                low_dim_embeddings = model(high_dim_embeddings)

                similarity_loss = 0.0
                positional_loss = 0.0

                if self.positional_loss_factor > 0:
                    positional_loss = self._compute_positional_loss(
                        low_dim_embeddings=low_dim_embeddings,
                        high_dim_embeddings=high_dim_embeddings
                    )
                if self.positional_loss_factor < 1:
                    similarity_loss = self._compute_similarity_loss(
                        low_dim_embeddings=low_dim_embeddings,
                        high_dim_embeddings=high_dim_embeddings
                    )
                
                loss = (
                    self.positional_loss_factor * positional_loss + 
                    (1 - self.positional_loss_factor) * similarity_loss
                )

                batch_size = high_dim_embeddings.shape[0]
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        
        return metrics

    def _compute_positional_loss(
        self, 
        low_dim_embeddings: torch.Tensor, 
        high_dim_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distance preservation loss."""
        low_dim_dist = torch.cdist(low_dim_embeddings, low_dim_embeddings, p=2)
        high_dim_dist = torch.cdist(high_dim_embeddings, high_dim_embeddings, p=2)
        
        # Use triu_indices for better memory efficiency
        n = low_dim_dist.size(0)
        triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)
        
        low_dim_dist_upper = low_dim_dist[triu_indices[0], triu_indices[1]]
        high_dim_dist_upper = high_dim_dist[triu_indices[0], triu_indices[1]]
        
        return torch.nn.functional.mse_loss(low_dim_dist_upper, high_dim_dist_upper)

    def _compute_similarity_loss(
        self, 
        low_dim_embeddings: torch.Tensor, 
        high_dim_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity preservation loss."""
        low_dim_norm = torch.nn.functional.normalize(low_dim_embeddings, dim=1)
        high_dim_norm = torch.nn.functional.normalize(high_dim_embeddings, dim=1)
        
        # Compute similarity matrices
        low_dim_sim = torch.mm(low_dim_norm, low_dim_norm.t())
        high_dim_sim = torch.mm(high_dim_norm, high_dim_norm.t())
        
        # Use triu_indices for better memory efficiency
        n = low_dim_sim.size(0)
        triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)
        
        low_dim_sim_upper = low_dim_sim[triu_indices[0], triu_indices[1]]
        high_dim_sim_upper = high_dim_sim[triu_indices[0], triu_indices[1]]
        
        return torch.nn.functional.mse_loss(low_dim_sim_upper, high_dim_sim_upper) * 100


def train_model(
    trainable_projection: nn.Module,
    backbone_model_path: str,
    target_dim: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_batch_size: int,
    val_batch_size: int,
    epochs: int = 10,
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
    optimizer_params: dict[str, Any] = {'lr': 1e-4},
    weight_decay: float = 0.01,
    evaluation_ratio: float = 0.5,
    output_path: str = None,
    positional_loss_factor: float = 1.0,
    lr_scheduler_type: str = "linear",
) -> nn.Module:
    
    steps_per_epoch = len(train_dataset) // train_batch_size
    evaluation_steps = max(1, int(steps_per_epoch * evaluation_ratio))

    # Create training arguments
    args = TrainingArguments(
        output_dir=output_path or "./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        weight_decay=weight_decay,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=evaluation_steps if val_dataset is not None else None,
        logging_dir="./logs",
        logging_strategy="epoch",
        # Model saving configurations - match eval strategy
        save_strategy="steps" if val_dataset is not None else "epoch",
        save_steps=evaluation_steps if val_dataset is not None else None,
        save_total_limit=5,
        load_best_model_at_end=True if val_dataset is not None else False,
        # metric_for_best_model="eval_spearmanr",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=False,
        disable_tqdm=False,
        warmup_ratio=0.0,
        lr_scheduler_type=lr_scheduler_type,
    )

    # Create optimizer
    optimizer = optimizer_class(trainable_projection.parameters(), **optimizer_params)

    # Initialize custom trainer
    trainer = SimilarityTrainer(
        model=trainable_projection,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        target_dim=target_dim,
        backbone_model_path=backbone_model_path,
        positional_loss_factor=positional_loss_factor,
        optimizers=(optimizer, None),
        data_collator=collate_embeddings,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005)],
    )

    trainer.train()

    return trainable_projection
