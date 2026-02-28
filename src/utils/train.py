import torch
import logging
from transformers import Trainer
from torch.utils.data import DataLoader

from utils.eval import compute_positional_loss, compute_angular_loss, compute_spearman_loss


torch.manual_seed(42)

logger = logging.getLogger(__name__)


def collate_embeddings(features):
    batched_embeddings = torch.stack(features, dim=0)
    return {"input": batched_embeddings}


class SimilarityTrainer(Trainer):
    def __init__(
        self,
        *args,
        target_dim: int,
        spearman: bool,
        positional_loss_factor: float = 1.0,
        weighted_loss: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.positional_loss_factor = positional_loss_factor
        self.target_dim = target_dim
        self.spearman = spearman
        self.weighted_loss = weighted_loss

    def compute_loss(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """Compute the combined loss for distillation."""
        high_dim_embeddings = inputs["input"]
        low_dim_embeddings = model(high_dim_embeddings)

        if self.spearman:
            return compute_spearman_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings,
                training=True,
                weighted=self.weighted_spearman,
                local=self.local
            )
        
        angular_loss = 0.0
        positional_loss = 0.0

        if self.positional_loss_factor > 0:
            positional_loss = compute_positional_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings,
                weighted=self.weighted_loss
            )
            positional_loss.requires_grad_(True)

        if self.positional_loss_factor < 1:
            angular_loss = compute_angular_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings,
                weighted=self.weighted_loss
            )
            angular_loss.requires_grad_(True)

        return (
            self.positional_loss_factor * positional_loss + 
            (1 - self.positional_loss_factor) * angular_loss
        )
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        metrics = self._evaluate_intrinsic(eval_dataset, metric_key_prefix)
        self.log(metrics)
        return metrics

    @torch.no_grad()
    def _evaluate_intrinsic(self, eval_dataset, metric_key_prefix="eval"):
        """Returns the intrinsic evaluation loss on the validation dataset."""            
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        for features in eval_dataloader:
            high_dim_embeddings = features["input"] if isinstance(features, dict) else features
            high_dim_embeddings = high_dim_embeddings.to(self.args.device)
            low_dim_embeddings = model(high_dim_embeddings)

            if self.spearman:
                loss = compute_spearman_loss(
                    low_dim_embeddings=low_dim_embeddings,
                    high_dim_embeddings=high_dim_embeddings,
                    training=False,
                    weighted=self.weighted_loss,
                )
            else:
                angular_loss = 0.0
                positional_loss = 0.0

                if self.positional_loss_factor > 0:
                    positional_loss = compute_positional_loss(
                        low_dim_embeddings=low_dim_embeddings,
                        high_dim_embeddings=high_dim_embeddings,
                        weighted=self.weighted_loss
                    )
                if self.positional_loss_factor < 1:
                    angular_loss = compute_angular_loss(
                        low_dim_embeddings=low_dim_embeddings,
                        high_dim_embeddings=high_dim_embeddings,
                        weighted=self.weighted_loss
                    )
                
                loss = (
                    self.positional_loss_factor * positional_loss + 
                    (1 - self.positional_loss_factor) * angular_loss
                )

            total_loss += loss.item()
            num_samples += 1
            
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        return metrics

    def get_eval_dataloader(self, eval_dataset):
        """Override to set drop_last=True for eval dataloader.
        
        Since we care about preserving relationships between a batch of embeddings, \
        we do not want to evaluate on a small number of embeddings.
        """
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    