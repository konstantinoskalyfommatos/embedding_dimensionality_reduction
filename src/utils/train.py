import torch
import logging
from transformers import Trainer

from utils.eval import compute_positional_loss, compute_angular_loss


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
        weight_exponent: int,
        positional_loss_factor: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_exponent = weight_exponent
        self.positional_loss_factor = positional_loss_factor
        self.target_dim = target_dim

    def compute_loss(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """Compute the combined loss for distillation."""
        high_dim_embeddings = inputs["input"]
        low_dim_embeddings = model(high_dim_embeddings)

        # Compute losses
        angular_loss = 0.0
        positional_loss = 0.0
        
        if self.positional_loss_factor > 0:
            positional_loss = compute_positional_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings,
                weight_exponent=self.weight_exponent
            )
            positional_loss.requires_grad_(True)

        if self.positional_loss_factor < 1:
            angular_loss = compute_angular_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings,
                weight_exponent=self.weight_exponent
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

    def _evaluate_intrinsic(self, eval_dataset, metric_key_prefix="eval"):
        """Returns the intrinsic evaluation loss on the validation dataset."""            
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for features in eval_dataloader:
                high_dim_embeddings = features["input"] if isinstance(features, dict) else features
                low_dim_embeddings = model(high_dim_embeddings)

                angular_loss = 0.0
                positional_loss = 0.0

                if self.positional_loss_factor > 0:
                    positional_loss = compute_positional_loss(
                        low_dim_embeddings=low_dim_embeddings,
                        high_dim_embeddings=high_dim_embeddings,
                        weight_exponent=self.weight_exponent
                    )
                if self.positional_loss_factor < 1:
                    angular_loss = compute_angular_loss(
                        low_dim_embeddings=low_dim_embeddings,
                        high_dim_embeddings=high_dim_embeddings,
                        weight_exponent=self.weight_exponent
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
