"""
Training utilities for distilled sentence transformers using the Sentence Transformers framework.
"""

import torch
import logging

from transformers import Trainer
import torch.nn.functional as F

from utils.distilled_sentence_transformer import DistilledSentenceTransformer
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
        backbone_model_path: str,
        weight_exponent: int,
        positional_loss_factor: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_exponent = weight_exponent
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
        high_dim_embeddings = inputs["input"]
        low_dim_embeddings = model(high_dim_embeddings)

        # Compute losses
        angular_loss = 0.0
        positional_loss = 0.0
        
        if self.positional_loss_factor > 0:
            positional_loss = self._compute_positional_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings
            )

            positional_loss.requires_grad_(True)

        if self.positional_loss_factor < 1:
            angular_loss = self._compute_angular_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings
            )
            angular_loss.requires_grad_(True)
        
        return (
            self.positional_loss_factor * positional_loss + 
            (1 - self.positional_loss_factor) * angular_loss
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

                batch_size = high_dim_embeddings.shape[0]
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        
        return metrics
