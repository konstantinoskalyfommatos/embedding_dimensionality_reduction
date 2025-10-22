"""
Training utilities for distilled sentence transformers using the Sentence Transformers framework.
"""

import torch
import logging

from transformers import Trainer

from utils.distilled_sentence_transformer import DistilledSentenceTransformer

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
        
        return torch.nn.functional.mse_loss(
            low_dim_dist_upper, 
            high_dim_dist_upper, 
            reduction="mean"
        )
    # def _compute_positional_loss(
    #     self,
    #     low_dim_embeddings: torch.Tensor,
    #     high_dim_embeddings: torch.Tensor,
    #     eps: float = 1e-8
    # ) -> torch.Tensor:
    #     """Compute pairwise distance preservation loss with inverse-distance weighting.
        
    #     Pairs that are close in the high-dimensional space are given more weight.
    #     """
    #     # Compute pairwise Euclidean distances
    #     low_dim_dist = torch.cdist(low_dim_embeddings, low_dim_embeddings, p=2)
    #     high_dim_dist = torch.cdist(high_dim_embeddings, high_dim_embeddings, p=2)

    #     # Use only the upper triangle (excluding diagonal) for efficiency
    #     n = low_dim_dist.size(0)
    #     triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)

    #     low_dim_dist_upper = low_dim_dist[triu_indices[0], triu_indices[1]]
    #     high_dim_dist_upper = high_dim_dist[triu_indices[0], triu_indices[1]]

    #     # Inverse-distance weights: small high-dim distances → larger weight
    #     weights = 1.0 / (high_dim_dist_upper + eps)

    #     # Normalize weights so loss scale stays consistent across batches
    #     weights = weights / (weights.sum() + eps)

    #     # Weighted mean squared error between pairwise distances
    #     loss = torch.sum(weights * (low_dim_dist_upper - high_dim_dist_upper) ** 2)

    #     return loss


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

        return torch.nn.functional.mse_loss(
            low_dim_sim_upper, 
            high_dim_sim_upper, 
            reduction="mean"
        ) * 100
