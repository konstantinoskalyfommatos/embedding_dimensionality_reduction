import torch
import torch.nn as nn
from mteb.models.model_meta import ModelMeta

from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file
import os


class ProjectionHead(nn.Module):
    """Custom projection head that preserves geometric relationships."""
    
    def __init__(
        self, 
        projection: nn.Module,
        output_dim: int,
    ):
        super().__init__()
        self.projection = projection
        self.output_dim = output_dim
    
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:        
        """Projects into lower dimensionality space"""
        features.update({
            'sentence_embedding': features['sentence_embedding']
        })
        return features

class DistilledSentenceTransformer(SentenceTransformer):
    """A SentenceTransformer wrapper for our distilled models.
    
    This class extends SentenceTransformer to integrate our 
    custom distillation approach with the standard Sentence 
    Transformers framework, providing a familiar API.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        projection: nn.Module,
        output_dim: int,
        device: str = "cuda",
        custom_model_name: str | None = None,
        max_seq_length: int = 512,
        **kwargs
    ):
        base_model = SentenceTransformer(
            model_name_or_path, 
            device=device, 
            trust_remote_code=True,
            **kwargs
        )
        base_model.max_sequence_length = max_seq_length
        
        projection_head = ProjectionHead(
            projection, 
            output_dim=output_dim, 
        )  
        modules = list(base_model._modules.values()) + [projection_head]
        super().__init__(modules=modules, device=device)
        
        self.output_dim = output_dim

        if custom_model_name:
            self._model_name = custom_model_name

            # MTEB combatibility
            self.model_card_data.name = self._model_name
            self.model_card_data.model_id = self._model_name
            self.model_card_data.model_name = self._model_name

    @property
    def projection_head(self) -> ProjectionHead:
        """A property to dynamically find and return the ProjectionHead module.

        This avoids storing a separate attribute that could conflict with the
        nn.Module's submodule registration.
        """
        for module in self._modules.values():
            if isinstance(module, ProjectionHead):
                return module
        raise AttributeError("ProjectionHead not found in model modules.")

    def load_checkpoint(self, path: str, **kwargs) -> 'DistilledSentenceTransformer':
        """Loads a saved distilled model checkpoint."""
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")
        
        checkpoint = load_file(path)
        
        # Fix key mismatch by adding 'projection.' prefix to keys
        fixed_checkpoint = {}
        for key, value in checkpoint.items():
            # Add 'projection.' prefix if not already present
            if not key.startswith('projection.'):
                new_key = f'projection.{key}'
            else:
                new_key = key
            
            # Move tensor to correct device
            if isinstance(value, torch.Tensor):
                fixed_checkpoint[new_key] = value.to(self.device)
            else:
                fixed_checkpoint[new_key] = value
        
        # Load the fixed checkpoint
        missing_keys, unexpected_keys = self.projection_head.load_state_dict(fixed_checkpoint, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        self.to(self.device)
        return self

    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimension of the final embeddings."""
        return self.projection_head.output_dim
    