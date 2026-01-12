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
        normalize_vector_before_projecting: bool = False,
    ):
        super().__init__()
        self.projection = projection
        self.output_dim = output_dim
        self.normalize_vector_before_projecting = normalize_vector_before_projecting
    
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:        
        """Projects into lower dimensionality space"""
        vector = features['sentence_embedding']
        if self.normalize_vector_before_projecting:
            vector = nn.functional.normalize(vector, dim=0)
        features.update({
            'sentence_embedding': self.projection(vector)
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
        normalize_vector_before_projecting: bool = False,
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
            normalize_vector_before_projecting=normalize_vector_before_projecting
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

        # Freeze only the backbone parameters
        for param in self.parameters():
            param.requires_grad = False
        for param in self.projection_head.parameters():
            param.requires_grad = True

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
    
    def change_projection_head(self, projection: nn.Module):
        """Change the projection head with a new projection module."""
        new_projection_head = ProjectionHead(projection, output_dim=self.output_dim)
        
        # Find and replace the existing projection head in the _modules dictionary
        for module_key in self._modules.keys():
            if isinstance(self._modules[module_key], ProjectionHead):
                self._modules[module_key] = new_projection_head
                self._modules[module_key].to(self.device)
                return

        raise RuntimeError("Could not find a ProjectionHead module to replace.")

    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimension of the final embeddings."""
        return self.projection_head.output_dim
    