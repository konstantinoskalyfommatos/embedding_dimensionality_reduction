"""
Distilled Sentence Transformer model that integrates with the Sentence Transformers framework.
This wrapper allows our custom distillation approach to be used as a standard SentenceTransformer model.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional
import numpy as np
from safetensors.torch import load_file
import os


class ProjectionHead(nn.Module):
    """Custom projection head that preserves geometric relationships."""
    
    def __init__(
        self, 
        projection: nn.Module,
        output_dim: int
    ):
        super().__init__()
        self.projection = projection
        self.output_dim = output_dim
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embeddings = features['sentence_embedding']
        
        # Add zero vector for isometric embedding
        zero_vector = torch.zeros((1, embeddings.shape[1]), device=embeddings.device)
        embeddings_with_zero = torch.cat([zero_vector, embeddings], dim=0)
        
        # Project to lower dimension
        projected = self.projection(embeddings_with_zero)
        
        # Remove zero vector for final output
        projected_without_zero = projected[1:]
        
        features.update({
            'sentence_embedding': projected_without_zero,
        })
        return features

class DistilledSentenceTransformer(SentenceTransformer):
    """
    A SentenceTransformer wrapper for our distilled models.
    This class extends SentenceTransformer to integrate our custom distillation approach
    with the standard Sentence Transformers framework, providing a familiar API.
    """
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        projection: nn.Module,
        output_dim: int,
        freeze_backbone: bool = True,
        device: str = "cuda",
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Create a DistilledSentenceTransformer from a pretrained model.
        
        Args:
            model_name_or_path: Path to the base model or HuggingFace model name
            projection: The projection network module
            output_dim: Target dimensionality for the distilled embeddings
            freeze_backbone: Whether to freeze the backbone during training
            device: Device to use ('cuda', 'cpu', etc.)
        """
        base_model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        # Create instance and copy over the modules
        instance = cls.__new__(cls)
        
        # Copy all attributes from the base model
        instance.__dict__.update(base_model.__dict__)
        
        # Add our custom attributes
        instance.output_dim = output_dim
        instance.freeze_backbone = freeze_backbone
        
        # Create and add the projection head
        projection_head = ProjectionHead(projection, output_dim=output_dim)
        instance._modules[str(len(instance._modules))] = projection_head
        instance.projection_head = projection_head
        
        # Freeze backbone if requested
        if freeze_backbone:
            instance._freeze_backbone()
        
        return instance
    
    def __init__(self, *args, **kwargs):
        # This should not be called directly - use from_pretrained instead
        raise RuntimeError(
            "Use DistilledSentenceTransformer.from_pretrained() instead of __init__()"
        )
    
    def _freeze_backbone(self):
        """Freeze all parameters except the projection head."""
        for name, param in self.named_parameters():
            if 'projection_head' not in name:
                param.requires_grad = False
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        **kwargs
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings using the distilled model.
        This method overrides the parent encode method to use our custom projection.
        """
        return super().encode(
            sentences=sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
    
    def load_checkpoint(self, path: str, **kwargs) -> 'DistilledSentenceTransformer':
        """
        Load a saved distilled model checkpoint.
        
        Example path: storage/models/jina-embeddings-v2-small-en_distilled_3/checkpoint-1680000/model.safetensors
        """
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")
        
        checkpoint = load_file(path)
        self.projection_head.load_state_dict(checkpoint, strict=False)
        self.to(self.device)
        return self
    
    def update_projection(self, projection: nn.Module):
        """Update the projection head with a new projection module."""
        self.projection_head.projection = projection
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimension of the final embeddings."""
        return self.projection_head.output_dim