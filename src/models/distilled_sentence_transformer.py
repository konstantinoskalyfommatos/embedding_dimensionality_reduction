"""
Distilled Sentence Transformer model that integrates with the Sentence Transformers framework.
This wrapper allows our custom distillation approach to be used as a standard SentenceTransformer model.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional
import numpy as np
import json
import os


class ProjectionHead(nn.Module):
    """Custom projection head that preserves geometric relationships."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
            hidden_dim = hidden_dim if hidden_dim % 2 == 0 else hidden_dim - 1
            
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.input_dim = input_dim
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
            'sentence_embedding_with_zero': projected  # Keep for loss computation
        })
        return features
    
    def get_sentence_embedding_dimension(self) -> int:
        return self.output_dim
    
    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        with open(os.path.join(output_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, input_path: str):
        config_path = os.path.join(input_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        model = cls(config["input_dim"], config["output_dim"])
        model.load_state_dict(torch.load(os.path.join(input_path, "pytorch_model.bin")))
        return model


class DistilledSentenceTransformer(SentenceTransformer):
    """
    A SentenceTransformer wrapper for our distilled models.
    
    This class extends SentenceTransformer to integrate our custom distillation approach
    with the standard Sentence Transformers framework, providing a familiar API.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        target_dim: int = 256,
        hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        device: Optional[str] = "cuda",
        **kwargs
    ):
        """
        Initialize the distilled sentence transformer.
        
        Args:
            model_name_or_path: Path to the base model or HuggingFace model name
            target_dim: Target dimensionality for the distilled embeddings
            hidden_dim: Hidden dimension size for the projection network
            freeze_backbone: Whether to freeze the backbone during training
            device: Device to use ('cuda', 'cpu', etc.)
        """
        super().__init__(model_name_or_path, device=device, **kwargs)
        
        self.target_dim = target_dim
        self.freeze_backbone = freeze_backbone
        
        original_dim = self.get_sentence_embedding_dimension()
        
        # Add our custom projection head
        projection_head = ProjectionHead(
            input_dim=original_dim,
            output_dim=target_dim,
            hidden_dim=hidden_dim
        )
        
        self.add_module('projection_head', projection_head)
        
        if freeze_backbone:
            self._freeze_backbone()
    
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
    
    def save(self, path: str, **kwargs):
        """Save the complete model including our custom components."""
        super().save(path, **kwargs)
        
        # Save additional configuration
        config = {
            "target_dim": self.target_dim,
            "freeze_backbone": self.freeze_backbone,
        }
        
        config_path = os.path.join(path, "distillation_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'DistilledSentenceTransformer':
        """Load a saved distilled model."""
        config_path = os.path.join(path, "distillation_config.json")
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            
            # Load with original configuration
            model = cls(
                model_name_or_path=path,
                target_dim=config.get("target_dim", 256),
                freeze_backbone=config.get("freeze_backbone", True),
                **kwargs
            )
        else:
            # Fallback to default loading
            model = super()(path, **kwargs)
        
        return model
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimension of the final embeddings."""
        if hasattr(self, 'projection_head'):
            return self.projection_head.output_dim
        return super().get_sentence_embedding_dimension()
