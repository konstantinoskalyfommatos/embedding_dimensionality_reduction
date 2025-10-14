"""
Main training script for distilled sentence transformers using the Sentence Transformers framework.

This script provides a simplified, high-level interface for training distilled models
that follows Sentence Transformers best practices and conventions.
"""

import os
import argparse
import logging

import torch.nn as nn
import torch

from torch.utils.data import ConcatDataset

from core.train import train_model
from utils.custom_datasets import get_precalculated_embeddings_dataset
from core.config import PROJECT_ROOT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a distilled sentence transformer model")
    
    # Model configuration
    parser.add_argument("--backbone_model", type=str, default="jinaai/jina-embeddings-v2-small-en",
                       help="Backbone model name or path")
    parser.add_argument("--backbone_model_output_dim", type=int, default=512,)
    parser.add_argument("--target_dim", type=int, default=32,
                       help="Target dimension for distilled embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension for projection network")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--evaluation_ratio", type=float, default=0.250,
                       help="Ratio of training steps to perform evaluation")
    parser.add_argument("--positional_loss_factor", type=float, default=1,
                       help="Weight for positional vs similarity loss")
    parser.add_argument("--train_batch_size", type=int, default=8192,
                       help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=8192,
                       help="Batch size for validation")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                       help="Learning rate scheduler type")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for saving the model")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name for the saved model")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Configure output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "storage", "models")
    
    if args.model_name is None:
        teacher_name = args.backbone_model.split('/')[-1]
        args.model_name = f"{teacher_name}_distilled_{args.target_dim}"
    
    output_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Starting distillation training:")
    logger.info(f"Teacher model: {args.backbone_model}")
    logger.info(f"Target dimension: {args.target_dim}")
    logger.info(f"Output path: {output_path}")    
    
    # Create student model
    logger.info("Creating student model")
    # Our student model is simply a projection layer
    match args.target_dim:
        case 32:
            trainable_projection = nn.Sequential(
                nn.Linear(args.backbone_model_output_dim, 128),
                nn.GELU(),
                nn.Linear(128, 32),
            )
        case 16:
            trainable_projection = nn.Sequential(
                nn.Linear(args.backbone_model_output_dim, 64),
                nn.GELU(),
                nn.Linear(64, 16),
            )
        case 3:
            trainable_projection = nn.Sequential(
                nn.Linear(args.backbone_model_output_dim, 128),
                nn.GELU(),
                nn.Linear(128, 3),
            )
    trainable_projection.to(torch.device("cuda"))
    
    logger.info("Preparing datasets")
    train_datasets = []
    val_datasets = []
    for dataset_name in [
        "allenai/c4", 
        "cl-nagoya/wikisplit-pp"
    ]:
        train_datasets.append(get_precalculated_embeddings_dataset(
            dataset_path=dataset_name,
            model_name=args.backbone_model,
            split="train",
        ))
        val_datasets.append(get_precalculated_embeddings_dataset(
            dataset_path=dataset_name,
            model_name=args.backbone_model,
            split="validation",
        ))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    # Training parameters
    optimizer_params = {'lr': args.learning_rate}

    # Train the model
    logger.info("Starting training")
    trained_model = train_model(
        trainable_projection=trainable_projection,
        backbone_model_path=args.backbone_model,
        target_dim=args.target_dim,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params=optimizer_params,
        evaluation_ratio=args.evaluation_ratio,
        output_path=output_path,
        positional_loss_factor=args.positional_loss_factor,
        lr_scheduler_type=args.lr_scheduler_type
    )


if __name__ == "__main__":
    main()
