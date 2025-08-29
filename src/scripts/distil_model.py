"""
Main training script for distilled sentence transformers using the Sentence Transformers framework.

This script provides a simplified, high-level interface for training distilled models
that follows Sentence Transformers best practices and conventions.
"""

import os
import argparse
import logging
from dotenv import load_dotenv

import torch.nn as nn
import torch

from torch.utils.data import ConcatDataset

from core.train import train_distilled_model
from utils.custom_datasets import get_precalculated_embeddings_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a distilled sentence transformer model")
    
    # Model configuration
    parser.add_argument("--teacher_model", type=str, default="jinaai/jina-embeddings-v2-small-en",
                       help="Teacher model name or path")
    parser.add_argument("--teacher_model_output_dim", type=int, default=512,)
    parser.add_argument("--target_dim", type=int, default=32,
                       help="Target dimension for distilled embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension for projection network")
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default="cl-nagoya/wikisplit-pp",
                       help="HuggingFace dataset path")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Dataset configuration name")
    parser.add_argument("--text_column", type=str, default="simple_original",
                       help="Text column name in the dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    parser.add_argument("--evaluation_steps", type=int, default=20000,
                       help="Steps between evaluations")
    parser.add_argument("--positional_loss_factor", type=float, default=0.5,
                       help="Weight for positional vs similarity loss")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for saving the model")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name for the saved model")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set up project root
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    if not PROJECT_ROOT:
        raise ValueError("PROJECT_ROOT environment variable not set")
    
    # Configure output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "storage", "models")
    
    if args.model_name is None:
        teacher_name = args.teacher_model.split('/')[-1]
        args.model_name = f"{teacher_name}_distilled_{args.target_dim}"
    
    output_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_path, exist_ok=True)

    args.dataset_path = "allenai/c4"
    
    logger.info(f"Starting distillation training:")
    logger.info(f"  Teacher model: {args.teacher_model}")
    logger.info(f"  Target dimension: {args.target_dim}")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Output path: {output_path}")    
    
    # Create student model
    logger.info("Creating student model")
    # Our student model is simply a projection layer
    match args.target_dim:
        case 32:
            student_model = nn.Sequential(
                nn.Linear(args.teacher_model_output_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
        case 3:
            student_model = nn.Sequential(
                nn.Linear(args.teacher_model_output_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
            )
    student_model.to(torch.device("cuda"))
    
    logger.info("Preparing datasets")
    train_datasets = []
    val_datasets = []
    for dataset_name in ["allenai/c4", "cl-nagoya/wikisplit-pp"]:
        train_datasets.append(get_precalculated_embeddings_dataset(
            dataset_path=dataset_name,
            model_name=args.teacher_model,
            split="train",
        ))
        val_datasets.append(get_precalculated_embeddings_dataset(
            dataset_path=dataset_name,
            model_name=args.teacher_model,
            split="validation",
        ))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    # Training parameters
    optimizer_params = {'lr': args.learning_rate}

    # Train the model
    logger.info("Starting training")
    trained_model = train_distilled_model(
        student_model=student_model,
        backbone_model_path=args.teacher_model,
        target_dim=args.target_dim,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=args.target_dim,  # Batch size depends on target_dim
        val_batch_size=8192,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_class=torch.optim.AdamW,
        optimizer_params=optimizer_params,
        scheduler='WarmupLinear',
        weight_decay=0.01,
        evaluation_steps=args.evaluation_steps,
        output_path=output_path,
        positional_loss_factor=args.positional_loss_factor,
    )


if __name__ == "__main__":
    main()
