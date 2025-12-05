"""
Main training script for distilled sentence transformers using the Sentence Transformers framework.

This script provides a simplified, high-level interface for training distilled models
that follows Sentence Transformers best practices and conventions.
"""

import os
import argparse
import logging
from typing import Any

import torch.nn as nn
import torch
from torch.utils.data import Dataset, ConcatDataset

from transformers import TrainingArguments, EarlyStoppingCallback

from utils.train import SimilarityTrainer, collate_embeddings
from utils.custom_datasets import get_precalculated_embeddings_dataset
from utils.config import TRAINED_MODELS_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_model(
    trainable_projection: nn.Module,
    backbone_model_path: str,
    target_dim: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_batch_size: int,
    val_batch_size: int,
    epochs: int = 10,
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
    optimizer_params: dict[str, Any] = {'lr': 1e-4},
    weight_decay: float = 0.00,
    output_path: str = None,
    positional_loss_factor: float = 1.0,
    lr_scheduler_type: str = "linear",
) -> None:

    # Create training arguments
    args = TrainingArguments(
        output_dir=output_path or "./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch" if val_dataset is not None else "no",
        logging_dir="./logs",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=False,
        disable_tqdm=False,
        warmup_ratio=0.0,
        lr_scheduler_type=lr_scheduler_type,
    )

    # Create optimizer
    optimizer = optimizer_class(trainable_projection.parameters(), **optimizer_params)

    # Initialize custom trainer
    trainer = SimilarityTrainer(
        model=trainable_projection,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        target_dim=target_dim,
        backbone_model_path=backbone_model_path,
        positional_loss_factor=positional_loss_factor,
        optimizers=(optimizer, None),
        data_collator=collate_embeddings,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1, 
                early_stopping_threshold=0.005
            )
        ],
    )
    trainer.train()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a distilled sentence transformer model")
    
    # Model configuration
    parser.add_argument("--backbone_model", type=str, default="jinaai/jina-embeddings-v2-small-en",
                       help="Backbone model name or path")
    parser.add_argument("--backbone_model_output_dim", type=int, default=512,)
    parser.add_argument("--target_dim", type=int, default=32,
                       help="Target dimension for distilled embeddings")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                       help="Learning rate")
    parser.add_argument("--positional_loss_factor", type=float, default=1,
                       help="Weight for positional vs similarity loss")
    parser.add_argument("--train_batch_size", type=int, default=8192 * 2,
                       help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=8192 * 2,
                       help="Batch size for validation")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                       help="Learning rate scheduler type")

    # Output configuration
    parser.add_argument(
        "--custom_model_save_dir", 
        type=str, 
        default=None,
        help=(
            "Custom directory to save the model at. "
            "Will be appended to the models path."
        )
    )
                    

    args = parser.parse_args()
        
    model_name = (
        f"{args.backbone_model.replace("/", "__")}"
        f"_distilled_{args.target_dim}"
        f"_batch_{args.train_batch_size}"
        f"_poslossfactor_{float(args.positional_loss_factor)}"
    )
    
    custom_save_dir = args.custom_model_save_dir
    if custom_save_dir:
        output_path = os.path.join(TRAINED_MODELS_PATH, args.backbone_model.replace("/", "__"), custom_save_dir, model_name)
    else:
        output_path = os.path.join(TRAINED_MODELS_PATH, args.backbone_model.replace("/", "__"), model_name)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Backbone model: {args.backbone_model}")
    logger.info(f"Target dimension: {args.target_dim}")
    logger.info(f"Output path: {output_path}")    
    
    logger.info("Creating trainable projection")
    # trainable_projection = nn.Sequential(
    #     nn.Linear(args.backbone_model_output_dim, args.backbone_model_output_dim * 4),
    #     nn.ReLU(),
    #     nn.Linear(args.backbone_model_output_dim * 4, args.target_dim)
    # )
    trainable_projection = nn.Sequential(
        nn.Linear(args.backbone_model_output_dim, args.target_dim),
        nn.ReLU(),
    )
    trainable_projection.to(torch.device("cuda"))
    
    logger.info("Preparing datasets")
    train_datasets = []
    val_datasets = []
    for dataset_name in [
        "allenai/c4", 
    ]:
        train_datasets.append(get_precalculated_embeddings_dataset(
            dataset_path=dataset_name,
            model_name=args.backbone_model.replace("/", "__"),
            split="train",
        ))
        val_datasets.append(get_precalculated_embeddings_dataset(
            dataset_path=dataset_name,
            model_name=args.backbone_model.replace("/", "__"),
            split="validation",
        ))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    # Train the model
    logger.info("Starting training")
    train_model(
        trainable_projection=trainable_projection,
        backbone_model_path=args.backbone_model,
        target_dim=args.target_dim,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': args.learning_rate},
        output_path=output_path,
        positional_loss_factor=args.positional_loss_factor,
        lr_scheduler_type=args.lr_scheduler_type
    )


if __name__ == "__main__":
    main()
