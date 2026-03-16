"""
Training script for autoencoder-based embedding compression.

The autoencoder learns to compress high-dimensional embeddings into a lower-dimensional
latent space and reconstruct them. The loss is: MSE between original and reconstructed embeddings 
(measures how well the decoder recovers the full-dimensional representation).

"""

import os
import argparse
import logging
from typing import Any
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from utils.custom_datasets import get_precalculated_embeddings_dataset
from utils.config import TRAINED_AUTOENCODERS_PATH, EVALUATION_RESULTS_PATH
from utils.distilled_sentence_transformer import DistilledSentenceTransformer
from utils.eval import evaluate_mteb, eval_intrinsic


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Bottleneck autoencoder for embedding compression.

    Encoder: Linear → ReLU
    Decoder: Linear  (no activation – output should be comparable to the
             original L2-normalised / unnormalised embeddings)
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, 
        x: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def collate_embeddings(features):
    return {"input": torch.stack(features, dim=0)}


class AutoencoderTrainer(Trainer):
    """Custom Trainer for the Autoencoder model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        x = inputs["input"]
        z, x_hat = model(x)
        return F.mse_loss(x_hat, x)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        metrics = self._evaluate_intrinsic(eval_dataset, metric_key_prefix)
        self.log(metrics)
        return metrics

    @torch.no_grad()
    def _evaluate_intrinsic(self, eval_dataset, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        total_loss = 0.0
        num_batches = 0

        for features in eval_dataloader:
            x = features["input"] if isinstance(features, dict) else features
            x = x.to(self.args.device)
            z, x_hat = model(x)
            loss = F.mse_loss(x_hat, x)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {f"{metric_key_prefix}_loss": avg_loss}

    def get_eval_dataloader(self, eval_dataset):
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    

def train_autoencoder(
    autoencoder: Autoencoder,
    custom_model_name: str,
    target_dim: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_batch_size: int,
    val_batch_size: int,
    backbone_model: str,
    epochs: int = 10,
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
    optimizer_params: dict[str, Any] = {"lr": 1e-2},
    weight_decay: float = 0.0,
    output_path: str = None,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    resume_from_checkpoint: str = None,
    eval_after_training: bool = False,
    sts_batch_size: int = 4096,
    retrieval_batch_size: int = 6,
    classification_batch_size: int = 20,
    clustering_batch_size: int = 16,
    overwrite_cache: bool = False,
) -> None:
    before = time.perf_counter()

    args = TrainingArguments(
        output_dir=output_path or "./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        weight_decay=weight_decay,
        eval_strategy="steps",
        eval_steps=100,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_drop_last=True,
        disable_tqdm=False,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        dataloader_pin_memory=True,
    )

    optimizer = optimizer_class(autoencoder.parameters(), **optimizer_params)

    trainer = AutoencoderTrainer(
        model=autoencoder,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        data_collator=collate_embeddings,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info(f"Training completed in {(time.perf_counter() - before) / 3600:.2f} hours")

    # Save best model weights
    torch.save(autoencoder.state_dict(), os.path.join(output_path, "best_model.pt"))

    if not eval_after_training:
        logger.info("Evaluation after training is disabled. Exiting.")
        return

    best_checkpoint_path = trainer.state.best_model_checkpoint
    best_checkpoint_num = int(best_checkpoint_path.split("-")[-1])
    logger.info(f"Best checkpoint: {best_checkpoint_num}")

    cache_path = os.path.join(
        EVALUATION_RESULTS_PATH,
        "autoencoders",
        backbone_model.replace("/", "__"),
    )

    # Intrinsic evaluation using only the encoder half
    logger.info("Evaluating intrinsic metrics on test set")
    eval_intrinsic(
        projection=autoencoder.encoder,
        backbone_model_path=backbone_model,
        checkpoint=best_checkpoint_num,
        cache_path=cache_path,
        model_name=custom_model_name,
        spearman_test_batch_size=5000,
    )

    # MTEB evaluation using only the encoder half
    logger.info("Evaluating on MTEB benchmark")
    sentence_transformer = DistilledSentenceTransformer(
        model_name_or_path=backbone_model,
        projection=autoencoder.encoder,
        output_dim=target_dim,
        custom_model_name=custom_model_name,
    )
    sentence_transformer.eval()

    evaluate_mteb(
        model=sentence_transformer,
        cache_path=cache_path,
        sts_batch_size=sts_batch_size,
        retrieval_batch_size=retrieval_batch_size,
        classification_batch_size=classification_batch_size,
        clustering_batch_size=clustering_batch_size,
        skip_sts=False,
        skip_retrieval=False,
        skip_classification=False,
        skip_clustering=False,
        overwrite_cache=overwrite_cache,
    )

    logger.info(
        f"Training + evaluation completed in "
        f"{(time.perf_counter() - before) / 3600:.2f} hours"
    )


# NOTE: Models:
# - Alibaba-NLP/gte-multilingual-base
# - jinaai/jina-embeddings-v2-small-en

def main():
    parser = argparse.ArgumentParser(description="Train an autoencoder for embedding compression")

    # Model configuration
    parser.add_argument("--backbone_model", type=str, default="jinaai/jina-embeddings-v2-small-en",
                        help="Backbone model name or path")
    parser.add_argument("--backbone_model_output_dim", type=int, default=512)
    parser.add_argument("--target_dim", type=int, default=32,
                        help="Latent (bottleneck) dimension")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--train_batch_size", type=int, default=20000)
    parser.add_argument("--val_batch_size", type=int, default=20000)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--skip_eval_after_training", action="store_true",
                        help="Skip MTEB + intrinsic evaluation after training")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Whether to overwrite cached evaluation results")

    parser.add_argument("--sts_batch_size", type=int, default=3000)
    parser.add_argument("--retrieval_batch_size", type=int, default=4)
    parser.add_argument("--classification_batch_size", type=int, default=20)
    parser.add_argument("--clustering_batch_size", type=int, default=16)

    # Output configuration
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Resume training from the last checkpoint in the output directory")

    args = parser.parse_args()

    model_name = (
        f"{args.backbone_model}"
        f"_distilled_{args.target_dim}"
        "_autoencoder"
    )

    output_path = os.path.join(
        TRAINED_AUTOENCODERS_PATH,
        args.backbone_model.replace("/", "__"),
        model_name.replace("/", "__"),
    )
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Backbone model: {args.backbone_model}")
    logger.info(f"Target (latent) dimension: {args.target_dim}")
    logger.info(f"Output path: {output_path}")

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        last_checkpoint = max(
            int(d.split("checkpoint-")[-1])
            for d in os.listdir(output_path)
            if d.startswith("checkpoint-")
        )
        resume_checkpoint = os.path.join(output_path, f"checkpoint-{last_checkpoint}")

    logger.info("Creating autoencoder")
    autoencoder = Autoencoder(
        input_dim=args.backbone_model_output_dim,
        latent_dim=args.target_dim,
    )
    autoencoder.to(torch.device("cuda"))

    logger.info("Preparing datasets")
    train_dataset = get_precalculated_embeddings_dataset(
        dataset_path="allenai/c4",
        model_name=args.backbone_model.replace("/", "__"),
        split="train",
    )
    val_dataset = get_precalculated_embeddings_dataset(
        dataset_path="sentence-paraphrases",
        model_name=args.backbone_model.replace("/", "__"),
        split="validation",
    )

    logger.info("Starting training")
    train_autoencoder(
        autoencoder=autoencoder,
        custom_model_name=model_name,
        target_dim=args.target_dim,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        backbone_model=args.backbone_model,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        weight_decay=args.weight_decay,
        optimizer_params={"lr": args.learning_rate},
        output_path=output_path,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        resume_from_checkpoint=resume_checkpoint,
        eval_after_training=not args.skip_eval_after_training,
        sts_batch_size=args.sts_batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        classification_batch_size=args.classification_batch_size,
        clustering_batch_size=args.clustering_batch_size,
        overwrite_cache=args.overwrite_cache,
    )


if __name__ == "__main__":
    main()
