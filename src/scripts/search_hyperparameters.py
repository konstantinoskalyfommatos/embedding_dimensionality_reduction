from transformers import TrainingArguments
from argparse import ArgumentParser
import torch
import torch.nn as nn
import optuna

from utils.train import SimilarityTrainer, collate_embeddings
from utils.custom_datasets import get_precalculated_embeddings_dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone-model", type=str, default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--target-dim", type=int, default=32)
    parser.add_argument("--backbone-output-dim", type=int, default=512)
    args = parser.parse_args()

    # Prepare datasets
    train_dataset = get_precalculated_embeddings_dataset(
        dataset_path="allenai/c4",
        model_name=args.backbone_model.replace("/", "__"),
        split="train",
    )
    val_dataset = get_precalculated_embeddings_dataset(
        dataset_path="allenai/c4",
        model_name=args.backbone_model.replace("/", "__"),
        split="validation",
    )

    def model_init(trial: optuna.Trial = None):
        # Get dimensions from config
        backbone_output_dim = 512  # Set this from your config
        target_dim = 32  # Set this from your config

        if trial is None:
            # Default hyperparameters for initial model instantiation
            dropout_rate = 0.1
            projection_type = "no_hidden"
            activation_fn = nn.ReLU()
        else:
            # Hyperparameters to tune
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
            projection_type = trial.suggest_categorical(
                "projection_type", 
                ["no_hidden", "low_hidden", "high_hidden"]
            )
            activation_fn_str = trial.suggest_categorical(
                "activation_fn",
                ["ReLU", "SELU", "GELU"]
            )
            match activation_fn_str:
                case "ReLU":
                    activation_fn = nn.ReLU()
                case "SELU":
                    activation_fn = nn.SELU()
                case "GELU":
                    activation_fn = nn.GELU()

        backbone_output_dim = 512
        target_dim = 32

        match projection_type:
            case "high_hidden":
                trainable_projection = nn.Sequential(
                    nn.Linear(backbone_output_dim, backbone_output_dim * 4),
                    nn.Dropout(dropout_rate),
                    activation_fn,
                    nn.Linear(backbone_output_dim * 4, target_dim)
                )
            case "low_hidden":
                trainable_projection = nn.Sequential(
                    nn.Linear(backbone_output_dim, backbone_output_dim // 2),
                    nn.Dropout(dropout_rate),
                    activation_fn,
                    nn.Linear(backbone_output_dim // 2, target_dim)
                )

            case "no_hidden":
                trainable_projection = nn.Sequential(
                    nn.Linear(backbone_output_dim, target_dim),
                    nn.Dropout(dropout_rate),
                    activation_fn,
                )
        
        trainable_projection.to(torch.device("cuda"))
        return trainable_projection

    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        }

    # Create training arguments
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=20000,
        per_device_eval_batch_size=20000,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        disable_tqdm=True,
        dataloader_pin_memory=True
    )

    # Initialize custom trainer with model_init
    trainer = SimilarityTrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        target_dim=args.target_dim,
        backbone_model_path=args.backbone_model,
        positional_loss_factor=1.0,
        data_collator=collate_embeddings,
    )
    
    # Run hyperparameter search
    trials = trainer.hyperparameter_search(
        hp_space=hp_space,
        direction="minimize",
        backend="optuna",
        n_trials=50,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    print(trials)