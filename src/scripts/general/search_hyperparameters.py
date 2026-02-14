from transformers import TrainingArguments, EarlyStoppingCallback
from argparse import ArgumentParser
import torch
import torch.nn as nn
import optuna
import os

from utils.train import SimilarityTrainer, collate_embeddings
from utils.custom_datasets import get_precalculated_embeddings_dataset
from utils.config import STORAGE_PATH


def save_to_csv(
    output_filepath: str,
    study_name: str = "projection_search", 
    storage: str = "sqlite:///optuna_study.db",
):
    study = optuna.load_study(
        study_name=study_name,
        storage=storage
    )

    df = study.trials_dataframe()
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone_model", type=str, default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--target_dim", type=int, default=32)
    parser.add_argument("--backbone_output_dim", type=int, default=512)
    args = parser.parse_args()

    # Prepare datasets
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

    def model_init(trial: optuna.Trial = None):
        global args

        if trial is None:
            # Default hyperparameters for initial model instantiation
            projection_type = "no_hidden"
            activation_fn = nn.ReLU()
        else:
            projection_type = trial.suggest_categorical(
                "projection_type", 
                [
                    "no_hidden", 
                    # "low_hidden", 
                    # "high_hidden"
                ]
            )
            activation_fn_str = trial.suggest_categorical(
                "activation_fn",
                [
                    "ReLU", 
                    # "SELU", 
                    # "GELU"
                ]
            )
            match activation_fn_str:
                case "ReLU":
                    activation_fn = nn.ReLU()
                case "SELU":
                    activation_fn = nn.SELU()
                case "GELU":
                    activation_fn = nn.GELU()

        match projection_type:
            case "high_hidden":
                trainable_projection = nn.Sequential(
                    nn.Linear(args.backbone_output_dim, args.backbone_output_dim * 4),
                    activation_fn,
                    nn.Linear(args.backbone_output_dim * 4, args.target_dim)
                )
            case "low_hidden":
                trainable_projection = nn.Sequential(
                    nn.Linear(args.backbone_output_dim, args.backbone_output_dim // 2),
                    activation_fn,
                    nn.Linear(args.backbone_output_dim // 2, args.target_dim)
                )

            case "no_hidden":
                trainable_projection = nn.Sequential(
                    nn.Linear(args.backbone_output_dim, args.target_dim),
                    activation_fn,
                )
        
        trainable_projection.to(torch.device("cuda"))
        return trainable_projection

    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True),
            # "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0, 0.1]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.1]),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(STORAGE_PATH, "optuna_results"),
        num_train_epochs=5,
        per_device_train_batch_size=20000,
        per_device_eval_batch_size=20000,
        metric_for_best_model="eval_loss",
        eval_strategy="steps",
        eval_steps=100,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=None,
        load_best_model_at_end=False,
        greater_is_better=False,
        dataloader_drop_last=True,
        disable_tqdm=True,
        dataloader_pin_memory=True,
        warmup_ratio=0.1,
    )

    trainer = SimilarityTrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        target_dim=args.target_dim,
        positional_loss_factor=0.0,
        weight_exponent=0,
        data_collator=collate_embeddings,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, 
                early_stopping_threshold=0.01
            )
        ],
    )
    
    # Run hyperparameter search
    trials = trainer.hyperparameter_search(
        hp_space=hp_space,
        direction="minimize",
        backend="optuna",
        n_trials=30,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage="sqlite:///optuna_study.db",
        study_name="projection_search_2",
        load_if_exists=True,
    )
    print(trials)

    save_to_csv(output_filepath=os.path.join(STORAGE_PATH, "optuna_results", "all_trials.csv"))