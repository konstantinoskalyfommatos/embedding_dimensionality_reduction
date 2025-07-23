"""Distill Jina v2 model using Wikisplit dataset.

# NOTE: Make sure to:
1. Set the PROJECT_ROOT environment variable in your .env file.
2. Precalculate the embeddings by running the script `src/scripts/precalculate_embeddings.py`.
"""

import os
from torch.utils.data import RandomSampler

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from argparse import ArgumentParser
from dotenv import load_dotenv

from core.student import Student
from core.teacher import Teacher
from core.train import train
from core.eval_functions import eval_intrinsic_original_vs_projected_space

from utils.custom_datasets.wikisplit_dataset import WikisplitDataset, PrecalculatedWikisplitDataset
from utils.datasets_info import get_dataset_max_length
from utils.embedding_precalculation import get_precalculated_embeddings_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


def main():
    parser = ArgumentParser(description="Train a distilled Jina model")
    parser.add_argument("--low_dim_size", type=int, default=256, help="Size of the low-dimensional space")
    parser.add_argument("--hidden_size", type=int, default=None, help="Size of the hidden layer in the projection network")
    parser.add_argument("--freeze_backbone", action='store_true', default=False , help="Whether to finetune the backbone model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Starting learning rate for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--warmup_validation_epochs", type=int, default=10, help="Number of warmup epochs before computing validation loss")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping")
    args = parser.parse_args()

    parser.parse_args()

    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    if not PROJECT_ROOT:
        raise ValueError(
            "PROJECT_ROOT environment variable is not set. "
            "Please set it in your .env file."
        )

    print(
        f"Started distillation process for Jina v2 model. "
        f"We have set freeze_backbone={args.freeze_backbone}"
    )

    ds = load_dataset("cl-nagoya/wikisplit-pp")

    # Find the maximum sentence length in the dataset
    tokenizer = AutoTokenizer.from_pretrained(
        'jinaai/jina-embeddings-v2-small-en', 
        trust_remote_code=True
    )
    max_seq_length = get_dataset_max_length(
        "cl-nagoya/wikisplit-pp", 
        tokenizer
    )
    print(f"Maximum sequence length for the dataset: {max_seq_length}")

    generator = torch.Generator()
    generator.manual_seed(42)

    # Student datasets and dataloaders
    if args.freeze_backbone:
        student_train_dataset = PrecalculatedWikisplitDataset(
            get_precalculated_embeddings_dataset(
                dataset_name="cl-nagoya/wikisplit-pp",
                model_name="jinaai/jina-embeddings-v2-small-en",
                split="train",
            )
        )
        student_val_dataset = PrecalculatedWikisplitDataset(
            get_precalculated_embeddings_dataset(
                dataset_name="cl-nagoya/wikisplit-pp",
                model_name="jinaai/jina-embeddings-v2-small-en",
                split="validation",
            )
        )
    else:
        student_train_dataset = WikisplitDataset(
            ds["train"],
            tokenizer=tokenizer,
            max_length=max_seq_length
        )
        student_val_dataset = WikisplitDataset(
            ds["validation"],
            tokenizer=tokenizer,
            max_length=max_seq_length
        )
        
    student_train_sampler = RandomSampler(student_train_dataset, generator=generator)
    student_train_loader = DataLoader(
        student_train_dataset,
        batch_size=args.low_dim_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=student_train_sampler,
        num_workers=4,
    )

    student_val_loader = DataLoader(
        student_val_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )

    # Teacher datasets and dataloaders
    teacher_train_dataset = PrecalculatedWikisplitDataset(
        get_precalculated_embeddings_dataset(
            dataset_name="cl-nagoya/wikisplit-pp",
            model_name="jinaai/jina-embeddings-v2-small-en",
            split="train",
        )           
    )
    teacher_train_sampler = RandomSampler(teacher_train_dataset, generator=generator)
    teacher_train_loader = DataLoader(
        teacher_train_dataset,
        batch_size=args.low_dim_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=teacher_train_sampler,
        num_workers=4,
    )

    teacher_val_dataset = PrecalculatedWikisplitDataset(
        get_precalculated_embeddings_dataset(
            dataset_name="cl-nagoya/wikisplit-pp",
            model_name="jinaai/jina-embeddings-v2-small-en",
            split="validation",
        )
    )
    teacher_val_loader = DataLoader(
        teacher_val_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    encoder = SentenceTransformer(
        "jinaai/jina-embeddings-v2-small-en",
        trust_remote_code=True
    ).to("cuda")
    encoder.max_seq_length = max_seq_length
    if args.hidden_size:
        hidden_size = args.hidden_size
    else:
        hidden_size = (encoder.get_sentence_embedding_dimension() + args.low_dim_size // 2)
        hidden_size = hidden_size if hidden_size % 2 == 0 else hidden_size - 1
    
    projection_net = nn.Sequential(
        nn.Linear(encoder.get_sentence_embedding_dimension(), hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, args.low_dim_size),
    )
        
    student = Student(
        backbone=encoder, 
        projection_net=projection_net,
        freeze_backbone=args.freeze_backbone
    )
    # We do not use the backbone for the teacher as we have precalculated the embeddings
    teacher = Teacher(backbone=encoder, use_backbone=False)

    distilled_model_path = os.path.join(
        PROJECT_ROOT,
        f"storage/models/jina-embeddings-v2-small-en_{args.low_dim_size}.pth"
    )
    os.makedirs(os.path.dirname(distilled_model_path), exist_ok=True)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    train(
        student=student,
        teacher=teacher,
        student_train_loader=student_train_loader,
        student_val_loader=student_val_loader,
        teacher_train_loader=teacher_train_loader,
        teacher_val_loader=teacher_val_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        validation_fn=eval_intrinsic_original_vs_projected_space,
        model_path=distilled_model_path,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        use_precalculated_student_embeddings=args.freeze_backbone,
        warmup_validation_epochs=args.warmup_validation_epochs
    )
    
    
if __name__ == "__main__":
    main()
