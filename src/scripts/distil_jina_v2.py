import os

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise ValueError(
        "PROJECT_ROOT environment variable is not set. "
        "Please set it in your .env file."
    )

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
from core.eval_functions import eval_intrinsic

from custom_datasets.wikisplit_dataset import WikisplitDataset
from custom_datasets.datasets_info import get_dataset_max_length

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

print("Finished loading environment variables and imports.")

def main():
    parser = ArgumentParser(description="Train a distilled Jina model")
    parser.add_argument("--low_dim_size", type=int, default=256, help="Size of the low-dimensional space")
    parser.add_argument("--hidden_size", type=int, default=312, help="Size of the hidden layer in the projection network")
    parser.add_argument("--finetune_backbone", action='store_true',default=True , help="Whether to finetune the backbone model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    args = parser.parse_args()

    parser.parse_args()

    print("Started distillation process for Jina v2 model")

    ds = load_dataset("cl-nagoya/wikisplit-pp")

    train_dataset = WikisplitDataset(ds["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.low_dim_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataset = WikisplitDataset(ds["validation"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=2024,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    encoder = SentenceTransformer(
        "jinaai/jina-embeddings-v2-small-en",
        trust_remote_code=True
    ).to("cuda")

    # Find the maximum sentence length in the dataset
    tokenizer = AutoTokenizer.from_pretrained(
        'jinaai/jina-embeddings-v2-small-en', 
        trust_remote_code=True
    )
    encoder.max_seq_length = get_dataset_max_length(
        "cl-nagoya/wikisplit-pp", 
        tokenizer
    )
    
    projection_net = nn.Sequential(
        nn.Linear(encoder.get_sentence_embedding_dimension(), args.hidden_size),
        nn.ReLU(),
        nn.Linear(args.hidden_size, args.low_dim_size),
    )
        
    student = Student(
        backbone=encoder, 
        projection_net=projection_net,
        finetune_backbone=args.finetune_backbone
    )
    teacher = Teacher(backbone=encoder)

    distilled_model_path = os.path.join(
        PROJECT_ROOT,
        "models/distilled_jina_v2.pth"
    )
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    train(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        validation_fn=eval_intrinsic,
        model_path=distilled_model_path,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    
if __name__ == "__main__":
    main()
