"""Evaluate the distilled Jina v2 student model on the Wikisplit test set.

- Uses precalculated embeddings for the teacher.
- Student uses its backbone for inference.
"""

import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from core.student import Student
from core.teacher import Teacher
from core.eval_functions import eval_extrinsic, eval_intrinsic
from utils.custom_datasets.wikisplit_dataset import WikisplitDataset, PrecalculatedWikisplitDataset
from utils.datasets_info import get_dataset_max_length
from utils.embedding_precalculation import get_precalculated_embeddings_dataset
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


def main():
    parser = ArgumentParser(description="Evaluate distilled Jina v2 student model")
    parser.add_argument("--low_dim_size", type=int, default=256, help="Low-dimensional size used during training")
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden size of the projection network")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for evaluation")
    parser.add_argument("--positional_loss_factor", type=float, default=0.5, help="Factor for positional loss in evaluation")
    args = parser.parse_args()

    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    if not PROJECT_ROOT:
        raise ValueError("PROJECT_ROOT environment variable is not set. Please set it in your .env file.")

    model_path = os.path.join(PROJECT_ROOT, f"storage/models/jina-embeddings-v2-small-en_{args.low_dim_size}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading test set and model from {model_path}")

    ds = load_dataset("cl-nagoya/wikisplit-pp")
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
    max_seq_length = get_dataset_max_length("cl-nagoya/wikisplit-pp", tokenizer)
    print(f"Max sequence length: {max_seq_length}")

    # Student test dataset
    student_test_dataset = WikisplitDataset(
        ds["test"],
        tokenizer=tokenizer,
        max_length=max_seq_length
    )
    student_test_loader = DataLoader(
        student_test_dataset,
        batch_size=2048,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )

    # Teacher test dataset
    teacher_test_dataset = PrecalculatedWikisplitDataset(
        get_precalculated_embeddings_dataset(
            dataset_name="cl-nagoya/wikisplit-pp",
            model_name="jinaai/jina-embeddings-v2-small-en",
            split="test",
        )
    )
    teacher_test_loader = DataLoader(
        teacher_test_dataset,
        batch_size=2048,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )

    encoder = SentenceTransformer(
        "jinaai/jina-embeddings-v2-small-en",
        trust_remote_code=True
    ).to("cuda")
    encoder.max_seq_length = max_seq_length
    if args.hidden_size is None:
        hidden_size = (encoder.get_sentence_embedding_dimension() + args.low_dim_size // 2)
        hidden_size = hidden_size if hidden_size % 2 == 0 else hidden_size - 1
    else:
        hidden_size = args.hidden_size

    projection_net = torch.nn.Sequential(
        torch.nn.Linear(encoder.get_sentence_embedding_dimension(), hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, args.low_dim_size),
    )

    student = Student(
        backbone=encoder,
        projection_net=projection_net,
        freeze_backbone=False
    )
    student.load_state_dict(torch.load(model_path, map_location="cuda"))
    student.eval()

    teacher = Teacher(backbone=encoder, use_backbone=False)

    print("Extrinsic evaluation on test set")
    extrinsic_test_loss = eval_extrinsic(
        student=student,
        teacher=teacher,
        student_val_loader=student_test_loader,
        teacher_val_loader=teacher_test_loader,
        device="cuda",
        positional_loss_factor=args.positional_loss_factor,
        use_precalculated_student_embeddings=False
    )
    print(f"Extrinsic test Loss: {extrinsic_test_loss:.4f}")

    print("Intrinsic evaluation on test set")
    intrinsic_test_loss = eval_intrinsic(
        student=student,
        student_val_loader=student_test_loader,
        teacher_val_loader=teacher_test_loader,
        device="cuda",
        positional_loss_factor=args.positional_loss_factor,
        use_precalculated_student_embeddings=False
    )
    print(f"Intrinsic test Loss: {intrinsic_test_loss:.4f}")


if __name__ == "__main__":
    main()
