"""Script to precalculate embeddings for a dataset using a specified model.

Saves the embeddings to disk for later use.
"""

from argparse import ArgumentParser
import os
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from utils.config import PROJECT_ROOT


class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    

def calculate_embeddings(
    model: SentenceTransformer, 
    dataloader: DataLoader, 
    device="cuda"
):
    all_embeddings = []
    with torch.no_grad():
        # for batch in tqdm(dataloader, desc="Calculating embeddings"):
        for batch in dataloader:
            embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def precalculate_train_embeddings(
    model_name: str,
    dataset_path: str,
    batch_size: int,
    text_column: str,
    dataset_name: str = None,
    max_examples: int = 2500000
):
    """Precalculate embeddings for a dataset using a specified model.

    Saves the embeddings to disk for later use.
    """
    output_base_path = os.path.join(
        PROJECT_ROOT,
        "storage/precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "__")
    )

    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True,
    ).to("cuda")

    model.max_seq_length = 300

    ds_iter = load_dataset(
        dataset_path, 
        dataset_name, 
        split="train", 
        streaming=True
    )

    examples = []
    total_seen = 0
    # for ex in tqdm(ds_iter, desc="Loading examples"):
    for ex in ds_iter:
        examples.append(ex)
        total_seen += 1
        if total_seen >= max_examples:
            break

    print(f"Filtered dataset to {len(examples)} examples.")

    texts = [ex[text_column] for ex in examples]
    dataset = CustomDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = calculate_embeddings(model, dataloader)

    split_output_path = os.path.join(output_base_path, f"train_embeddings.pt")
    os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
    torch.save(embeddings, split_output_path)

    print(f"Saved train embeddings to {split_output_path}")


def precalculate_val_test_embeddings(
    model_name: str,
    dataset_path: str,
    batch_size: int,
    max_validation_examples: int = 10000,
    max_test_examples: int = 10000
):
    """Precalculate embeddings for a dataset using a specified model.

    Saves the embeddings to disk for later use.
    """
    output_base_path = os.path.join(
        PROJECT_ROOT,
        "storage/precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "__")
    )

    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True,
    ).to("cuda")

    model.max_seq_length = 512

    ds_iter = load_dataset(
        dataset_path, 
        split="validation",  # Only validation available 
    )
    for split, max_examples in zip(
        ["validation", "test"],
        [max_validation_examples, max_test_examples]
    ):
        print(f"Processing split: {split}")

        examples = []
        total_seen = 0
        for ex in tqdm(ds_iter, desc="Loading examples"):
            text, paraphrase = ex["text"], ex["paraphrase"]
            # Keep small ones without tokenizing to save time
            if len(text.split()) < 350 and len(paraphrase.split()) < 350: 
                examples += [text, paraphrase]
                total_seen += 1
            if total_seen >= max_examples:
                break

        print(f"Filtered {split} dataset to {len(examples)} examples.")

        dataset = CustomDataset(examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = calculate_embeddings(model, dataloader)

        split_output_path = os.path.join(output_base_path, f"{split}_embeddings.pt")
        os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
        torch.save(embeddings, split_output_path)

        print(f"Saved {split} embeddings to {split_output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Precalculate embeddings for a dataset")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--train_or_eval", type=str, choices=["train", "eval"], required=True, help="Precalulate train or validation-test embeddings")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_train_examples", type=int, default=2500000, help="Maximum number of training examples to process")
    parser.add_argument("--max_val_examples", type=int, default=10000, help="Maximum number of validation examples to process")
    parser.add_argument("--max_test_examples", type=int, default=10000, help="Maximum number of test examples to process")
    args = parser.parse_args()

    if args.train_or_eval == "train":
        precalculate_train_embeddings(
            model_name=args.model_name,
            dataset_path="allenai/c4",
            dataset_name="en",
            batch_size=args.batch_size,
            text_column="text"
        )
    else:
        precalculate_val_test_embeddings(
            model_name=args.model_name,
            dataset_path="agentlans/sentence-paraphrases",
            batch_size=args.batch_size
        )