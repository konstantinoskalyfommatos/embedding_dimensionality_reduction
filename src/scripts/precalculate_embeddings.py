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

    split_to_max_entries = {
        # "train": 2500000,
        "validation": 250000
    }
    for split, max_entries in split_to_max_entries.items():
        print(f"Processing split: {split}")
        ds_split = load_dataset(
            dataset_path, 
            dataset_name, 
            split=split, 
            streaming=True
        )
        print(f"Loaded {split} split from dataset: {dataset_path}")

        examples = []
        total_seen = 0
        # for ex in tqdm(ds_split, desc="Loading examples"):
        for ex in ds_split:
            examples.append(ex)
            total_seen += 1
            if total_seen >= max_entries:
                break

        print(f"Filtered {split} dataset to {len(examples)} examples.")

        texts = [ex[text_column] for ex in examples]
        dataset = CustomDataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = calculate_embeddings(model, dataloader)

        split_output_path = os.path.join(output_base_path, f"{split}_embeddings.pt")
        os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
        torch.save(embeddings, split_output_path)

        print(f"Saved {split} embeddings to {split_output_path}")


def precalculate_val_test_embeddings(
    model_name: str,
    dataset_path: str,
    batch_size: int,
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

    split_to_max_entries = {
        "test": 10000,
        "validation": 10000
    }
    ds_iter = load_dataset(
        dataset_path, 
        split="validation",  # Only validation available 
        # streaming=True
    )
    for split, max_entries in split_to_max_entries.items():
        print(f"Processing split: {split}")

        examples = []
        total_seen = 0
        for ex in tqdm(ds_iter, desc="Loading examples"):
            text, paraphrase = ex["text"], ex["paraphrase"]
            # Keep small ones without tokenizing to save time
            if len(text.split()) < 350 and len(paraphrase.split()) < 350: 
                examples += [text, paraphrase]
                total_seen += 1
            if total_seen >= max_entries:
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
    parser.add_argument("--dataset_path", type=str, default="allenai/c4", choices=["allenai/c4", "agentlans/sentence-paraphrases"])
    parser.add_argument("--dataset_name", type=str, default="en")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()

    if args.dataset_path == "allenai/c4":
        precalculate_train_embeddings(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            text_column=args.text_column
        )
    else:
        precalculate_val_test_embeddings(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size
        )