"""Script to precalculate embeddings for a dataset using a specified model.

Saves the embeddings to disk for later use.
"""

from argparse import ArgumentParser
import os
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer  
from utils.embedding_precalculation import calculate_embeddings, get_dataset_max_length
from src.utils.custom_datasets import TokenizedDataset
from dotenv import load_dotenv
import sys


def precalculate_embeddings(
    model_name: str,
    dataset_path: str,
    batch_size: int,
    text_column: str,
    dataset_name: str = None,
):
    """Precalculate embeddings for a dataset using a specified model.

    Saves the embeddings to disk for later use.
    """
    def is_short_enough_batch(examples, max_len=200):
        # examples: list of dicts
        texts = [ex[text_column] for ex in examples]
        tokenized = tokenizer(texts, padding=False, truncation=False)
        return [len(ids) < max_len for ids in tokenized["input_ids"]]

    output_base_path = os.path.join(
        os.getenv("PROJECT_ROOT"),
        "storage/precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "_")
    )

    model = SentenceTransformer(model_name, trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    splits = ["train"]
    # Limit to 600k entries per split, useful for large datasets
    max_entries = 3_000_000
    for split in splits:
        print(f"Processing split: {split}")
        ds_split = load_dataset(
            dataset_path, 
            dataset_name, 
            split=split, 
            streaming=True
        )
        print(f"Loaded {split} split from dataset: {dataset_path}")

        batch_size_filter = 300_000
        filtered_examples = []
        batch = []
        total_seen = 0
        for ex in ds_split:
            batch.append(ex)
            total_seen += 1
            if total_seen >= max_entries:
                break
            if len(batch) == batch_size_filter:
                mask = is_short_enough_batch(batch)
                filtered_examples.extend([ex for ex, keep in zip(batch, mask) if keep])
                batch = []
        # Process any remaining examples
        if batch:
            mask = is_short_enough_batch(batch)
            filtered_examples.extend([ex for ex, keep in zip(batch, mask) if keep])

        print(f"Filtered {split} dataset to {len(filtered_examples)} examples.")

        tokenized_dataset = TokenizedDataset(
            [ex[text_column] for ex in filtered_examples],
            tokenizer=tokenizer,
            # max_length=get_dataset_max_length(dataset_path, tokenizer)
            max_length=200
        )
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

        embeddings = calculate_embeddings(model, dataloader)

        split_output_path = os.path.join(output_base_path, f"{split}_embeddings.pt")
        os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
        torch.save(embeddings, split_output_path)

        print(f"Saved {split} embeddings to {split_output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Precalculate embeddings for a dataset")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--dataset_path", type=str, default="cl-nagoya/wikisplit-pp")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="simple_original")
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    load_dotenv()

    precalculate_embeddings(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        text_column=args.text_column
    )
