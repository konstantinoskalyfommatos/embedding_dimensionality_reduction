"""Script to precalculate embeddings for a dataset using a specified model.

Saves the embeddings to disk for later use.
"""
from argparse import ArgumentParser
import os
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from torch.utils.data import DataLoader, Dataset
from utils.config import PROJECT_ROOT


def make_dropout_func(do_rate):
    def f(x):
        mask = (torch.rand_like(x) < do_rate)
        return x.masked_fill(mask, 0.0) * (1.0 / (1 - do_rate))
    return f

def enable_mc_dropout(module):
    for c in module.children():
        if "dropout" in c.__class__.__name__.lower():
            if c.p > 0:
                c.forward = make_dropout_func(c.p)
        enable_mc_dropout(c)


class MyDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index]


def calculate_embeddings(
    model: SentenceTransformer, 
    dataloader: DataLoader, 
) -> torch.tensor:
    all_embeddings = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=len(batch)
            ).cpu()
        all_embeddings.append(embeddings)
        print(f"Encoded {i} batches")
        if i % 10 == 0:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)


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
    def is_text_short_enough(text, max_words=200):
        return len(text.split()) < max_words

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
    model.max_seq_length = 320

    splits = [
        "validation",
        # "train",
    ]
    max_entries = {
        "train": 2500000,
        "test": 250000
    }
    for split in splits:
        ds_split = load_dataset(
            dataset_path,
            dataset_name,
            split=split,
            streaming=True
        )
        print(f"Loaded {split} split from dataset: {dataset_path}")

        filtered_texts = []
        for ex in ds_split:
            text = ex[text_column]
            if is_text_short_enough(text):
                filtered_texts.append(text)
            if len(filtered_texts) >= max_entries[split]:
                break

        print(f"Filtered {split} dataset to {len(filtered_texts)} examples.")

        for run in ["normal", "monte_carlo"]:
            if run == "monte_carlo":
                enable_mc_dropout(model)

            dataset = MyDataset(filtered_texts)
            dataloader = DataLoader(dataset, batch_size=batch_size)

            print("Starting encoding")
            embeddings = calculate_embeddings(model, dataloader)

            split_output_path = os.path.join(
                output_base_path, 
                f"temp_{run}",
                f"{split}_embeddings.pt"
            )
            os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
            torch.save(embeddings, split_output_path)

            print(f"Saved {split} embeddings, run {run} to {split_output_path}")

        # Merge the two runs into one file, keeping pairs
        # of normal-monte_carlo embeddings one after the other
        normal_path = os.path.join(output_base_path, f"temp_normal/{split}_embeddings.pt")
        monte_carlo_path = os.path.join(output_base_path, f"temp_monte_carlo/{split}_embeddings.pt")
        
        normal_embeddings = torch.load(normal_path)
        monte_carlo_embeddings = torch.load(monte_carlo_path)
        
        # Interleave embeddings: [normal[0], mc[0], normal[1], mc[1], ...]
        merged_embeddings = torch.stack([normal_embeddings, monte_carlo_embeddings], dim=1)
        merged_embeddings = merged_embeddings.view(-1, normal_embeddings.shape[-1])
        
        # Save merged embeddings
        merged_output_path = os.path.join(output_base_path, f"{split}_embeddings.pt")
        os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)
        torch.save(merged_embeddings, merged_output_path)
        
        print(f"Merged {split} embeddings saved to {merged_output_path}")
        
        # Clean up temporary files
        os.remove(normal_path)
        os.remove(monte_carlo_path)
        os.rmdir(os.path.dirname(normal_path))
        os.rmdir(os.path.dirname(monte_carlo_path))


if __name__ == "__main__":
    parser = ArgumentParser(description="Precalculate embeddings for a dataset")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--dataset_path", type=str, default="allenai/c4")
    parser.add_argument("--dataset_name", type=str, default="en")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=768)
    args = parser.parse_args()

    precalculate_embeddings(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        text_column=args.text_column
    )
