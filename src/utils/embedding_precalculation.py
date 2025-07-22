import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()

from utils.custom_datasets.wikisplit_dataset import WikisplitDataset, PrecalculatedWikisplitDataset
from utils.datasets_info import get_dataset_max_length


def calculate_embeddings(
    model: SentenceTransformer, 
    dataloader: DataLoader, 
    device="cuda"
):
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            embeddings = model(
                {
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask
                }
            )["sentence_embedding"]
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def get_precalculated_embeddings_dataset(
    dataset_name: str, 
    model_name: str,
    split: str, 
):
    output_path = os.path.join(
        os.getenv("PROJECT_ROOT"),
        "storage",
        "precalculated_embeddings",
        dataset_name.split("/")[-1],
        model_name.replace("/", "_"),
        f"{split}_embeddings.pt"
    )
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Precalculated embeddings not found at {output_path}")

    embeddings = torch.load(output_path)
    return PrecalculatedWikisplitDataset(embeddings)


def precalculate_embeddings(
    model_name: str,
    dataset_name: str,
    batch_size: int
):
    """Precalculate embeddings for a dataset using a specified model.

    Saves the embeddings to disk for later use.
    """
    output_base_path = os.path.join(
        os.getenv("PROJECT_ROOT"),
        "storage",
        "precalculated_embeddings",
        dataset_name.split("/")[-1],
        model_name.replace("/", "_")
    )

    model = SentenceTransformer(model_name, trust_remote_code=True).to("cuda")

    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    splits = ["train", "validation", "test"]
    for split in splits:
        print(f"Processing split: {split}")
        ds_split = dataset[split]
        custom_dataset = WikisplitDataset(
            ds_split, 
            tokenizer=tokenizer, 
            max_length=get_dataset_max_length(dataset_name, tokenizer)
        )
        dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

        embeddings = calculate_embeddings(model, dataloader)
        split_output_path = os.path.join(output_base_path, f"{split}_embeddings.pt")
        os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
        torch.save(embeddings, split_output_path)
        print(f"Saved {split} embeddings to {split_output_path}")
