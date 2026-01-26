"""Script to precalculate embeddings for a dataset using a specified model.

Saves the embeddings to disk for later use.
"""

from argparse import ArgumentParser
import os
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import logging

from torch.utils.data import DataLoader, Dataset
from utils.config import PROJECT_ROOT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    max_examples: int = 2500000,
    encode_every: int = 1000000
):
    """Precalculate embeddings for a dataset using a specified model.

    Saves the embeddings to disk for later use.
    """
    logger.info("Precalculating training embeddings")

    output_base_path = os.path.join(
        PROJECT_ROOT,
        "storage/precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "__")
    )

    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True,
        device="cuda"
    )

    model.max_seq_length = 300

    ds_iter = load_dataset(
        dataset_path, 
        dataset_name, 
        split="train", 
        streaming=True
    )
    
    logger.info("Processing train dataset")
    
    total_seen = 0
    examples = []
    # for ex in tqdm(ds_iter, desc="Loading examples"):
    for ex in ds_iter:
        text = ex[text_column]
        # Keep small ones without tokenizing to save time
        if len(text.split()) < 200:
            continue
        examples.append(ex)
        total_seen += 1

        if total_seen % encode_every == 0:
            logger.info(f"Encoding train examples. Total seen: {total_seen}")
            texts = [ex[text_column] for ex in examples]

            dataset = CustomDataset(texts)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            embeddings = calculate_embeddings(model, dataloader)

            split_output_path = os.path.join(output_base_path, f"train_embeddings_{total_seen}.pt")
            os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
            torch.save(embeddings, split_output_path)

            logger.info(f"Saved train embeddings to {split_output_path}")
            examples = []
        if total_seen >= max_examples:
            break
    
    # Merge saved tensors
    logger.info("Merging saved train embeddings")
    all_embeddings = []
    for i in range(encode_every, total_seen + 1, encode_every):
        split_output_path = os.path.join(output_base_path, f"train_embeddings_{i}.pt")
        embeddings = torch.load(split_output_path)
        all_embeddings.append(embeddings)
        os.remove(split_output_path)  # Clean up intermediate file
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    final_output_path = os.path.join(output_base_path, "train_embeddings.pt")
    torch.save(all_embeddings, final_output_path)

    logger.info("Finished precalculating train embeddings")

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
    logger.info("Precalculating validation and test embeddings")
    output_base_path = os.path.join(
        PROJECT_ROOT,
        "storage/precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "__")
    )

    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True,
        device="cuda"
    )

    model.max_seq_length = 512

    ds_iter = iter(
        load_dataset(
            dataset_path, 
            split="validation",  # Only validation available 
        )
    )
    for split, max_examples in zip(
        ["validation", "test"],
        [max_validation_examples, max_test_examples]
    ):
        logger.info(f"Processing split: {split}")

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

        logger.info(f"Filtered {split} dataset to {len(examples) / 2} paraphrase pairs")

        dataset = CustomDataset(examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = calculate_embeddings(model, dataloader)

        split_output_path = os.path.join(output_base_path, f"{split}_embeddings.pt")
        os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
        torch.save(embeddings, split_output_path)

        logger.info(f"Saved {split} embeddings to {split_output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Precalculate embeddings for a dataset")
    
    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--train", action="store_true", help="Precalulate training embeddings")
    parser.add_argument("--val_test", action="store_true", help="Precalulate validation and test embeddings")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_train_examples", type=int, default=10000000, help="Maximum number of training examples to process")
    parser.add_argument("--max_val_examples", type=int, default=10000, help="Maximum number of validation examples to process")
    parser.add_argument("--max_test_examples", type=int, default=10000, help="Maximum number of test examples to process")
    
    args = parser.parse_args()

    if not args.train and not args.val_test:
        raise ValueError("Provide at least one of --train or --val_test arguments.")

    if args.train:
        precalculate_train_embeddings(
            model_name=args.model_name,
            dataset_path="allenai/c4",
            dataset_name="en",
            batch_size=args.batch_size,
            text_column="text",
            max_examples=args.max_train_examples
        )
    if args.val_test:
        precalculate_val_test_embeddings(
            model_name=args.model_name,
            dataset_path="agentlans/sentence-paraphrases",
            batch_size=args.batch_size,
            max_validation_examples=args.max_val_examples,
            max_test_examples=args.max_test_examples
        )
