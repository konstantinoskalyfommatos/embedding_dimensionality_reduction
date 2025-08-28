from torch.utils.data import Dataset
import torch
from sentence_transformers import SentenceTransformer
import os
from torch.utils.data import DataLoader


class TokenizedDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)
    

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]
    

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
    dataset_path: str, 
    model_name: str,
    split: str, 
):
    output_path = os.path.join(
        os.getenv("PROJECT_ROOT"),
        "storage",
        "precalculated_embeddings",
        dataset_path.split("/")[-1],
        model_name.replace("/", "_"),
        f"{split}_embeddings.pt"
    )
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Precalculated embeddings not found at {output_path}")

    embeddings = torch.load(output_path)
    return EmbeddingsDataset(embeddings)
