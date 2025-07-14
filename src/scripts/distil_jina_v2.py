from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

from utils.distilled_model import DistilledModel
from utils.datasets.wikisplit_dataset import WikisplitDataset
from utils.datasets.datasets_info import get_dataset_max_length
from utils.train import train_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


def main():
    ds = load_dataset("cl-nagoya/wikisplit-pp")
    dataset = WikisplitDataset(ds["train"])

    LOW_DIM_SIZE = 256

    dataloader = DataLoader(
        dataset,
        batch_size=LOW_DIM_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    encoder = SentenceTransformer(
        "jinaai/jina-embeddings-v2-small-en",
        trust_remote_code=True
    ).to("cuda")

    # Find the maximum sentence length in the dataset
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
    encoder.max_seq_length = get_dataset_max_length("cl-nagoya/wikisplit-pp", tokenizer)

    distilled_model = DistilledModel(
        encoder, 
        low_dim_size=LOW_DIM_SIZE,
        hidden_size=368,
        finetune_backbone=True,
    )
    
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    if not PROJECT_ROOT:
        raise ValueError(
            "PROJECT_ROOT environment variable is not set. "
            "Please set it in your .env file."
        )

    distilled_model_path = os.path.join(
        PROJECT_ROOT,
        "models/distilled_jina_v2.pth"
    )
    checkpoint_path = distilled_model_path.replace('.pth', '_checkpoint.pth')
    
    optimizer = torch.optim.AdamW(distilled_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    if os.path.exists(distilled_model_path):
        state_dict = torch.load(distilled_model_path, map_location="cuda")
        distilled_model.load_state_dict(state_dict)
        print("Loaded previously trained model!")
        
        # Load optimizer and scheduler states if checkpoint exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Loaded optimizer and scheduler states!")
    
    train_model(
        distilled_model, 
        dataloader, 
        model_path=distilled_model_path,
        epochs=2, 
        lr=1e-4,
        optimizer=optimizer,
        scheduler=scheduler
    )
    print("Training complete!")


if __name__ == "__main__":
    main()
