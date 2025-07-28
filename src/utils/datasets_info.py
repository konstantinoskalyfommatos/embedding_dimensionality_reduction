from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


# Skips recalculating the maximum length for known datasets
DATASETS_LENGTH_COUNT = {
    "cl-nagoya/wikisplit-pp": [
        {
            "tokenizer": "jinaai/jina-embeddings-v2-small-en",
            "max_length": 162,
        },
    ],
    "allenai/c4": [  # en only
        {
            "tokenizer": "jinaai/jina-embeddings-v2-small-en",
            "max_length": 200,  # We have set this limit manually
        },
    ],
}


def _get_wikisplit_max_length(
    dataset_name: str,
    tokenizer: AutoTokenizer,
) -> int:
    """Returns the maximum sentence length for the Wikisplit dataset."""    
    ds_train = load_dataset(dataset_name, split="train")
    ds_val = load_dataset(dataset_name, split="validation")
    ds_test = load_dataset(dataset_name, split="test")

    # Use concatenate_datasets to combine datasets
    ds = concatenate_datasets([ds_train, ds_val, ds_test])
    sentences = ds["simple_original"]
    
    return max(len(tokenizer.encode(sentence)) for sentence in sentences)


def get_dataset_max_length(
    dataset_name: str,
    tokenizer: AutoTokenizer,
) -> str:
    """Returns the maximum sentence length for a given dataset and tokenizer."""
    if dataset_name in DATASETS_LENGTH_COUNT:
        for entry in DATASETS_LENGTH_COUNT[dataset_name]:
            if entry["tokenizer"] == tokenizer.name_or_path:
                return entry["max_length"]
    
    match dataset_name:
        case "cl-nagoya/wikisplit-pp":
            return _get_wikisplit_max_length(dataset_name, tokenizer)
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        