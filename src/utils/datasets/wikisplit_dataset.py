"""Contains the Wikisplit dataset class."""

from torch.utils.data import Dataset


class WikisplitDataset(Dataset):
    def __init__(self, ds):
        self.sentences = ds["simple_original"]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    