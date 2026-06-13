import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.tokens = tokenizer.encode(text)

        assert len(self.tokens) > block_size + 1
        assert max(self.tokens) < tokenizer.vocab_size

    def __len__(self):
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])

    return x, y