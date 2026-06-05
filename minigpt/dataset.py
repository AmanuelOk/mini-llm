import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        tokens = tokenizer.encode(text)

        self.data = []

        # sliding window
        for i in range(0, len(tokens) - block_size - 1):
            chunk = tokens[i:i + block_size + 1]
            self.data.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        chunk = self.data[idx]

        x = chunk[:-1]   # input
        y = chunk[1:]    # target

        return x, y
        
def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    return x, y