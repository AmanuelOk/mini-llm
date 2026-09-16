from torch.utils.data import DataLoader
from dataset import GPTDataset
from dataset import collate_fn
import random

def get_dataloader(examples, tokenizer, block_size, batch_size):

    shuffled_text = "\n\n".join(examples)
    print(shuffled_text)
    
    dataset = GPTDataset(shuffled_text, tokenizer, block_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return loader