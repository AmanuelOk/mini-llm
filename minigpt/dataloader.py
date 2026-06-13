from torch.utils.data import DataLoader
from dataset import GPTDataset
from dataset import collate_fn
import random

def get_dataloader(text, tokenizer, block_size, batch_size):

    examples = text.split("<eos>")

    examples = [
        ex.strip() + " <eos>"
        for ex in examples
        if ex.strip()
    ]

    random.shuffle(examples)

    shuffled_text = "\n".join(examples)

    dataset = GPTDataset(shuffled_text, tokenizer, block_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return loader