from torch.utils.data import DataLoader
from dataset import GPTDataset
from dataset import collate_fn
def get_dataloader(text, tokenizer, block_size, batch_size):

    dataset = GPTDataset(text, tokenizer, block_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return loader