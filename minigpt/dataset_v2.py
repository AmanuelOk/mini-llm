import random
import torch
from torch.utils.data import DataLoader, Dataset


class PretokenizedGPTDataset(Dataset):
    def __init__(self, examples, tokenizer, block_size, shuffle_docs=True):
        self.block_size = block_size
        
        # 1. Shuffle documents at the list level if requested
        if shuffle_docs:
            examples = list(examples)
            random.shuffle(examples)

        # 2. Get EOS token ID from tokenizer
        eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else tokenizer.eos_token_id

        # 3. Tokenize each document separately and inject <eos> boundaries
        all_tokens = []
        for doc in examples:
            if not doc.strip():
                continue
                
            if hasattr(tokenizer, "encode_as_ids"):
                ids = tokenizer.encode_as_ids(doc)
            else:
                ids = tokenizer.encode(doc)
                
            all_tokens.extend(ids)
            if eos_id is not None:
                all_tokens.append(eos_id)

        # Convert to PyTorch tensor for fast indexing
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        # 4. Calculate total non-overlapping blocks
        self.num_samples = (len(self.tokens) - 1) // block_size

        if self.num_samples == 0:
            raise ValueError(
                f"Total tokenized length ({len(self.tokens)}) is too short "
                f"for block_size ({block_size}). Need at least {block_size + 1} tokens."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1

        chunk = self.tokens[start_idx:end_idx]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    return x, y


def get_dataloader(examples, tokenizer, block_size, batch_size, num_workers=2, pin_memory=True):
    dataset = PretokenizedGPTDataset(examples, tokenizer, block_size, shuffle_docs=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle contiguous blocks
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True  # Ensures uniform batch shape on final step
    )

    return loader