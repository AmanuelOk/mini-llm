import os
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from config import *
from spm_tokenizer_v2 import SPTokenizer
from pathlib import Path
import sys
import math

os.makedirs("checkpoints/tigrinya", exist_ok=True)

# -----------------------------
# 1. Pretokenized Dataset Definition
# -----------------------------
class PretokenizedGPTDataset(Dataset):
    def __init__(self, examples, tokenizer, block_size, shuffle_docs=True):
        self.block_size = block_size
        
        if shuffle_docs:
            examples = list(examples)
            random.shuffle(examples)

        # Retrieve EOS token ID if available
        # Check if eos_id exists, and call it only if it is a method/function
        eos_id = None
        if hasattr(tokenizer, "eos_id"):
            eos_attr = getattr(tokenizer, "eos_id")
            eos_id = eos_attr() if callable(eos_attr) else eos_attr
        elif hasattr(tokenizer, "eos_token_id"):
            eos_attr = getattr(tokenizer, "eos_token_id")
            eos_id = eos_attr() if callable(eos_attr) else eos_attr
        if eos_id is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)

        all_tokens = []
        for doc in examples:
            if not doc.strip():
                continue
                
            if hasattr(tokenizer, "encode_as_ids"):
                ids = tokenizer.encode_as_ids(doc)
            elif hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(doc)
            else:
                raise TypeError("Tokenizer must support 'encode_as_ids' or 'encode'.")
                
            all_tokens.extend(ids)
            if eos_id is not None:
                all_tokens.append(eos_id)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.num_samples = (len(self.tokens) - 1) // block_size

        if self.num_samples == 0:
            raise ValueError(
                f"Total tokenized length ({len(self.tokens)}) is too short "
                f"for block_size ({block_size})."
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


def create_dataloader(examples, tokenizer, block_size, batch_size, shuffle=True):
    dataset = PretokenizedGPTDataset(examples, tokenizer, block_size, shuffle_docs=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=shuffle
    )
MIN_LR = 1e-5

def get_lr(step):

    progress = min(
        step / TOTAL_STEPS,
        1.0
    )

    cosine = (
        0.5 *
        (
            1.0 +
            math.cos(
                math.pi * progress
            )
        )
    )

    return (
        MIN_LR +
        (LR - MIN_LR) *
        cosine
    )
def get_learning_rate(step):

    steps_since_resume = max(
        0,
        step - start_step
    )

    progress = min(
        steps_since_resume / DECAY_STEPS,
        1.0
    )

    cosine_factor = (
        0.5 *
        (
            1.0 +
            math.cos(
                math.pi * progress
            )
        )
    )

    return (
        MIN_LR +
        (RESUME_LR - MIN_LR)
        * cosine_factor
    )
# -----------------------------
# 2. Tokenizer & Model Initialization
# -----------------------------
tokenizer = SPTokenizer("checkpoints/tigrinya/spm.model")

model = MiniGPT(
    tokenizer.vocab_size,
    BLOCK_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = None
# -----------------------------
# 5. Validation Function
# -----------------------------
@torch.no_grad()
def estimate_val_loss(model, val_loader, max_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


# Continuous iterator for training
def get_infinite_train_loader():
    while True:
        train_loader = create_dataloader(
            examples=train_examples,
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        for batch in train_loader:
            yield batch

# -----------------------------
# 3. Train / Validation Split
# -----------------------------
traing_files_dir = Path("/Users/amanuelokubamichael/Documents/projects/video-audio-overlay/llm/hadasErtra")
if not traing_files_dir.is_dir():
    print("training directory is not valid or is empty")
    sys.exit(1)
# for training_file in traing_files_dir.iterdir():
#     text = ""
#     with open(training_file, "r", encoding="utf-8") as file:
#         text = file.read()

#     examples = [ex.strip() for ex in re.split(r"\n+", text) if ex.strip()]
#  random.seed(42)
#     random.shuffle(examples)
all_examples = []

for training_file in sorted(
    traing_files_dir.iterdir()
):

    if not training_file.is_file():
        continue

    with open(
        training_file,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    examples = [
        ex.strip()
        for ex in re.split(
            r"\n+",
            text
        )
        if ex.strip()
    ]

    all_examples.extend(
        examples
    )
    rng = random.Random(42)

    rng.shuffle(
        all_examples
    )
   
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    val_loader = create_dataloader(
        examples=val_examples,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

# -----------------------------
# 4. Checkpoint Setup
# -----------------------------
checkpoint_path = "checkpoints/tigrinya/latest.pt"
best_checkpoint_path = "checkpoints/tigrinya/best.pt"

start_step = 0
best_val_loss = float("inf")

STEP_PACER = 3_000

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"] + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=STEP_PACER + start_step,eta_min=1e-5)
    # if checkpoint.get("scheduler_state_dict", None):
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resumed successfully from step {start_step} (Best Val Loss: {best_val_loss:.4f})")
    for group in optimizer.param_groups:
            group["lr"] = LR
    print("LR:", optimizer.param_groups[0]["lr"])

TOTAL_STEPS = start_step + STEP_PACER
DECAY_STEPS = TOTAL_STEPS - start_step
RESUME_LR = optimizer.param_groups[0]["lr"]
# Continuous iterator for training
train_iter = get_infinite_train_loader()

# -----------------------------
# 6. Training Loop
# -----------------------------
model.train()
step = start_step

while step < TOTAL_STEPS:

    current_lr = get_learning_rate(step)

    for group in optimizer.param_groups:
        group["lr"] = current_lr

    x, y = next(train_iter)
    x, y = x.to(DEVICE), y.to(DEVICE)

    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(x, y)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    train_loss = loss.item()

    if step % 100 == 0:
        print(f"Step {step:5d} / {TOTAL_STEPS} | Train Loss: {train_loss:.4f}")
        print("LR:", optimizer.param_groups[0]["lr"])

    if step > 0 and step % 500 == 0:
        val_loss = estimate_val_loss(model, val_loader)
        is_best = val_loss < best_val_loss
        
        if is_best:
            best_val_loss = val_loss

        print(f"---> Step {step:5d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {'(NEW BEST)' if is_best else ''}")

        checkpoint_data = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "vocab_size": tokenizer.vocab_size,
            "block_size": BLOCK_SIZE,
        }

        torch.save(checkpoint_data, checkpoint_path)
        if is_best:
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"Saved BEST checkpoint to '{best_checkpoint_path}'")

    step += 1
    
print("Training completed!")