import os
import torch
import random
from model import MiniGPT
from config import *
from dataloader import get_dataloader
from spm_tokenizer import SPTokenizer

os.makedirs("checkpoints/tigrinya", exist_ok=True)

tokenizer = SPTokenizer("checkpoints/tigrinya/spm.model")

model = MiniGPT(
    tokenizer.vocab_size,
    BLOCK_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

text = open("data/tigrinya.txt", "r", encoding="utf-8").read()

# -----------------------------
# Train / validation split
# -----------------------------
# split_idx = int(len(text) * 0.9)

# train_text = text[:split_idx]
# val_text = text[split_idx:]

# train_loader = get_dataloader(
#     text=train_text,
#     tokenizer=tokenizer,
#     block_size=BLOCK_SIZE,
#     batch_size=BATCH_SIZE
# )

# val_loader = get_dataloader(
#     text=val_text,
#     tokenizer=tokenizer,
#     block_size=BLOCK_SIZE,
#     batch_size=BATCH_SIZE
# )

checkpoint_path = "checkpoints/tigrinya/latest.pt"
best_checkpoint_path = "checkpoints/tigrinya/best.pt"

start_step = 0
best_val_loss = float("inf")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print("Resuming from step:", start_step)


# -----------------------------
# Validation function
# -----------------------------
@torch.no_grad()
def estimate_val_loss(model, val_loader, max_batches=20):
    model.eval()

    losses = []

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break

        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, y)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


model.train()

MAX_STEPS = 20000 + start_step
step = start_step

while step < MAX_STEPS:
    examples = text.split("<eos>")

    examples = [
        ex.strip() + " <eos>"
        for ex in examples
        if ex.strip()
    ]

    random.shuffle(examples)

    shuffled_text = "\n".join(examples)
    # -----------------------------
    # Train / validation split
    # -----------------------------
    split_idx = int(len(text) * 0.9)

    train_text = shuffled_text[:split_idx]
    val_text = shuffled_text[split_idx:]

    train_loader = get_dataloader(
    train_text,
    tokenizer,
    BLOCK_SIZE,
    BATCH_SIZE
    )
    val_loader = get_dataloader(
        text=val_text,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE
        )
   
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss = loss.item()

        if step % 100 == 0:
            print(f"step {step} | train loss {train_loss:.4f}")

        # -----------------------------
        # Run validation every 500 steps
        # -----------------------------
        if step % 500 == 0:
            val_loss = estimate_val_loss(model, val_loader)

            print(
                f"step {step} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
            )

            # Save latest checkpoint
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "vocab_size": tokenizer.vocab_size,
                "block_size": BLOCK_SIZE,
            }, checkpoint_path)

            print(f"Saved latest checkpoint at step {step}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "vocab_size": tokenizer.vocab_size,
                    "block_size": BLOCK_SIZE,
                }, best_checkpoint_path)

                print(f"Saved BEST checkpoint at step {step} | val loss {val_loss:.4f}")

        step += 1

        if step >= MAX_STEPS:
            break