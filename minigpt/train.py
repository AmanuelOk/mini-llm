import os
import torch

from model import MiniGPT
from config import *
from dataloader import get_dataloader
from spm_tokenizer import SPTokenizer

os.makedirs("checkpoints", exist_ok=True)

tokenizer = SPTokenizer("checkpoints/spm.model")

model = MiniGPT(
    tokenizer.vocab_size,
    BLOCK_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

text = open("data/spm_50000.txt", "r", encoding="utf-8").read()

loader = get_dataloader(
    text=text,
    tokenizer=tokenizer,
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE
)

checkpoint_path = "checkpoints/latest.pt"

start_step = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"] + 1
    print("Resuming from step:", start_step)

model.train()

MAX_STEPS = 20000
best_loss = float("inf")
step = start_step

while step < MAX_STEPS:
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()

        if step % 100 == 0:
            print(f"step {step} | loss {loss_val:.4f}")

        if step % 500 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_val,
                "vocab_size": tokenizer.vocab_size,
                "block_size": BLOCK_SIZE,
            }, "checkpoints/latest.pt")

            print(f"Saved checkpoint at step {step}")

        step += 1

        if step >= MAX_STEPS:
            break