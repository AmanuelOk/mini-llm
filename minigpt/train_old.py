import torch
from model import MiniGPT
from config import *
from dataloader import get_dataloader
from dataset import GPTDataset
from tokenizer import CharTokenizer
from spm_tokenizer import SPTokenizer
import os
import torch
import json

os.makedirs("checkpoints", exist_ok=True)
    
model = MiniGPT(
    VOCAB_SIZE,
    BLOCK_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
text = open("data/spm_50000.txt").read()

# tokenizer = CharTokenizer(text)

# with open("checkpoints/tokenizer.json", "w") as f:
#     json.dump({
#         "stoi": tokenizer.stoi,
#         "itos": {str(k): v for k, v in tokenizer.itos.items()}
#     }, f)

tokenizer = SPTokenizer()
loader = get_dataloader(
    text=text,
    tokenizer=tokenizer,
    block_size=128,
    batch_size=16
)

checkpoint_path = "checkpoints/latest.pt"
if os.path.exists(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_step = checkpoint["step"]

    print("Resuming from step:", start_step)
else:
    start_step = 0
best_loss = 2.0
model.train()

best_loss = float("inf")

for step, (x, y) in enumerate(loader):

    x, y = x.to(DEVICE), y.to(DEVICE)

    logits, loss = model(x, y)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()

    loss_val = loss.item()

    if step % 100 == 0:
        print(loss_val)

    if loss_val < best_loss:
        best_loss = loss_val

        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_val
        }

        torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_val
                 }, "checkpoints/latest.pt")

        print(f"Saved checkpoint at step {step}")