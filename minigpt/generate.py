import torch
from model import MiniGPT
from config import *

model = MiniGPT(
    VOCAB_SIZE,
    BLOCK_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for step, (x, y) in enumerate(dataloader):

    x, y = x.to(DEVICE), y.to(DEVICE)

    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(loss.item())