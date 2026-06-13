import torch
from model import MiniGPT
from config import *
from tokenizer import CharTokenizer, load_tokenizer
from spm_tokenizer import SPTokenizer

model = MiniGPT(
    VOCAB_SIZE,
    BLOCK_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS
).to(DEVICE)

checkpoint = torch.load("checkpoints/tigrinya/best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :]
     
        logits = logits / temperature

        top_k = 50
        values, indices = torch.topk(logits, top_k)

        probs = torch.softmax(values, dim=-1)

        next_token = indices.gather(
            -1,
            torch.multinomial(probs, 1)
        )

        idx = torch.cat([idx, next_token], dim=1)

    return idx



# ✅ LOAD SAME TOKENIZER USED IN TRAINING
tokenizer = SPTokenizer("checkpoints/tigrinya/spm.model")
while(True):
   
    input_text = input("Enter a prompt: ")
    prompt_text = input_text if input_text else "ብሉይን ሓድሽን ኪዳን"


    start = torch.tensor([tokenizer.encode(prompt_text, add_bos=True)]).to(DEVICE)

    out = generate(model, start, 50)
    text = tokenizer.decode(out[0].tolist())

    print(f'{text}')
