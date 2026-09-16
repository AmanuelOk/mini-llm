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

# checkpoint = torch.load("checkpoints/tigrinya/latest.pt", map_location="cpu")
checkpoint = torch.load("checkpoints/tigrinya/best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(model, idx, max_new_tokens, eos_id, temperature=0.8, top_k=50):
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
        # Stop when EOS is generated 
        if next_token.item() == eos_id: 
            break
        idx = torch.cat([idx, next_token], dim=1)

    return idx
    
# masking to exclude noise
def generate_top_k_masked(model, idx, max_new_tokens, eos_id, temperature=0.8, top_k=50):
    count = 0
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature

        # 1. Find the threshold value of the k-th largest logit
        v, _ = torch.topk(logits, top_k)
        
        # 2. Create a mask for any logits smaller than the k-th largest value
        # logits[0, -1] gets the scalar threshold for the batch
        min_values = v[:, -1].unsqueeze(-1) 
        
        # 3. Fill excluded logits with -inf so their softmax probability becomes 0
        logits[logits < min_values] = float('-inf')

        # 4. Apply softmax to the safely masked logit vector
        probs = torch.softmax(logits, dim=-1)

        # 5. Sample directly from the full vocabulary distribution
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eos_id: 
            break
     
        idx = torch.cat([idx, next_token], dim=1)

    return idx

def generate_full_softmax(model, idx, max_new_tokens, eos_id, temperature=0.8):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Get the logits of the very last token in the sequence
        logits = logits[:, -1, :] / temperature

        # Convert all logits directly to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample from the entire distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Stop immediately if EOS is generated
        if next_token.item() == eos_id: 
            break
            
        idx = torch.cat([idx, next_token], dim=1)

    return idx

# ✅ LOAD SAME TOKENIZER USED IN TRAINING
tokenizer = SPTokenizer("checkpoints/tigrinya/spm.model")
eos_id =  tokenizer.eos_id
print(eos_id)
while(True):
   
    input_text = input("Enter a prompt: ")
    prompt_text = input_text if input_text else "Explain machine learning."

    prompt = f"""<|User|>
    {prompt_text}

    <|Assistant|>
    """
    start = torch.tensor([tokenizer.encode(prompt, add_bos=False, add_eos=False)]).to(DEVICE)


    out = generate(model, start, 200, eos_id=tokenizer.eos_id)
    # out = generate_top_k_masked(model, start, 200, eos_id=tokenizer.eos_id)
    # out = generate_full_softmax(model, start, 200, eos_id=tokenizer.eos_id)
    text = tokenizer.decode(out[0].tolist())

    print(f'{text}')
  