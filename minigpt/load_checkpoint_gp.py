import torch
import sys
from model import MiniGPT
from config import *
from spm_tokenizer_v2 import SPTokenizer
import re
model = MiniGPT(
    VOCAB_SIZE,
    BLOCK_SIZE,
    D_MODEL,
    N_HEADS,
    N_LAYERS
).to(DEVICE)

checkpoint = torch.load("checkpoints/tigrinya/best.pt", map_location="mps")
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
         # Stop immediately if EOS is generated
        if next_token.item() == eos_id: 
            break

        idx = torch.cat([idx, next_token], dim=1)

    return idx


def generate_word_by_word(
    pipeline, tokenizer,
    initial_prompt: str,
    max_steps: int = 5,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
  """Generates text word-by-word by running inference up to max_steps times.

  At each step, it generates tokens until a space/delimiter is hit, extracts
  the first word, appends it to the prompt, and feeds it back into the model.
  """
  current_prompt = initial_prompt
  print(f"Initial Prompt: '{current_prompt}'\n" + "=" * 50)

  for step in range(1, max_steps + 1):
    # 1. Encode the accumulated prompt
    tokens = tokenizer.encode(current_prompt, add_bos=True)
    start_tensor = torch.tensor([tokens], dtype=torch.long, device="mps")

    # 2. Generate a small buffer of tokens (e.g., 15) to ensure we capture at least one full word
    with torch.no_grad():
      generated_ids = generate(model=model,
          idx=start_tensor,
          max_new_tokens=15,
          eos_id=tokenizer.eos_id,
          temperature=temperature,
          top_k=top_k,
      )

    # 3. Decode only the NEW tokens generated in this iteration
    new_token_ids = generated_ids[0][start_tensor.shape[1] :].tolist()
    raw_generated_text = tokenizer.decode(new_token_ids).strip()

    # If model output is empty or stopped at EOS, terminate early
    if not raw_generated_text:
      print(f"[Step {step}] EOS or empty generation reached.")
      break

    # 4. Extract ONLY the first word
    # .split() handles whitespace/newlines across Ge'ez/Tigrinya and Latin text
    first_word = raw_generated_text.split()[0]

    # 5. Append the extracted word to the current prompt
    current_prompt = f"{current_prompt} {first_word}".strip()

    print(f"Step {step}: Extracted Word -> '{first_word}'")
    print(f"Updated Prompt: '{current_prompt}'\n" + "-" * 50)

  return current_prompt


# # ✅ LOAD SAME TOKENIZER USED IN TRAINING
tokenizer = SPTokenizer("checkpoints/tigrinya/spm.model")
eos_id = tokenizer.eos_id
print(eos_id)
while(True):
   
    input_text = input("Enter a prompt: ")
    prompt_text = input_text if input_text else "ብሉይን ሓድሽን ኪዳን"
    count = 0

    prompt_text = prompt_text.strip()
    start = torch.tensor([tokenizer.encode(prompt_text, add_bos=True)]).to(DEVICE)
    # out = generate(model, start, 200, eos_id)
    out = generate(model,start,200, eos_id, temperature=0.8, top_k=50)
    # out = generate(model, start, 50)
    text = tokenizer.decode(out[0].tolist())
    print(f'{text}')

# --- Example Execution Integration ---
# if __name__ == "__main__":
#   starting_prompt = input("add ur prompt here\n----> : ")
#   if not starting_prompt:
#        starting_prompt = "ብሉይን ሓድሽን"
#   elif starting_prompt == 'quit':
#      sys.exit(0)
#   # Run word-by-word autoregressive loop
#   final_text = generate_word_by_word(
#       pipeline=model,
#       tokenizer=tokenizer,
#       initial_prompt=starting_prompt,
#       max_steps=5,  # Strictly limits loop to 5 appended words
#       temperature=0.8,
#       top_k=50,
#   )

#   print(f"\nFinal Accumulated Text:\n{final_text}")