from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32
)

device = "cpu"
model.to(device)

messages = [
    {"role": "user", "content": "hi"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
tokenizer.pad_token_id = tokenizer.eos_token_id
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False
    )

generated = output[0][inputs.input_ids.shape[1]:]

print(generated)

text = tokenizer.decode(
    generated,
    skip_special_tokens=True
)

print(text)