from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32
)

device = "cpu"
model.to(device)

while True:
    user = input("You: ")

    messages = [{"role": "user", "content": user}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )
        print(output[0][inputs.input_ids.shape[1]:])
        generated_tokens = output[0][inputs.input_ids.shape[1]:]

        text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
            ).strip()

    print("Assistant:", text)