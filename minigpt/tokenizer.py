import json
class CharTokenizer:

    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])


def load_tokenizer(path):
    with open(path, "r") as f:
        data = json.load(f)

    tokenizer = CharTokenizer("")  # dummy init
    tokenizer.stoi = data["stoi"]
    tokenizer.itos = {int(k): v for k, v in data["itos"].items()}

    return tokenizer