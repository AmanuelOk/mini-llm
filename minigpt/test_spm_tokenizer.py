import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/spm.model")

text = "Hello, how are you?"
ids = sp.encode(text, out_type=int)
decoded = sp.decode(ids)

print(ids)
print(decoded)