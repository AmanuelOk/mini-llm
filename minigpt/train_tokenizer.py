import sentencepiece as spm

# 1. Train the model correctly
spm.SentencePieceTrainer.train(
    input="data/tigrinya.txt",
    model_prefix="checkpoints/tigrinya/spm",
    vocab_size=900,
    max_sentencepiece_length=32,
    hard_vocab_limit=False,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=1,
    unk_id=0,
    pad_id=3,
    # Disable the default control EOS completely
    eos_id=2, 
    eos_piece="<eos>",
    # Explicitly pass the tokens as separate items in the list
    user_defined_symbols=["<|User|>", "<|Assistant|>"]
)

# 2. Load the processor
sp = spm.SentencePieceProcessor(
    model_file="checkpoints/tigrinya/spm.model"
)

# 3. Test text
text = "tornado"

# Encode as pieces (strings)
print("Pieces:", sp.encode_as_pieces(text))
# Expected Output: [' This', ' is', ' the', ' end', ' of', ' the', ' sentence', ' <eos>', ' and', ' this', ' is', ' a', ' new', ' one', '.']

# Encode as integer IDs
print("IDs:", sp.encode_as_ids(text))

