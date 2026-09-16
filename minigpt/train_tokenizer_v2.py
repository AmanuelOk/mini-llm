import sentencepiece as spm

# 1. Train the model optimized specifically for Pure Tigrinya (Ge'ez Script)
spm.SentencePieceTrainer.train(
    input="data/bible_2.txt",
    model_prefix="checkpoints/tigrinya/spm",
    
    # 4,000-6,000 is optimal for a Bible-sized Tigrinya corpus (~1M-2M tokens)
    vocab_size=5000,
    model_type="unigram",
    hard_vocab_limit=False,
    
    # Keep true for standard Transformer / LLM input representation
    add_dummy_prefix=True,
    
    # 1.0 ensures no rare Ge'ez characters are mapped to byte-fallback
    character_coverage=1.0,
    byte_fallback=True,
    
    # Limit max piece length to prevent learning huge whole-sentence chunks
    max_sentencepiece_length=16,
    
    # Preserve whitespace boundaries and Ge'ez word separators
    split_by_whitespace=True,
    
    # Standard Reserved IDs
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    
    # Protect traditional Ge'ez/Ethiopic punctuation marks as standalone tokens
    user_defined_symbols=[
        "፡",  # Ge'ez word space (ለየን)
        "።",  # Ge'ez full stop (ዓራት ነጥቢ)
        "፤",  # Ge'ez semicolon (ሠረዝ)
        "፥",  # Ge'ez colon (ነጥብ ሠረዝ)
        "፦",  # Ge'ez preface colon (ነጥብ)
        "፧",  # Ge'ez question mark
    ]
)

# 2. Load the processor
sp = spm.SentencePieceProcessor(model_file="checkpoints/tigrinya/spm.model")

# 3. Validation Test with Tigrinya text
test_tigrinya = "ኣብ መጀመርታ ኣምላኽ ሰማይን ምድርን ፈጠረ።"

print("Tigrinya Pieces:", sp.encode_as_pieces(test_tigrinya))
print("Tigrinya IDs:", sp.encode_as_ids(test_tigrinya))

# Unknown English word (will fall back to bytes cleanly)
print("English Fallback Pieces:", sp.encode_as_pieces("tornado"))