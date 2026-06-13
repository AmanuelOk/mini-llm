import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/tigrinya.txt",          # your raw text file
    model_prefix="checkpoints/tigrinya/spm",    # saves spm.model and spm.vocab
    vocab_size=900,
    model_type="bpe",               # or "unigram"
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3,
    eos_piece="<eos>",
    user_defined_symbols="<eos>,<sos>,### User,### Assistant"
)

