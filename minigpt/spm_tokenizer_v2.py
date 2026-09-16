import sentencepiece as spm

class SPTokenizer:
    def __init__(self, model_path="checkpoints/tigrinya/spm.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.vocab_size = self.sp.get_piece_size()

        # SentencePiece returns -1 if a special token isn't defined in the model
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 0
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() != -1 else 1
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 2
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 3

    def encode(self, text, add_bos=False, add_eos=False):
        """Encodes a single string into token IDs.
        
        Leaves BOS/EOS addition disabled by default so dataset classes 
        (like PretokenizedGPTDataset) can handle token boundaries cleanly.
        """
        if not text.strip():
            return []

        ids = self.sp.encode(text, out_type=int)

        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids

        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]

        return ids

    def encode_as_ids(self, text):
        """Compatibility wrapper for HuggingFace/SentencePiece style calls."""
        return self.encode(text)

    def decode(self, ids):
        """Decodes a list or tensor of token IDs back into string."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.sp.decode(ids)