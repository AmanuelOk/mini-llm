import sentencepiece as spm

class SPTokenizer:
    def __init__(self, model_path="checkpoints/tigrinya/spm.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.vocab_size = self.sp.get_piece_size()
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)