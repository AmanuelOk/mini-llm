import torch

# Tokenizer & Context
VOCAB_SIZE = 5000
BLOCK_SIZE = 64        # Increased from 32 for better context without heavy CPU penalty

# Architecture (~3.3M Parameters)
D_MODEL = 256          # Embedding & hidden dimension
N_HEADS = 4            # Head dimension = 256 / 4 = 64
N_LAYERS = 4           # Transformer block depth
DROPOUT = 0.1          # Regularization

# Training
BATCH_SIZE = 16
LR = 3e-4              # Base learning rate for AdamW
WEIGHT_DECAY = 0.01

# Hardware Setup
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(DEVICE)