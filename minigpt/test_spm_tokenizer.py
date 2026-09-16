import sentencepiece as spm
from dataset import GPTDataset
from dataset_v2 import PretokenizedGPTDataset
import re

sp = spm.SentencePieceProcessor(model_file="checkpoints/tigrinya/spm.model")

tigrinya_text = ["ኣብ መጀመርታ ኣምላኽ ሰማይን ምድርን ፈጠረ።","ምድሪ ድማ ኣልቦን ባዶን ነበረት።"]
block_size = 8

dataset = PretokenizedGPTDataset(tigrinya_text, sp, block_size=block_size)

x, y = dataset[0]
print(x)


