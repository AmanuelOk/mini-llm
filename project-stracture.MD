mini-llm/
│
├── configs/
│   ├── model.yaml
│   ├── training.yaml
│   └── inference.yaml
│
├── tokenizer/
│   ├── tokenizer.json
│   ├── vocab.json
│   └── merges.txt
│
├── checkpoints/
│   ├── step_10000.pt
│   ├── step_50000.pt
│   ├── best.pt
│   └── latest.pt
│
├── model/
│   ├── attention.py
│   ├── mlp.py
│   ├── block.py
│   ├── embeddings.py
│   └── gpt.py
│
├── training/
│   ├── dataset.py
│   ├── trainer.py
│   ├── optimizer.py
│   └── scheduler.py
│
├── inference/
│   ├── generate.py
│   ├── chat.py
│   ├── server.py
│   └── stream.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── benchmarks.py
│   └── samples.py
│
├── scripts/
│   ├── train.py
│   ├── export.py
│   ├── convert.py
│   └── quantize.py
│
└── README.md