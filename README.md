# transformer-groundup

I'm using TinyStories dataset from HugginFace to go through data preparation and training a transformer from scratch - a tiny GPT for children's stories!

I'm using ComputeCanada's Narval cluster for all the experiments. 

## current structure plan:

transformer-groundup/
├── model/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters and model config
│   ├── embedding.py        # Token + positional embeddings
│   ├── attention.py        # Multi-head attention + masking
│   ├── feedforward.py      # Feed-forward network
│   ├── block.py            # One transformer decoder block
│   ├── transformer.py      # The full Transformer model class
│   └── utils.py            # Masking utilities, positional encodings
├── train/
│   ├── __init__.py
│   ├── trainer.py          # Training loop, optimizer, checkpointing
│   ├── dataset.py          # Dataset loading & batching
│   ├── loss.py             # Cross-entropy loss, accuracy, etc.
│   └── eval.py             # Evaluation: perplexity, loss, sampling
├── generate/
│   └── inference.py        # Sampling/generation from trained model
├── logs/
│   └── ...                 # Training logs or plots
├── configs/
│   └── tiny.yaml           # Optional YAML config (for sweep or reproducibility)
├── scripts/
│   └── run_train.sh        # sbatch script to train on Narval
├── requirements.txt
└── README.md
