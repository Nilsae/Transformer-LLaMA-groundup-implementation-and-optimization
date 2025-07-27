import torch
import json
import torch.nn as nn
import torch.optim as optim
from model.transformer import TransformerDecoder, TransformerEncoder
from transformers import AutoTokenizer
from torch.utils.data import  Dataset, DataLoader


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class TokenizedDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        logger.info(sample["input_ids"].shape)
        logger.info(sample["attention_mask"].shape)
        
        squeezed_sample = {
            "input_ids" : sample["input_ids"].view(-1),  # or .squeeze(0)
            "attention_mask" : sample["attention_mask"].view(-1)
        }
        return squeezed_sample
        
    
    
    
    
with open('data/generated_dataset_gpt2-medium.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained('./.hf_models/gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
tokenized_data = []
for d in data:
    tokenizer_out = tokenizer(d["story"], return_tensors="pt", padding="max_length", truncation=True, return_attention_mask=True)
    tokenized_d = {
        "input_ids" : tokenizer_out["input_ids"],
        "attention_mask" : tokenizer_out["attention_mask"]
    }
    tokenized_data.append(tokenized_d)

torch.save(tokenized_data, "tokenized_data.pt")
train_loader = DataLoader(TokenizedDataset(tokenized_data), batch_size = 16, shuffle=True)

encoder = TransformerEncoder(vocab_size = 16, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, is_autoregressive = True, dropout_rate = 0.1)
