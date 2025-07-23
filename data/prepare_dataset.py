import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class TokenizedDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        squeezed_sample = {
            "input_ids" : sample["input_ids"].view(-1),
            "attention_mask" : sample["attention_mask"].view(-1)
        }
        return squeezed_sample
        
    
    
    
    
with open('data/generated_dataset_gpt2-medium.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained('./.hf_models/gpt2-medium')
tokenized_data = []
for d in data:
    input_ids, attention_mask = tokenizer(d["story"], return_tensors="pt", padding="max_length", truncation=True, return_attention_mask=True)
    tokenized_d = {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask
    }
    tokenized_data.append(tokenized_d)

torch.save(tokenized_data, "tokenized_data.pt")
train_loader = DataLoader(TokenizedDataset(tokenized_data), batch_size = 16, shuffle=True)