import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

with open('generated_dataset_gpt2-medium.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained('./.hf_models/gpt2-medium')
tensors = []
for d in data:
    pt_tensor = tokenizer.encode(d["story"], return_tensors="pt")
    tensors.append(pt_tensor)

torch.save(tensors, "stories_tokenized.pt")