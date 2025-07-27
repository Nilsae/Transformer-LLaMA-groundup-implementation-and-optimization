import torch
import json
import torch.nn as nn
import torch.optim as optim
from transformer import TransformerDecoder, TransformerEncoder
from transformers import AutoTokenizer
from torch.utils.data import  Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class TokenizedDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        
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
    tokenizer_out = tokenizer(d["story"], return_tensors="pt", padding="max_length", max_length=16, truncation=True, return_attention_mask=True)
    tokenized_d = {
        "input_ids" : tokenizer_out["input_ids"],
        "attention_mask" : tokenizer_out["attention_mask"]
    }
    tokenized_data.append(tokenized_d)

torch.save(tokenized_data, "tokenized_data.pt")
train_loader = DataLoader(TokenizedDataset(tokenized_data), batch_size = 16, shuffle=True)
vocab_size = tokenizer.vocab_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = TransformerEncoder(vocab_size = vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, is_autoregressive = True, dropout_rate = 0.1)
decoder = TransformerDecoder(vocab_size = vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, is_autoregressive = True, dropout_rate = 0.1)
encoder.to(device)
decoder.to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


params =list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
writer = SummaryWriter(log_dir="runs/transformer_experiment")

num_epochs = 10
global_step = 0
for i in range(num_epochs):
    encoder.train()
    decoder.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)  
        target = input_ids[:, 1:].to(device)                # All but first token, All but last token
        encoder_output, *_ = encoder(input_ids[:, 1:])
        decoder_output, *_ = decoder(input_ids[:, :-1] , encoder_output)
        loss = loss_fn(decoder_output.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1
        
    
torch.save(encoder.state_dict(), "encoder.pt")
torch.save(decoder.state_dict(), "decoder.pt")

writer.close()