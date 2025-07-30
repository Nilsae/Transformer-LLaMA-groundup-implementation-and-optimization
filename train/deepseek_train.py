import torch
import json, os, sys, math
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.deepseek import TransformerDecoder
from transformers import AutoTokenizer
from torch.utils.data import  Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
checkpoint_dir = "checkpoints/DeepSeek-v1"
log_writer_dir = "runs/DeepSeek-v1/training"
os.makedirs(checkpoint_dir, exist_ok=True)




def lr_schedule(step, total_steps):  # Prevents divergence in early steps; allows optimizer to “ramp up.” 
    warmup_steps = int(0.05 * total_steps) 
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


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
        
    
    
tokenizer = AutoTokenizer.from_pretrained('./.hf_models/gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
    
# with open('data/generated_dataset_gpt2-medium.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f]
# tokenized_data = []
# for d in data:
#     tokenizer_out = tokenizer(d["story"], return_tensors="pt", padding="max_length", max_length=16, truncation=True, return_attention_mask=True)
#     tokenized_d = {
#         "input_ids" : tokenizer_out["input_ids"],
#         "attention_mask" : tokenizer_out["attention_mask"]
#     }
#     tokenized_data.append(tokenized_d)

# torch.save(tokenized_data, "tokenized_data.pt")
tokenized_data = torch.load("tokenized_data.pt")
train_loader = DataLoader(TokenizedDataset(tokenized_data), batch_size = 16, shuffle=True)
vocab_size = tokenizer.vocab_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = TransformerDecoder(vocab_size = vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2)
decoder.to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


params =list(decoder.parameters())
# optimizer = optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
base_lr = 1e-4
optimizer = torch.optim.AdamW(params, lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1)
# AdamW decouples weight decay from gradient updates → better generalization.
writer = SummaryWriter(log_dir=log_writer_dir)
scaler = torch.amp.GradScaler(str(device))

num_epochs = 11

num_samples = len(tokenized_data)
batch_size = 16
total_steps  = math.ceil(num_samples / batch_size) * num_epochs
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_schedule(step, total_steps))

global_step = 0
resume_from_checkpoint = True  
start_epoch = 0
if resume_from_checkpoint:
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if checkpoint_files:
        latest = sorted(checkpoint_files)[-1]
        print(f"Loading checkpoint: {latest}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest), map_location=device)
        decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")

for i in range(start_epoch, num_epochs):
    decoder.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)  
        target = input_ids[:, 1:].to(device)                # All but first token, All but last token
        with torch.amp.autocast(str(device)):
            decoder_output, *_ = decoder(input_ids[:, :-1], padding_mask=attention_mask[:, :-1])
            loss = loss_fn(decoder_output.reshape(-1, vocab_size), target.reshape(-1))
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() 
        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1
        
    checkpoint = {
    "epoch": i + 1,
    "decoder_state_dict": decoder.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item()
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{i+1}.pt"))
        

torch.save(decoder.state_dict(), "decoder.pt")

writer.close()