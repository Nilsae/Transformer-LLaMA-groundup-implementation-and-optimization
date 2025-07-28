import torch, os
from transformer import TransformerDecoder, TransformerEncoder
from transformers import AutoTokenizer
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("checkpoints/checkpoint_epoch_10.pt", map_location=device) 
tokenizer = AutoTokenizer.from_pretrained('./.hf_models/gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
encoder = TransformerEncoder(vocab_size = vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, is_autoregressive = True, dropout_rate = 0.1)
decoder = TransformerDecoder(vocab_size = vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, is_autoregressive = True, dropout_rate = 0.1)
encoder.to(device)
decoder.to(device)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
encoder.eval()
decoder.eval()
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1) # (batch_size, vocab_size) or (batch_size, seq_len, vocab_size)
    cumulative_probs = torch.cumsum(probs, dim=-1) #if x = [1, 2, 3, 4],  torch.sum(x) = 10,  torch.cumsum(x, dim=0) =  [1, 3, 6, 10]
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone() #  shift the mask one to right -> we keep at least 1 token (first token)
    # we use clone() to avoid in-place modification errors
    sorted_mask[..., 0] = 0 # manually un-masks the first token - redundant but safety net
    sorted_logits[sorted_mask] = float('-inf')
    filtered_logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_indices, dim=-1))
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1) # samples from a discrete probability distribution (categorical), like rolling a biased die.

def generate(prompt_text, max_gen_len=15):
    with torch.no_grad():
        prompt = tokenizer(prompt_text, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        encoder_output, *_ = encoder(input_ids, inference = True)

        # decoder starts with EOS token
        decoder_input = torch.tensor([[tokenizer.eos_token_id]], device=device)  # (1, 1)

        for cur_seq in range(max_gen_len):
            decoder_output, *_ = decoder(decoder_input[:,max(0, cur_seq-15):], encoder_output, inference = True)
            next_token_logits = decoder_output[:, -1, :] 
            # next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1) greedy
            next_token = top_p_sampling(next_token_logits, p=0.9)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_ids = decoder_input[0].tolist()
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "The stars whispered a secret."
    result = generate(prompt, max_gen_len=10)
    print("\nGenerated story:\n", result)