import logging
import torch, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.attention_with_KV_caching import MultiHeadSelfAttention
from model.positional_encoding import SinPositionalEncoding
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# multihead attention test
# dummy_inputs = torch.randn(4, 8, 2) #(batch_size, seq_len, embed_dim)
# autoregressive_model = MultiHeadSelfAttention(2, 2)
# attention_out, raw_attention, k, v = autoregressive_model(dummy_inputs)
# logger.info(attention_out)
# model = MultiHeadSelfAttention(2, 2, is_autoregressive = False)
# attention_out_no_mask, _, _, _ = model(dummy_inputs)
# logger.info(attention_out_no_mask)


# positional encoding test
dummy_inputs = torch.zeros(2, 128, 6) #(batch_size, seq_len, embed_dim)
seq_len, embed_dim = 128, 6
batch_size = 2
pos_enc = SinPositionalEncoding.sinusoidal_encoding(batch_size, seq_len, embed_dim)
plt.figure(figsize=(10, 6))
for dim in range(embed_dim):
    plt.plot(pos_enc[0, :, dim], label=f"dim {dim}")

plt.title("Sinusoidal Positional Encodings")
plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=300)
