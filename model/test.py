import logging
import torch
from attention_with_KV_caching import MultiHeadSelfAttention

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG, WARNING, etc.

# Create handler (console output)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(handler)

dummy_inputs = torch.randn(4, 8, 2) #(batch_size, seq_len, embed_dim)
autoregressive_model = MultiHeadSelfAttention(2, 2)
attention_out, raw_attention, k, v = autoregressive_model(dummy_inputs)
logger.info(attention_out)
model = MultiHeadSelfAttention(2, 2, is_autoregressive = False)
attention_out_no_mask, _, _, _ = model(dummy_inputs)
logger.info(attention_out_no_mask)