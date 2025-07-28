import torch 
import torch.nn as nn
import math
from model.positional_encoding import RotaryPositionalEmbedding
from optimization_utils import LoRALinear

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate= 0.1, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_projection_layer = LoRALinear(embed_dim, use_lora=use_lora, r=r, alpha = alpha) # nn.Linear(embed_dim, embed_dim) 
        self.k_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.v_projection_layer = LoRALinear(embed_dim, use_lora=use_lora, r=r, alpha = alpha) #nn.Linear(embed_dim, embed_dim) 
        self.final_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
    
    # decoder-only -> always use casual masking
    def scaled_dot_product_attention(self, q, k, v, causal_mask=None, padding_mask = None, past_k = None, past_v = None, inference = False): #(batch_size, seq_len_k, d_k) or (batch_size,num_heads, seq_len_k, d_k)
        if inference and past_k is not None and past_v is not None:
            assert past_k.shape == past_v.shape
            k = torch.cat([past_k, k], dim = -2) # torch.cat([tensor1, tensor2, ...])
            v = torch.cat([past_v, v], dim = -2)
        if not inference and (past_k is not None or past_v is not None):
            raise ValueError("past_k and past_v should only be used during inference.")
        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores/math.sqrt(k.size(-1)) 
        
     
        if causal_mask is None: 
            seq_len_q = scores.size(-2) # negative because num_heads might be or not be there
            seq_len_k = scores.size(-1)
            causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len_q, seq_len_k) 1 -> all the batches
        if padding_mask is not None:
            combined_mask = causal_mask & padding_mask
            scaled_scores = scaled_scores.masked_fill(combined_mask == 0, -float('inf'))
        else:
            scaled_scores = scaled_scores.masked_fill(causal_mask == 0, -float('inf'))

                
        if not inference:
            scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True).values #Prevents softmax overflow by centering scores.  # not in the original transformer paper
        attn_weights = torch.softmax(scaled_scores, dim = -1) 
        attn_weights = self.dropout(attn_weights) if not inference else attn_weights
        return torch.matmul(attn_weights, v)
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(), # or GELU - why?
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self,input):
        return self.dropout(self.MLP(input))
    
    
class TransformerDecoderUnit(nn.Module):
    def __init__(self, embed_dim = 64, num_heads = 2, hidden_dim = 128, dropout_rate = 0.1, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.self_attention_layer = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate=dropout_rate, use_lora=use_lora, r=r, alpha = alpha)
        self.self_attention_layer_norm = nn.LayerNorm(embed_dim)
        self.FFN_layer_norm = nn.LayerNorm(embed_dim)
        self.drop_out = nn.Dropout(dropout_rate)
        self.FFN = FeedForward(embed_dim, hidden_dim, dropout_rate)
    def forward(self, decoder_in, encoder_out, past_k_self = None, past_v_self = None, past_k_cross = None, past_v_cross = None, inference = False):
        self_attn_out, past_k_self, past_v_self = self.self_attention_layer(decoder_in, past_k_self, past_v_self, inference = inference)
        norm_self_attn_out = decoder_in + self.self_attention_layer_norm(self_attn_out)
        FFN_out = self.FFN(norm_self_attn_out)
        norm_FFN_out  = norm_self_attn_out + self.FFN_layer_norm(FFN_out)
        return norm_FFN_out, past_k_self, past_v_self, past_k_cross, past_v_cross
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, dropout_rate = 0.1, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.num_layers = num_layers
        self.decoder_stack = nn.ModuleList([TransformerDecoderUnit(embed_dim, num_heads, hidden_dim, dropout_rate, use_lora, r, alpha) for i in range(num_layers)])
        # self.pos_encoding = RoPE(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.final_projection_layer = nn.Linear(embed_dim, vocab_size)
    def forward(self,decoder_input, encoder_output, inference = False):
        embedded_x = self.embedding_layer(decoder_input)
        batch_size = embedded_x.size(0)
        seq_len = embedded_x.size(1)
        # x = embedded_x + self.pos_encoding(seq_len, batch_size)
        x = self.dropout(x)
        past_k_self = [None] * self.num_layers
        past_v_self = [None] * self.num_layers
        past_k_cross = [None] * self.num_layers
        past_v_cross = [None] * self.num_layers

        for i in range(self.num_layers):
            x,  past_k_self[i], past_v_self[i], past_k_cross[i], past_v_cross[i]= \
                self.decoder_stack[i](x, encoder_output, past_k_self[i], past_v_self[i], past_k_cross[i], past_v_cross[i], inference = inference)
        return self.final_projection_layer(x), past_k_self, past_v_self, past_k_cross, past_v_cross