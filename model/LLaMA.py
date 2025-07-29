import torch 
import torch.nn as nn
import math
from model.positional_encoding import RotaryPositionalEmbedding
from model.optimization_utils import LoRALinear

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim//num_heads # "//" :integer devision but "/"": float devision
        self.head_dim = head_dim
        # in LLaMA bias = False in Q/K/V/Out projections. Because 
        # in the context of a LayerNorm → Linear → Activation → Linear → Residual block, 
        # biases can be omitted without hurting capacity.
        self.q_projection_layer = LoRALinear(embed_dim, use_lora=use_lora, r=r, alpha = alpha, bias=False) # nn.Linear(embed_dim, embed_dim) 
        self.k_projection_layer = nn.Linear(embed_dim, embed_dim, bias=False) 
        self.v_projection_layer = LoRALinear(embed_dim, use_lora=use_lora, r=r, alpha = alpha, bias=False) #nn.Linear(embed_dim, embed_dim) 
        self.RoPE  = RotaryPositionalEmbedding(seq_len, head_dim) # rope calcs the PEs per HEAD unlike SinPE
        self.final_projection_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.embed_dim = embed_dim
    
    # decoder-only -> always use casual masking
    def scaled_dot_product_attention(self, q, k, v, causal_mask=None, padding_mask = None, past_k = None, past_v = None, inference = False): #(batch_size, seq_len_k, d_k) or (batch_size,num_heads, seq_len_k, d_k)
        if inference and past_k is not None and past_v is not None:
            assert past_k.shape == past_v.shape
            k = torch.cat([past_k, k], dim = -2) # torch.cat([tensor1, tensor2, ...])
            v = torch.cat([past_v, v], dim = -2)
        if not inference and (past_k is not None or past_v is not None):
            raise ValueError("past_k and past_v should only be used during inference.")
        q_rotated = self.RoPE(q)
        k_rotated = self.RoPE(k)
        scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))
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
        return torch.matmul(attn_weights, v)
    
    def forward(self, input, past_k=None, past_v=None, padding_mask = None, inference  = False):
        q = self.q_projection_layer(input)
        k = self.k_projection_layer(input)
        v = self.v_projection_layer(input)
        [batch_size, seq_len, embed_dim] = q.size() # or batch_size, seq_len, embed_dim = q.size() - they both work
        assert embed_dim % self.num_heads == 0
        head_dim = int(embed_dim//self.num_heads)
        q = q.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        k = k.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        v = v.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        attn_out = self.scaled_dot_product_attention(q, k, v, causal_mask = None, padding_mask=padding_mask, past_k = past_k, past_v = past_v, inference = inference)
        # for head in self.num_heads:  prevents GPU from parallelizing operations efficiently and introduces slow Python-level control flow.
        #     attn_out = scaled_dot_product_attention(q[:, :, head, :], k[:, :, head, :], v[:, :, head, :])
        attn_out = attn_out.transpose(1, 2).reshape([batch_size, seq_len, embed_dim])
        assert attn_out.shape == input.shape
        attn_out = self.final_projection_layer(attn_out)
        return attn_out, k, v
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = 4 * embed_dim # based on LLaMA-v1 paper
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU() , # vs GELU or ReLU #TODOs
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self,input):
        return self.MLP(input)
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim = 64, num_heads = 2, seq_len = 16, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.self_attention_layer = MultiHeadSelfAttention(embed_dim, num_heads, seq_len, use_lora=use_lora, r=r, alpha = alpha)
        self.rms_norm1 = nn.RMSNorm(embed_dim) # RMS norm vs Layernorm: RMSNorm only scales, doesn't shift #TODO
        self.rms_norm2 = nn.RMSNorm(embed_dim)
        #  LLaMA-v1 uses no dropouts at all
        self.FFN = FeedForward(embed_dim)
    def forward(self, decoder_in, past_k_self = None, past_v_self = None, inference = False):
        norm_decoder_in = self.rms_norm1(decoder_in)
        self_attn_out, past_k_self, past_v_self = self.self_attention_layer(norm_decoder_in, past_k_self, past_v_self, inference = inference)
        attn_out_residual = self_attn_out + decoder_in
        norm_attn_out = self.rms_norm2(attn_out_residual)
        FFN_out = self.FFN(norm_attn_out)
        FFN_out_residual = FFN_out + attn_out_residual
        return FFN_out_residual, past_k_self, past_v_self
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.num_layers = num_layers
        self.decoder_stack = nn.ModuleList([TransformerBlock(embed_dim, num_heads, seq_len, use_lora, r, alpha) for i in range(num_layers)])
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.output_projection.weight = self.embedding_layer.weight #why does LLaMA tie weights? #TODO
    def forward(self,x, inference = False):
        x = self.embedding_layer(x)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x = embedded_x + self.pos_encoding(seq_len, batch_size) #sinusidal for transformer would be applied here
        past_k_self = [None] * self.num_layers
        past_v_self = [None] * self.num_layers

        for i in range(self.num_layers):
            x,  past_k_self[i], past_v_self[i]= \
                self.decoder_stack[i](x, past_k_self[i], past_v_self[i], inference = inference)
        return self.output_projection(x), past_k_self, past_v_self
    
# llama uses RMS norm instead of layernorm and does prenorm instead of postnorm 
# in case torch.nn does not have it
# class RMSNorm(nn.Module):
#     def __init__(self, d, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(d))
#         self.eps = eps

#     def forward(self, x):
#         norm = x.norm(2, dim=-1, keepdim=True)
#         return x / (norm + self.eps) * self.weight