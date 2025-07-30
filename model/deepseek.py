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


# Multi Query Attention (MQA):

# Each head still has its own Query projection (Q_h = W_q^h * X).
# But all heads share the same K and V projections:
# Different heads still attend to different things, but from the same context representation - the same K/V vectors

# The diversity now comes only from the Q-side.
# Why Doesn’t This Completely Kill Performance?
# Because:

# Q Projections Still Differ → Each head asks a different question.
# Same K/V ≠ Same Attention → Even with shared context, different Qs can produce very different attention maps.


# Multiple heads still help by:
# Producing multiple different Q projections = different attention scores.
# Having different softmax distributions → different attention outputs even over the same K/V.

# Think of it like this:
# You’re asking 8 experts (heads) the same set of facts (K/V), but each expert asks their own question (Q). You'll get 8 different perspectives on the same knowledge base.

class MultiQueryAttention(nn.Module):             
    def __init__(self, embed_dim, num_heads, seq_len, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim//num_heads 
        self.head_dim = head_dim
        # self.q_projection_layer = LoRALinear(embed_dim, use_lora=use_lora, r=r, alpha = alpha, bias=False) # nn.Linear(embed_dim, embed_dim) 
        self.q_projection_layer = nn.Linear(embed_dim, embed_dim, bias=False)  
        self.k_projection_layer = nn.Linear(embed_dim, head_dim, bias=False) 
        self.v_projection_layer = nn.Linear(embed_dim, head_dim, bias=False) 
        # self.v_projection_layer = LoRALinear(embed_dim, use_lora=use_lora, r=r, alpha = alpha, bias=False) #nn.Linear(embed_dim, embed_dim) 
        self.RoPE  = RotaryPositionalEmbedding(seq_len, head_dim) # rope calcs the PEs per HEAD unlike SinPE
        self.final_projection_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.embed_dim = embed_dim
    
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
            padding_mask = padding_mask[:, None, None, :]
            combined_mask = causal_mask & padding_mask
            scaled_scores = scaled_scores.masked_fill(combined_mask == 0, -float('inf'))
        else:
            scaled_scores = scaled_scores.masked_fill(causal_mask == 0, -float('inf'))

                
        if not inference:
            scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True).values #Prevents softmax overflow by centering scores.  # not in the original transformer paper
        attn_weights = torch.softmax(scaled_scores, dim = -1) 
        return torch.matmul(attn_weights, v)
    
    def forward(self, input, padding_mask, past_k=None, past_v=None, inference  = False):
        q = self.q_projection_layer(input)
        k = self.k_projection_layer(input)
        v = self.v_projection_layer(input)
        [batch_size, seq_len, embed_dim] = q.size() # or batch_size, seq_len, embed_dim = q.size() - they both work
        assert embed_dim % self.num_heads == 0
        head_dim = int(embed_dim//self.num_heads)
        q = q.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        # k = k.reshape([batch_size, seq_len, 1, head_dim]).transpose(1, 2)
        # v = v.reshape([batch_size, seq_len, 1, head_dim]).transpose(1, 2)
        # how deepseek does it: (same outcome)
        k = k.view(batch_size, seq_len, self.head_dim)
        v = v.view(batch_size, seq_len, self.head_dim)
        k = k.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        v = v.unsqueeze(1)  # (batch, 1, seq_len, head_dim)

        attn_out = self.scaled_dot_product_attention(q, k, v, causal_mask = None, padding_mask=padding_mask, past_k = past_k, past_v = past_v, inference = inference)
        attn_out = attn_out.transpose(1, 2).reshape([batch_size, seq_len, embed_dim])
        assert attn_out.shape == input.shape
        attn_out = self.final_projection_layer(attn_out)
        return attn_out, k, v
    
# L = num_layers    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim = 64, num_heads = 2, seq_len = 16, use_lora=False, r=8, alpha = 16, deepnorm_alpha =4):
        super().__init__()
        self.one_over_deepnorm_alpha = 1 / deepnorm_alpha
        self.self_attention_layer = MultiQueryAttention(embed_dim, num_heads, seq_len, use_lora=use_lora, r=r, alpha = alpha)
        self.rms_norm1 = nn.RMSNorm(embed_dim) # RMS norm vs Layernorm: RMSNorm only scales, doesn't shift
        self.rms_norm2 = nn.RMSNorm(embed_dim)
        self.rms_norm3 = nn.RMSNorm(embed_dim)
        self.FFN = FeedForward(embed_dim)
    def forward(self, decoder_in, padding_mask, past_k_self = None, past_v_self = None, inference = False):
        norm_decoder_in = self.rms_norm1(decoder_in)
        self_attn_out, past_k_self, past_v_self = self.self_attention_layer(norm_decoder_in, padding_mask, past_k_self, past_v_self, inference = inference)
        attn_out_residual = self.one_over_deepnorm_alpha * self_attn_out + decoder_in
        norm_attn_out = self.rms_norm2(attn_out_residual)
        FFN_out = self.FFN(norm_attn_out)
        FFN_out_residual = self.one_over_deepnorm_alpha * FFN_out + attn_out_residual
        return self.rms_norm3(FFN_out_residual), past_k_self, past_v_self      # final RMS norm added in deepseek - non existent in llama
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.num_layers = num_layers
        deepnorm_alpha = math.sqrt(4 * num_layers)
        self.decoder_stack = nn.ModuleList([TransformerBlock(embed_dim, num_heads, seq_len, use_lora, r, alpha, deepnorm_alpha) for i in range(num_layers)])
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_projection.weight = self.embedding_layer.weight #why do we tie weights? 
 
        # why do we tie weights?
        #  Saves memory 
        #  Slightly improves performance and generalization
        #
        # output_projection(lm_head) must be bias-free linear layer: nn.Linear(dim, vocab_size, bias=False)
        # because embeddings don't have bias
        #  embedding and output_projection weight shapes must match: [vocab_size, dim]


        
    def forward(self,x, padding_mask, inference = False):
        x = self.embedding_layer(x)
        batch_size = x.size(0)
        seq_len = x.size(1)
        past_k_self = [None] * self.num_layers
        past_v_self = [None] * self.num_layers

        for i in range(self.num_layers):
            x,  past_k_self[i], past_v_self[i]= \
                self.decoder_stack[i](x, padding_mask, past_k_self[i], past_v_self[i], inference = inference)
        return self.output_projection(x), past_k_self, past_v_self
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = 4 * embed_dim # based on LLaMA-v1 paper
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim*2),
            SwiGLU() , 
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self,input):
        return self.MLP(input)
    
class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # split last dim into halves
        return torch.nn.functional.silu(x1) * x2  # SwiGLU: SiLU(x1) * x2         
    # silu = swish = x* sigmoid(x)