
import torch 
import torch.nn as nn
import math
from positional_encoding import SinPositionalEncoding, LearnedPositionalEncoding


def scaled_dot_product_attention(q, k, v, mask=None, past_k = None, past_v = None, is_autoregressive= True): #(batch_size, seq_len_k, d_k) or (batch_size,num_heads, seq_len_k, d_k)
    if past_k is not None and past_v is not None:
        assert past_k.shape == past_v.shape
        k = torch.cat([past_k, k], dim = -2) # torch.cat([tensor1, tensor2, ...])
        v = torch.cat([past_v, v], dim = -2)
    scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = scores/math.sqrt(k.size(-1)) 
    if is_autoregressive:
        if mask is None: 
            seq_len_q = scores.size(-2) # negative because num_heads might be or not be there
            seq_len_k = scores.size(-1)
            mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device)).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len_q, seq_len_k) 1 -> all the batches
        scaled_scores = scaled_scores.masked_fill(mask == 0, -float('inf'))
    scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True).values #Prevents softmax overflow by centering scores.
    attn_weights = torch.softmax(scaled_scores, dim = -1) 
    return torch.matmul(attn_weights, v)


# embed_dim : size of the vector used to represent each token (word, character, etc.)
# seq_len :  number of tokens in the input sequence which is equal to number of time steps ( one token at a time)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, is_autoregressive = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.k_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.v_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.final_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.is_autoregressive = is_autoregressive
        
    def forward(self, input, past_k=None, past_v=None):
        q = self.q_projection_layer(input)
        k = self.k_projection_layer(input)
        v = self.v_projection_layer(input)
        [batch_size, seq_len, embed_dim] = q.size() # or batch_size, seq_len, embed_dim = q.size() - they both work
        assert embed_dim % self.num_heads == 0
        head_dim = int(embed_dim//self.num_heads)
        q = q.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        k = k.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        v = v.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        
        attn_out = scaled_dot_product_attention(q, k, v, mask = None, past_k = past_k, past_v = past_v, is_autoregressive = self.is_autoregressive)
        # for head in self.num_heads:  prevents GPU from parallelizing operations efficiently and introduces slow Python-level control flow.
        #     attn_out = scaled_dot_product_attention(q[:, :, head, :], k[:, :, head, :], v[:, :, head, :])
        attn_out = attn_out.transpose(1, 2).reshape([batch_size, seq_len, embed_dim])
        assert attn_out.shape == input.shape
        attn_out = self.final_projection_layer(attn_out)
        attn_out_and_input = attn_out + input 
        return self.layer_norm(attn_out_and_input) ,attn_out, k, v #This is post-norm, used in older models. Most modern architectures like GPT-2/3 use pre-norm: x_norm = self.layer_norm(input)
    
    #detatching: not compute gradients - we can do it in inference where we are not training and we want to save memory, or when we dont want the information of the gradient of a specific variable in our trainign because it would be cheating from the labels
    
    
# This FFN is position-wse, which means
# the same FFN is applied to each token (aka position) independently
# not mixing the tokens unlike attention
# each token's embedding vec is processed separately but with the shared weights
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Gelu(), # or RELU - why?
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self,input):
        return self.MLP(input)
    
    
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, is_autoregressive = True, dropout_rate = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, is_autoregressive)
        self.FFN = FeedForward(embed_dim, hidden_dim)
        self.attention_layer_norm = nn.LayerNorm(embed_dim)
        self.FFN_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, input):
        normalized_x = self.attention_layer_norm(input)
        attn_out, _, _, _ = self.attention(normalized_x)
        FFN_out = self.FFN(attn_out)
        FFN_out_added_input = FFN_out + attn_out # attn_out already has a + input in it - so our residula conection here is with the attn_out
        normalized_FFN = self.FFN_layer_norm(FFN_out_added_input)
        transformer_out = self.dropout(normalized_FFN)
        return transformer_out

## What is GELU:

## Why do we have to apply RELU or GELU between the layers in the FFN: