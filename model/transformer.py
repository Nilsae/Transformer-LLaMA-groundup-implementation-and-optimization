
import torch 
import torch.nn as nn
import math
from positional_encoding import SinPositionalEncoding, LearnedPositionalEncoding


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)




# embed_dim : size of the vector used to represent each token (word, character, etc.)
# seq_len :  number of tokens in the input sequence which is equal to number of time steps ( one token at a time)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, is_autoregressive = True, dropout_rate= 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.k_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.v_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.final_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.embed_dim = embed_dim
        # self.layer_norm = nn.LayerNorm(self.embed_dim) there is already attention  normalization in the transformer class
        self.is_autoregressive = is_autoregressive
        self.dropout = nn.Dropout(dropout_rate)
    
    def scaled_dot_product_attention(self, q, k, v, causal_mask=None, padding_mask = None, past_k = None, past_v = None, is_autoregressive= True, inference = False): #(batch_size, seq_len_k, d_k) or (batch_size,num_heads, seq_len_k, d_k)
        if past_k is not None and past_v is not None:
            assert past_k.shape == past_v.shape
            k = torch.cat([past_k, k], dim = -2) # torch.cat([tensor1, tensor2, ...])
            v = torch.cat([past_v, v], dim = -2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores/math.sqrt(k.size(-1)) 
        
        if is_autoregressive:
            if causal_mask is None: 
                seq_len_q = scores.size(-2) # negative because num_heads might be or not be there
                seq_len_k = scores.size(-1)
                causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device)).bool()
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len_q, seq_len_k) 1 -> all the batches
        if is_autoregressive:
            if padding_mask is not None:
                combined_mask = padding_mask & causal_mask
                scaled_scores = scaled_scores.masked_fill(combined_mask == 0, -float('inf'))
            else:
                scaled_scores = scaled_scores.masked_fill(causal_mask == 0, -float('inf'))
        if not inference:
            scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True).values #Prevents softmax overflow by centering scores.  # not in the original transformer paper
        attn_weights = torch.softmax(scaled_scores, dim = -1) 
        attn_weights_dropout = self.dropout(attn_weights)
        return torch.matmul(attn_weights_dropout, v)
    
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
        attn_out = self.scaled_dot_product_attention(q, k, v, causal_mask = None, padding_mask=padding_mask, past_k = past_k, past_v = past_v, is_autoregressive = self.is_autoregressive, inference = inference)
        # for head in self.num_heads:  prevents GPU from parallelizing operations efficiently and introduces slow Python-level control flow.
        #     attn_out = scaled_dot_product_attention(q[:, :, head, :], k[:, :, head, :], v[:, :, head, :])
        attn_out = attn_out.transpose(1, 2).reshape([batch_size, seq_len, embed_dim])
        assert attn_out.shape == input.shape
        attn_out = self.final_projection_layer(attn_out)
        return attn_out, k, v #This is post-norm, used in older models. Most modern architectures like GPT-2/3 use pre-norm: x_norm = self.layer_norm(input)
    
    #detatching: not compute gradients - we can do it in inference where we are not training and we want to save memory, or when we dont want the information of the gradient of a specific variable in our trainign because it would be cheating from the labels

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate= 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.k_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.v_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.final_projection_layer = nn.Linear(embed_dim, embed_dim) 
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
    def cross_attention(self,q, k, v, past_k = None, past_v = None, padding_mask = None): 
        if past_k is not None and past_v is not None:
            assert past_k.shape == past_v.shape
            k = torch.cat([past_k, k], dim = -2) # torch.cat([tensor1, tensor2, ...])
            v = torch.cat([past_v, v], dim = -2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores/math.sqrt(k.size(-1)) 
        if padding_mask is not None:
                scaled_scores = scaled_scores.masked_fill(padding_mask == 0, -float('inf'))
        scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True).values 
        attn_weights = torch.softmax(scaled_scores, dim = -1) 
        attn_weights_dropout = self.dropout(attn_weights)
        return torch.matmul(attn_weights_dropout, v)
    
    def forward(self, input, encoder_out, past_k=None, past_v=None):
        q = self.q_projection_layer(input)
        k = self.k_projection_layer(encoder_out)
        v = self.v_projection_layer(encoder_out)
        [batch_size, seq_len, embed_dim] = q.size() # or batch_size, seq_len, embed_dim = q.size() - they both work
        assert embed_dim % self.num_heads == 0
        head_dim = int(embed_dim//self.num_heads)
        src_len = k.size(1)
        q = q.reshape([batch_size, seq_len, self.num_heads, head_dim]).transpose(1, 2)
        k = k.reshape([batch_size, src_len, self.num_heads, head_dim]).transpose(1, 2)
        v = v.reshape([batch_size, src_len, self.num_heads, head_dim]).transpose(1, 2)
        attn_out = self.cross_attention(q, k, v, past_k = past_k, past_v = past_v)
        attn_out = attn_out.transpose(1, 2).reshape([batch_size, seq_len, embed_dim])
        assert attn_out.shape == input.shape
        attn_out = self.final_projection_layer(attn_out)
        return attn_out, k, v #This is post-norm, used in older models. Most modern architectures like GPT-2/3 use pre-norm: x_norm = self.layer_norm(input)
    
# This FFN is position-wse, which means
# the same FFN is applied to each token (aka position) independently
# not mixing the tokens unlike attention
# each token's embedding vec is processed separately but with the shared weights
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
    
    
    
class TransformerEncoderUnit(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, is_autoregressive = True, dropout_rate = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, is_autoregressive)
        self.FFN = FeedForward(embed_dim, hidden_dim, dropout_rate)
        self.attention_layer_norm = nn.LayerNorm(embed_dim)
        self.FFN_layer_norm = nn.LayerNorm(embed_dim)
        self.drop_out = nn.Dropout(dropout_rate)
    def forward(self, input, past_k=None, past_v=None, inference = False):
        attn_out, past_k, past_v = self.attention(input, past_k, past_v ,inference = inference)
        attn_out_added_input = attn_out + input # residula 1
        normalized_attn_out = self.attention_layer_norm(attn_out_added_input)
        FFN_out = self.FFN(normalized_attn_out)
        FFN_out_added_input = FFN_out + normalized_attn_out # residual 2 # before we did norm in the attn class - so now the residul is to the final attn output which is the normalized one
        normalized_FFN = self.FFN_layer_norm(FFN_out_added_input)
        transformer_out = self.drop_out(normalized_FFN)
        return transformer_out, past_k, past_v
 
## What is GELU:
    # GELU is smoother than ReLU and captures more subtle non-linearities.
    # ReLU is simpler and used in the original paper.
    # GELU is now used in models like BERT and GPT for better training dynamics
    
    
## Why do we have to apply RELU or GELU between the layers in the FFN:




# Pre-norm makes training deep Transformers more stable.
# Post-norm was used in the original Transformer but can become unstable when stacking many layers (why newer models moved to pre-norm).


# attention is all you need: 
# We apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
# We also apply dropout to the output of each sub-layer (before it is added to the sub-layer input and normalized).
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, is_autoregressive = True, dropout_rate = 0.1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.num_layers = num_layers
        self.encoder_stack = nn.ModuleList([TransformerEncoderUnit(embed_dim, num_heads, hidden_dim, is_autoregressive, dropout_rate) for i in range(num_layers)])
        self.pos_encoding = SinPositionalEncoding(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.seq_len= seq_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
    def forward(self,x, inference = False):
        x = self.embedding_layer(x)
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x + self.pos_encoding(seq_len, batch_size)
        x = self.dropout(x)
        past_k = [None] * self.num_layers
        past_v = [None] * self.num_layers
        for i in range(self.num_layers):
            x, past_k[i], past_v[i]  = self.encoder_stack[i](x, past_k[i], past_v[i], inference = inference)
        return x, past_k, past_v
    
class TransformerDecoderUnit(nn.Module):
    def __init__(self, embed_dim = 64, num_heads = 2, hidden_dim = 128, dropout_rate = 0.1):
        super().__init__()
        self.self_attention_layer = MultiHeadSelfAttention(embed_dim, num_heads, is_autoregressive = True)
        self.self_attention_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attention_layer_norm = nn.LayerNorm(embed_dim)
        self.FFN_layer_norm = nn.LayerNorm(embed_dim)
        self.drop_out = nn.Dropout(dropout_rate)
        self.cross_attention_layer = CrossAttention(embed_dim, num_heads, dropout_rate= 0.1)
        self.FFN = FeedForward(embed_dim, hidden_dim, dropout_rate)
    def forward(self, decoder_in, encoder_out, past_k_self = None, past_v_self = None, past_k_cross = None, past_v_cross = None, inference = False):
        self_attn_out, past_k_self, past_v_self = self.self_attention_layer(decoder_in, past_k_self, past_v_self, inference = inference)
        norm_self_attn_out = decoder_in + self.self_attention_layer_norm(self_attn_out)
        cross_attn_out, past_k_cross, past_v_cross = self.cross_attention_layer(norm_self_attn_out, encoder_out, past_k_cross, past_v_cross)
        norm_cross_attn_out = norm_self_attn_out + self.cross_attention_layer_norm(cross_attn_out)
        FFN_out = self.FFN(norm_cross_attn_out)
        norm_FFN_out  = norm_cross_attn_out + self.FFN_layer_norm(FFN_out)
        return norm_FFN_out, past_k_self, past_v_self, past_k_cross, past_v_cross
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, batch_size = 16, num_layers = 6, seq_len = 16, embed_dim = 64, num_heads = 2, hidden_dim = 128, is_autoregressive = True, dropout_rate = 0.1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.num_layers = num_layers
        self.decoder_stack = nn.ModuleList([TransformerDecoderUnit(embed_dim, num_heads, hidden_dim, dropout_rate) for i in range(num_layers)])
        self.pos_encoding = SinPositionalEncoding(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.final_projection_layer = nn.Linear(embed_dim, vocab_size)
    def forward(self,decoder_input, encoder_output, inference = False):
        embedded_x = self.embedding_layer(decoder_input)
        batch_size = embedded_x.size(0)
        seq_len = embedded_x.size(1)
        x = embedded_x + self.pos_encoding(seq_len, batch_size)
        x = self.dropout(x)
        past_k_self = [None] * self.num_layers
        past_v_self = [None] * self.num_layers
        past_k_cross = [None] * self.num_layers
        past_v_cross = [None] * self.num_layers

        for i in range(self.num_layers):
            x,  past_k_self[i], past_v_self[i], past_k_cross[i], past_v_cross[i]= \
                self.decoder_stack[i](x, encoder_output, past_k_self[i], past_v_self[i], past_k_cross[i], past_v_cross[i], inference = inference)
        return self.final_projection_layer(x), past_k_self, past_v_self, past_k_cross, past_v_cross