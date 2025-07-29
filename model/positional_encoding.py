import torch, math
import torch.nn as nn
import torch
import json
import torch.optim as optim

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or DEBUG, WARNING, etc.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Unlike sinusidal PE (which encodes absolute positions), 
# RoPE ensures that the dot product between two rotated vectors encodes their relative distance
# sinusidal Adds sin/cos to embeddings	- uses all of  embed dim - modifies Token embeddings (input) - 	Absolute position 
# rotary Rotates query/key vectors - appies on Usually first half of head_dim - modifies Query and Key (before attention) - relative position - distance aware and casual-aware
class RotaryPositionalEmbedding(nn.Module): #[batch, num_heads, seq_len, head_dim]
    def __init__(self, max_seq_len, head_dim ):
        super().__init__()
        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, head_dim//2, 1, dtype=torch.float32) 
        divisor_term = 10000**(2 * i / head_dim)
        matrix = pos/divisor_term
        sin_term = torch.sin(matrix).unsqueeze(0).unsqueeze(0) 
        cos_term = torch.cos(matrix).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sin_cached", sin_term)
        self.register_buffer("cos_cached", cos_term)
        
    def forward(self, vector): 
        vec_even = vector[:, :, :, 0::2] # or [..., 0::2] even dims
        vec_odd = vector[:, :, :, 1::2]
        seq_len = vector.shape[2]
        rotated_even = vec_even * self.cos_cached[:, :, :seq_len, :] - vec_odd * self.sin_cached[:, :, :seq_len, :] # Element-wise Multiplication
        rotated_odd = vec_even * self.sin_cached[:, :, :seq_len, :] + vec_odd * self.cos_cached[:, :, :seq_len, :]
        # rotated = torch.stack((rotated_even, rotated_odd), dim=-1)  # shape: [B, H, T, D/2, 2]
        # return rotated.reshape(rotated.shape[0], rotated.shape[1], seq_len, rotated.shape[3]*rotated.shape[4])
        return torch.cat([rotated_even.unsqueeze(-1), rotated_odd.unsqueeze(-1)], dim=-1).flatten(-2)

# tensor.flatten(start_dim) works same as tensor.flatten(start_dim, end_dim)
# it flattens all the dims from start dim to the end dim 
# above, we ue .flatten(-2) bc flatten dims -1 and -2 :[B, H, T, D/2, 2] -> [B, H, T, D]

# Q apply rope - 	Defines "what to look for", so inject relative position
# K apply rope - 	Defines "where to look", so inject relative position
# V do not apply -	Carries content, rotation would distort information

# Sin PE is ADDED to the input coming from the embedding layer before in moves to the encoder/decoder stacks
 #  unsqueeze on the last dimention converts each element to a tensor: [ element,  element  ]->[ [element],  [element]    ], (a, b) -> (a, b, 1)
        # unquzeeze on the first dimention(dim0) converst the whole tensor to [whole tensor], tensor - > [tensor], (a, b) -> (1, a, b)







class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
    def forward(self, x): #batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        positions   = torch.arange(seq_len, device= x.device)
        embed_out = self.pos_embedding(positions)
        return embed_out.unsqueeze(0) + x
        
# about why we use buffer:
# the learnable parameters when we use nn.Module automatically get a state_dict and get moved to GPU usign .cude() or to_device()
# so they are tracked. but if there's a tensor that we want to save and is nt learnable like this positional encoding, we use
# register_buffer . this way it gets saved/loaded with the model and moes to the right device but has no foorptint on the gradients
# other than pos_encoding, we might wanna use it for indices or masks or running stats in BatchNorm (e.g., running_mean, running_var) - see below for batch normalization
class SinPositionalEncoding(nn.Module): # we need to inherit because of register_buffer
    def __init__(self,max_seq_len, embed_dim ):
        super().__init__()
        pos = torch.tensor([[i] for i in range(max_seq_len)]) # or torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1) 
       
        i = torch.arange(0, embed_dim//2, 1)
        pos_encoding = torch.zeros(max_seq_len, embed_dim)    
        divisor_term = 10000**(2 * i / embed_dim)
        matrix = pos/divisor_term
        pos_encoding[:, 0::2] = torch.sin(matrix) # first dim is batch sizes
        pos_encoding[:, 1::2] = torch.cos(matrix)
        self.register_buffer("pos_encoding", pos_encoding)
    def forward(self, seq_len, batch_size): # bc batch size may vary
        return self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)[:, :seq_len, :] # .unsqueeze(0) :[max_seq_len, embed_dim] -> [1, max_seq_len, embed_dim]


# conceptually for sinus pos encoding:
        # out  = torch.zeros(seq_len, embed_dim)
        # for pos in range(seq_len):
        #     for dim in range(embed_dim):
        #         if dim%2 ==0:
        #             out[pos][dim] = math.sin(pos/10000**(2*(dim//2)/embed_dim))
        #         else:
        #             out[pos][dim] = math.cos(pos/10000**(2*(dim//2)/embed_dim))
        
        
# Comments on Batch_normalization
# we use it for :
    # normalizing the input - scaling the feature to a standard
    # controlling extreme shifts in weights during training 
    # speeding convergence 
    # improvign generalization
# we only calculate the batch normalizaation durign training and in inference we use it
# here's the formula to calculate it:
# if momentum is 0.1, then 10 percent of it comes from the current batch and 90 percent is kept from all the previous batches


