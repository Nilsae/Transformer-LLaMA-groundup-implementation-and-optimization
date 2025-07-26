import torch, math
import torch.nn as nn

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
        pos = torch.tensor([[i] for i in range(max_seq_len)])
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