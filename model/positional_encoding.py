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
        
class SinPositionalEncoding():
    
    def sinusoidal_encoding(batch_size, seq_len, embed_dim):
        
        # conceptually:
        # out  = torch.zeros(seq_len, embed_dim)
        # for pos in range(seq_len):
        #     for dim in range(embed_dim):
        #         if dim%2 ==0:
        #             out[pos][dim] = math.sin(pos/10000**(2*(dim//2)/embed_dim))
        #         else:
        #             out[pos][dim] = math.cos(pos/10000**(2*(dim//2)/embed_dim))
        
        # properly;
        
        pos = torch.tensor([[i] for i in range(seq_len)])
        i = torch.arange(0, embed_dim//2, 1)
        pos_encoding = torch.zeros(seq_len, embed_dim)    
        divisor_term = 10000**(2 * i / embed_dim)
        matrix = pos/divisor_term
        pos_encoding[:, 0::2] = torch.sin(matrix) # first dim is batch sizes
        pos_encoding[:, 1::2] = torch.cos(matrix)
        out = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        return out

