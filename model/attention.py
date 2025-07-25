import torch 
import math
# Note: tensor.transpose(dim0, dim1) swaps the two specified dimensions dim0 and dim1 of the tensor

# Note: In attention mechanisms, we often mask out certain tokens — for example:
    #  (1) To prevent attending to padding tokens
    #  (2) To prevent looking into the future in causal/self-attention
    # Then when we do softmax on top of that, softmax(-inf) → 0 so we dont attend to them
def scaled_dot_product_attention(q, k, v, mask=None): 
    # k looks like (batch_size, seq_len_k, d_k)
    # q looks like (batch_size, seq_len_q, d_k)
    # if we do k.transpose(-2, -1), we get (batch_size, d_k, seq_len_k)
    # so after the multiplication, we will have (batch_size, seq_len_q, seq_len_k)
    # for example, if q, k both look like torch.randn(2, 5, 64): batch of 2, 5 queries each, 64-dim,; then scores will look like torch.Size([2, 5, 5])
    # so for each of the 5 queries in each batch, we get 5 scores (one per query or key)
    scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = scores/math.sqrt(k.size(-1)) # tensor.size(dim0) gets the size of dimention dim0 of the tensor| k.size(-1) returns an int but torch.sqrt expects a tensor, so if used we should convert:  torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
    # math.sqrt expects int so no problem there
    # scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True).values 
    if mask is None:
        mask = torch.tril(torch.ones([q.size(0), q.size(1), q.size(1)]), diagonal=0) # the tokens do attend to themselves bc diagonal is also 1 (by setting diagonal = 0!)- the syntax looks confusing to me 
    scaled_scores = scaled_scores.masked_fill(mask == 0, -float('inf'))
    attn_weights = torch.softmax(scaled_scores, dim = -1) # we apply softmax to the key dimension so that For each query, all key weights sum to 1

    return torch.matmul(attn_weights, v)

# Query (Q)
# A query vector represents the current token's request for information.

# It asks: “What am I looking for in other tokens?”

# Key (K)
# A key vector is associated with each token in the sequence and defines what content that token contains.

# It answers: “What information do I have?”

# Value (V)
# A value vector contains the actual information or content to be passed along or combined.

# It is used to compute the final output after weighting the attention scores.