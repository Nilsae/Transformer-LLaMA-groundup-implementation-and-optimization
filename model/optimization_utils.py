import torch
import torch.nn as nn

#Sometimes LoRA initializes A and B weights to small values (e.g., ~N(0, 0.01)) to avoid destabilizing inference early.
class LoRALinear(nn.Module):
    def __init__(self, embed_dim, use_lora=False, r=8, alpha = 16):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        if use_lora:
            for param in self.linear.parameters():
                param.requires_grad = False
        
            self.A  = nn.Linear(embed_dim, r, bias=False) # removing bias saves 2× d_model parameters (for A and B)
            self.B  = nn.Linear(r, embed_dim, bias=False) # also, adding more biases here(in finetuning) would introduce 
                                                    #additional parameters and possibly offset the pretrained behavior too much.
            nn.init.normal_(self.A.weight, std=1e-3)
            nn.init.zeros_(self.B.weight)
            
            self.scaling = alpha / r # for adjusting how much influence the LORA adapter has during fine-tuning 
        self.use_lora = use_lora
    def forward(self, x):
        if not self.use_lora:
            return self.linear(x)
        linear_output = self.linear(x)
        A_out = self.A(x)
        B_out = self.B(A_out)
        return linear_output + self.scaling * B_out
    
# Queries define what each token attends to.
# Changing Q means changing the perspective from which tokens seek information from others.

# Values define what is returned once attention weights are computed.
# So adapting V changes the information flow back to the token.
# Especially important when outputs need to be adjusted for new tasks or styles.

# K is shared across all queries.
# K represents the static content of each token.
# In many tasks, especially decoder-only language models, adjusting Q and V is more directly impactful than adjusting K.

# In many LoRA ablation studies:
# Q+V → 95–100% of full FT performance
# Q only → ~90–95%
# V only → ~90–95%
# K only → less effective alone