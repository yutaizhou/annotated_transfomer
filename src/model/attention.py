import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import repeat_module

def attention(q, k, v, mask=None, dropout:nn.Module=None):
    """ scaled dot product attention """
    model_dim = q.shape[-1]

    attn = (q @ k.transpose(-2,-1)) / math.sqrt(model_dim)
    attn = attn if (mask is None) else attn.masked_fill(mask==0, -float('inf'))
    attn = F.softmax(attn, dim=-1)
    attn = attn if (dropout is None) else dropout(attn)

    return attn @ v, attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head, model_dim, dropout:float = 0.1):
        super().__init__()
        assert model_dim % num_head == 0 
        self.head_size = model_dim // num_head
        self.num_head = num_head
        self.linears = repeat_module(nn.Linear(model_dim, model_dim), 4) # to_q, to_k, to_v, proj
        self.attn_weights = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        
        B = query.shape[0]
        if mask is not None: 
            mask = mask.unsqueeze(1) # same mask for all heads

        # linear projections of src or tgt or mem into q,k,v
        sz = (B, -1, self.num_head, self.head_size)
        q, k, v = [linear(x).view(*sz).transpose(1,2) for linear, x in zip(self.linears, (query, key, value))]

        # apply attention to q,k,v in batch
        x, self.attn_weights = attention(q, k, v, mask, self.dropout)

        # concat all heads together, final projection
        x = x.transpose(1,2).contiguous().view(B, -1, self.num_head * self.head_size)
        x = self.linears[-1](x)

        return x




