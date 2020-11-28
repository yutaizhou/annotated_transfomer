import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_module(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def causal_mask(model_dim):
    """ have an entry attend to previous positions and itself ONLY, no subsequent enetries"""
    attn_shape = (1, model_dim, model_dim)
    mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8), diagonal=1) # 1 for subsequent positions
    return mask == 0 # True for attending positions


class Generator(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.out_fc = nn.Linear(model_dim, vocab_size)
    
    def forward(self, x):
        logits = self.out_fc(x)
        return F.softmax(logits, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-6):
        self.a_2 = nn.Parameter(torch.ones(model_dim))
        self.b_2 = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """ residual connection + layer norm """
    def __init__(self, model_dim, dropout_p):
        super().__init__()
        self.norm = LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))