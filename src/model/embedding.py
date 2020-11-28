import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, model_dim)
        self.model_dim = model_dim
    
    def forward(self, x):
        return self.embeddings(x) * torch.sqrt(self.model_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout:float, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim)) #???

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0) # batch dim
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.shape[1]
        x = x + self.pe[:,:seq_len,:].detach()

        return self.dropout(x)
