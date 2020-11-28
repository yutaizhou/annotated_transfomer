import torch.nn as nn
import torch.nn.functional as F
from .utils import repeat_module, LayerNorm, SublayerConnection

class Encoder(nn.Module):
    """ stack of N encoder layers """
    def __init__(self, layer, N):
        super().__init__()
        self.layers = repeat_module(layer, N)
        self.norm = LayerNorm(layer.model_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    """ self (bidirectional or causal) attention + FC """
    def __init__(self, model_dim, self_attn, fc_net, dropout_p):
        super().__init__()
        self.self_attn = self_attn
        self.fc_net = fc_net
        self.sublayers = repeat_module(SublayerConnection(model_dim, dropout_p), 2)
        self.model_dim = model_dim
    
    def forward(self, x, mask):
        self_attn_sublayer = lambda x: self.self_attn(x,x,x,mask)

        x = self.sublayers[0](x, self_attn_sublayer)
        x = self.sublayers[1](x, self.fc_net)
        return x