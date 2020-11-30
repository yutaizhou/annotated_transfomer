import torch.nn as nn
from .utils import repeat_module, LayerNorm, SublayerConnection

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = repeat_module(layer, N)
        self.norm = LayerNorm(layer.model_dim)

    def forward(self, x, mem, tgt_mask, mem_mask):
        for layer in self.layers:
            x = layer(x, mem, tgt_mask,  mem_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """ causal self attention + cross attention + FC """
    def __init__(self, model_dim, self_attn, cross_attn, fc_net, dropout):
        super().__init__()
        self.model_dim = model_dim
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.fc_net = fc_net
        self.sublayers = repeat_module(SublayerConnection(model_dim, dropout), 3)
    
    def forward(self, x, mem, tgt_mask, mem_mask):
        m = mem
        self_attn_sublayer =  lambda x: self.self_attn(x,x,x,tgt_mask)
        cross_attn_sublayer = lambda x: self.cross_attn(x,m,m,mem_mask)

        x = self.sublayers[0](x, self_attn_sublayer)
        x = self.sublayers[1](x, cross_attn_sublayer)
        x = self.sublayers[2](x, self.fc_net)
        return x 