import torch.nn as nn

class EncoderDecoderTF(nn.Module):
    """ Original Transformer architecture that uses both the encoder and decoder side"""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def _encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)

    def _decode(self, tgt, mem, tgt_mask, mem_mask):
        tgt = self.tgt_embed(tgt)
        return self.decoder(tgt, mem, tgt_mask, mem_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        mem = self._encode(src, src_mask)
        decoded = self._decode(tgt, mem, tgt_mask, src_mask)
        return decoded
