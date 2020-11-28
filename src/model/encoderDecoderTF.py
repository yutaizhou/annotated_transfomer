import torch.nn as nn
class EncoderDecoderTF(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def _encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def _decode(self, memory, mem_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, mem_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self._encode(src, src_mask)
        return self._decode(encoded, src_mask, tgt, tgt_mask)

