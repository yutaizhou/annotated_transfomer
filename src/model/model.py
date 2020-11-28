import copy
import torch.nn as nn

from model import *
from .embedding import PositionalEncoding, Embedding
from model.utils import PositionWiseFC
from .utils import PositionWiseFC, Generator
from .attention import MultiHeadedAttention
from .encoderDecoderTF import EncoderDecoderTF
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer


def make_EncoderDecoderTF_model(src_vocab_size, tgt_vocab_size,
                                num_layers=6, num_heads=8, model_dim=512, fc_dim=2048, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, model_dim, dropout)
    fc = PositionWiseFC(model_dim, fc_dim, dropout)
    position_encoder = PositionalEncoding(model_dim, dropout)

    encoder = Encoder(
            EncoderLayer(model_dim, c(attn), c(fc), dropout),
            num_layers
    )
    decoder = Decoder(
            DecoderLayer(model_dim, c(attn), c(attn), c(fc), dropout),
            num_layers
    )
    src_embedder = nn.Sequential(
        Embedding(model_dim, src_vocab_size),
        c(position_encoder)
    )
    tgt_embedder = nn.Sequential(
        Embedding(model_dim, tgt_vocab_size),
        c(position_encoder)
    )
    generator = Generator(model_dim, tgt_vocab_size)

    model = EncoderDecoderTF(
        encoder,
        decoder,
        src_embedder,
        tgt_embedder,
        generator
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model
    

