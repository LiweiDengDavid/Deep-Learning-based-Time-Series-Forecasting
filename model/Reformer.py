import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.reformer_transformer_encdec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.ReformerLayer import ReformerLayer
from layers.Embed import DataEmbedding
import numpy as np


class Reformer(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, args,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,bucket_size = 4,n_hashes = 4,
                 dropout=0.05,pos = False, activation='gelu'):
        super(Reformer, self).__init__()

        self.pred_len = args.pred_len
        self.output_attention = None
        self.d_feature = args.d_feature
        self.c_out = args.c_out
        self.d_mark = args.d_mark

        # Embedding
        self.enc_embedding = DataEmbedding(self.d_feature, d_model)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, d_model, n_heads, bucket_size=bucket_size,
                                  n_hashes=n_hashes),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, self.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, y_batch, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_dec = torch.zeros_like(y_batch)
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]  # [B, L, D]
