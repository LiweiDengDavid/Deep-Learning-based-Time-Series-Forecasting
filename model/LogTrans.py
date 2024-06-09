import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.reformer_transformer_encdec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.LogTrans_embeded import LogTrans_DataEmbedding
from layers.probattention import FullAttention, ProbAttention, AttentionLayer
import numpy as np


class LogTrans(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, args,
                 factor=5,  e_layers=3, d_layers=2,bucket_size = 4,n_hashes = 4,pos = False, activation='gelu'):
        super(LogTrans, self).__init__()

        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.d_feature = args.d_feature
        self.d_mark = args.d_mark
        self.d_model = args.d_model
        self.d_ff = args.d_ff


        self.output_attention = None

        # Embedding
        self.enc_embedding = LogTrans_DataEmbedding(self.d_feature, self.d_mark, self.d_model, args.dropout, pos)
        # Encoder
        Attn = FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=args.dropout, output_attention=self.output_attention),
                                   self.d_model, args.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=args.dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projection = nn.Linear(self.d_model, args.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, y_batch, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        pred_init = torch.zeros_like(y_batch)
        x_dec = torch.cat([y_batch[:, :self.label_len, :], pred_init[:, -self.pred_len:, :]], dim=1)

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