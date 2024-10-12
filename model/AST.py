import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import ProbMask
from layers.informer_encdec import Encoder, EncoderLayer, ConvLayer, Decoder, DecoderLayer
from layers.probattention import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class AST(nn.Module):
    def __init__(self, args,
                 factor=5, e_layers=3, d_layers=2, attn='prob',pos = False, activation='gelu',
                 output_attention=False, distil=True, mix=True,):
        super(AST, self).__init__()
        self.pred_len = args.pred_len
        self.attn = attn
        self.label_len = args.label_len
        self.output_attention = output_attention

        self.d_feature = args.d_feature
        self.d_mark = args.d_mark
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # Encoding
        self.enc_embedding = DataEmbedding(self.d_feature,  self.d_model, args.dropout, pos)
        self.dec_embedding = DataEmbedding(self.d_feature, self.d_model, args.dropout, pos)
        Attn = FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=args.dropout, output_attention=output_attention),
                                   self.d_model, args.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=args.dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=args.dropout, output_attention=False),
                                   self.d_model, args.n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=args.dropout, output_attention=False),
                                   self.d_model, args.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=args.dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(self.d_model, args.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        #----------------------------------------------------------
        #   x_enc[B,S,D]         x_mark_enc[B,S,D]
        #   x_dec[B,L+P,D_time]         x_dec_mark[B,L+P,D_time]
        #----------------------------------------------------------
        dec_inp = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
        dec_x = torch.cat([x_dec[:, :self.label_len, :], dec_inp], dim=1).float().to(x_dec.device)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)


        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(dec_x, x_mark_dec)
        #   dec_out[B,L+P,512]

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #   dec_out[B,L+P,512]
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, P, D]



class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.label_len+args.pred_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        #z[B,S,d]->[B,d,S]
        z = z.permute(0,2,1)
        validity = self.model(z)
        return validity.permute(0,2,1)



def loss_quantile(pred, labels, quantile=0.5):
    loss = 0
    for i in range(pred.shape[1]):
        mu_e = pred[:, i]
        labels_e = labels[:, i]

        I = (labels_e >= mu_e).float()
        each_loss = 2*(torch.sum(quantile*((labels_e -mu_e)*I)+ (1-quantile) *(mu_e- labels_e)*(1-I)))
        loss += each_loss

    return loss






