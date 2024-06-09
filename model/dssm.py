import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class DeepSSM(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(DeepSSM, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.h_dim = 32

        # Embedding
        self.enc_embedding = DataEmbedding(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.dense_l_prior = nn.Linear(configs.c_out,self.h_dim)
        self.dense_P_prior = nn.Linear(configs.c_out,self.h_dim)
        self.activation = nn.Softplus()

        self.dense_F = nn.Linear(configs.c_out, self.h_dim*self.h_dim)
        self.dense_H = nn.Linear(configs.c_out, configs.c_out*self.h_dim)
        self.dense_b = nn.Linear(configs.c_out, configs.c_out)
        self.dense_w = nn.Linear(configs.c_out, self.h_dim)
        self.dense_v = nn.Linear(configs.c_out, configs.c_out)
        self.sigma_0 = nn.Linear(configs.c_out, self.h_dim)
        self.mu_0 = nn.Linear(configs.c_out, self.h_dim)
        self.activation_normal = nn.ReLU()


    def kalman_step(self,F,H,w,b,v,l,e):

        l = torch.matmul(F,l) + w*e
        z = torch.matmul(H,l) + b + v*e

        return l,z


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):


        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)



        outputs = dec_out[:, -self.pred_len:, :]

        B,S,D = outputs.shape
        F = self.dense_F(outputs).unsqueeze(-1).view(B,S,self.h_dim,self.h_dim)
        H = self.dense_H(outputs).unsqueeze(-1).view(B, S, D, self.h_dim)
        b = self.dense_b(outputs).unsqueeze(-1)
        w = self.dense_w(outputs).unsqueeze(-1)
        v = self.dense_v(outputs).unsqueeze(-1)

        sigma = self.activation(self.sigma_0(outputs[:,0:1,:]).unsqueeze(-1))
        mu = self.mu_0(outputs[:,0:1,:]).unsqueeze(-1)

        sigma_test = torch.ones_like(sigma)
        mu_test = torch.zeros_like(mu)

        l_0 = torch.distributions.Normal(mu_test,sigma_test).sample()
        e = torch.distributions.Normal(0,1).sample()


        l = l_0
        samples = []
        for t in range(self.pred_len):

            Ft = F[:, t:t + 1, :, :]
            Ht = H[:, t:t + 1, :, :]
            bt = b[:, t:t + 1, :, :]
            wt = w[:, t:t + 1, :, :]
            vt = v[:, t:t + 1, :, :]
            l,z_t = self.kalman_step(Ft,Ht,wt,bt,vt,l,e)
            samples.append(z_t)
        pred = torch.cat(samples,dim = 1)
        return pred.squeeze(-1)



















