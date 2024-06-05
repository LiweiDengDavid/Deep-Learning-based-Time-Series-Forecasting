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
        # Attention
        #   Prob：稀疏attention Full:普通的attention
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
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(self.d_model, args.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        #----------------------------------------------------------
        #   x_enc[B,S,D]         x_mark_enc[B,S,D]
        #   x_dec[B,L+P,D_time]         x_dec_mark[B,L+P,D_time]
        #----------------------------------------------------------
        dec_inp = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
        # ---------------------------------------------
        #   原数据矩阵的[I/2:I]拼上长度为[O]的零矩阵
        #   这样改应该更合理一点
        # ---------------------------------------------
        dec_x = torch.cat([x_dec[:, :self.label_len, :], dec_inp], dim=1).float().to(x_dec.device)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #enc_out[B,S,512:d_model]

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # enc_out [B,L,512] [batch,seq_len蒸馏后,dim]

        dec_out = self.dec_embedding(dec_x, x_mark_dec)
        #   dec_out[B,L+P,512]

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #   dec_out[B,L+P,512]
        dec_out = self.projection(dec_out)
        #   映射到输出维度
        #   [B,L+P,D]

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
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
        #训练生成器时，z是fake input
        validity = self.model(z)
        #最终让时间维度S降为1
        return validity.permute(0,2,1)



def loss_quantile(pred, labels, quantile=0.5):
    #mu[batch,预测长度]
    #labels[batch,预测长度]
    #分位数：0.5 当等于0.5的时候就是无偏的就相当于mae
    loss = 0
    for i in range(pred.shape[1]):
        #把每个时间戳单独拿出来
        mu_e = pred[:, i]
        labels_e = labels[:, i]

        I = (labels_e >= mu_e).float()
        each_loss = 2*(torch.sum(quantile*((labels_e -mu_e)*I)+ (1-quantile) *(mu_e- labels_e)*(1-I)))
        loss += each_loss

    return loss






