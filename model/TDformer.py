import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierCrossAttention
from layers.Autoformer_EncDec import Encoder, Decoder, my_Layernorm, series_decomp,series_decomp_multi,TD_encoderlayer,TD_decoderlayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TDformer(nn.Module):
    """
    TDformer  author:sxb 2023/1/30
    """
    def __init__(self, configs):
        super(TDformer, self).__init__()

        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = False

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        #Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Trend Processing
        self.trend_pred_len = nn.Sequential(
            nn.Linear(self.pred_len+self.seq_len, self.pred_len),
            nn.ReLU(),
            nn.Linear(self.pred_len, self.pred_len),
        )

        self.trend_pred_model = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.c_out)
        )

        # Season Processing
        FA = FourierCrossAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  mode_select_method=configs.mode_select)

        self.encoder = Encoder(
            [
                TD_encoderlayer(
                    AutoCorrelationLayer(
                        FA,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                TD_decoderlayer(
                    AutoCorrelationLayer(
                        FA,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        FA,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, y_batch, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        seasonal_init, trend_init = self.decomp(x_enc)

        pred_init_season = torch.zeros_like(y_batch)
        pred_init_trend = torch.zeros_like(y_batch)

        Seasonal_init = torch.cat([seasonal_init[:, :self.seq_len, :], pred_init_season[:, -self.pred_len:, :]], dim=1)
        Trend_init = torch.cat([trend_init[:, :self.seq_len, :], pred_init_trend[:, -self.pred_len:, :]], dim=1)


        Seasonal_mark = torch.cat([x_mark_enc[:, :self.seq_len, :], x_mark_dec[:, -self.pred_len:, :]],dim=1)
        Trend_mark = torch.cat([x_mark_enc[:,:self.seq_len,:],x_mark_dec[:,-self.pred_len:,:]],dim=1)

        # Embedding
        Trend_inp = self.enc_embedding(Trend_init, Trend_mark)
        Seasonal_inp = self.dec_embedding(Seasonal_init, Seasonal_mark)

        Trend_pred_len = self.trend_pred_len(Trend_inp.permute(0,2,1))
        Trend_pred = self.trend_pred_model(Trend_pred_len.permute(0,2,1))

        enc_out, _ = self.encoder(Seasonal_inp, attn_mask=enc_self_mask)

        dec_out, _ = self.decoder(Seasonal_inp, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=None)
        output = dec_out[:, -self.pred_len:, :] + Trend_pred

        return output[:, -self.pred_len:, :]





