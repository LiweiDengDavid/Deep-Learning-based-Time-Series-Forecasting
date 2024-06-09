import torch.nn.functional as F
from torch import nn

from layers.autocorrelation import AutoCorrelationLayer, AutoCorrelation
from layers.embeded import DataEmbedding
from layers.tools import LayerNorm, SeriesDecomp

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, mov_avg, n_heads, factor, dropout):
        super(EncoderLayer, self).__init__()
        self.self_correlation = AutoCorrelationLayer(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomp(mov_avg)
        self.decomp2 = SeriesDecomp(mov_avg)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,), bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,), bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        residual = x.clone()
        x = self.dropout(self.self_correlation(x, x, x))
        _, x = self.decomp1(x + residual)

        residual = x.clone()
        # ---------------------------------------------
        #   用conv的形式去替代FC
        # ---------------------------------------------
        B,L,V = x.shape
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        _, x = self.decomp2(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, e_layers, dropout, pos):
        super(Encoder, self).__init__()
        # ---------------------------------------------
        #   对于时间序列位置编码可以丢弃
        # ---------------------------------------------
        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout, pos)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, mov_avg,
                                                          n_heads, factor, dropout) for _ in range(e_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, enc_x, enc_mark):
        x = self.embedding(enc_x, enc_mark)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, mov_avg, n_heads, factor, dropout):
        super(DecoderLayer, self).__init__()
        #------------------------------------------------------
        #   因为decoder中有两种不同输入的auto-correlation
        #   一个接的是encoder的输出和前一个时序分解的结果 cross-correlation
        #   一个接的是前一个时序分解的结果   self-correlation
        #------------------------------------------------------

        self.self_correlation = AutoCorrelationLayer(d_model, n_heads, factor)
        self.cross_correlation = AutoCorrelationLayer(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomp(mov_avg)
        self.decomp2 = SeriesDecomp(mov_avg)
        self.decomp3 = SeriesDecomp(mov_avg)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,), bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,), bias=False)

        #------------------------------------------------------
        #   映射projection负责K，Q，V
        #------------------------------------------------------


        self.projection1 = nn.Linear(d_model, d_feature, bias=True)
        self.projection2 = nn.Linear(d_model, d_feature, bias=True)
        self.projection3 = nn.Linear(d_model, d_feature, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, season, enc_out):
        #------------------------------------------------------
        #   克隆是为了存残差
        #------------------------------------------------------
        residual = season.clone()
        season = self.dropout(self.self_correlation(season, season, season))
        trend1, season = self.decomp1(season + residual)

        residual = season.clone()
        season = self.dropout(self.cross_correlation(season, enc_out, enc_out))
        trend2, season = self.decomp2(season + residual)

        residual = season.clone()
        B,L,V = season.shape


        season = self.dropout(self.activation(self.conv1(season.permute(0, 2, 1))))
        season = self.dropout(self.conv2(season).permute(0, 2, 1))

        trend3, season = self.decomp3(season + residual)

        trend = self.projection1(trend1) + self.projection2(trend2) + self.projection3(trend3)
        return trend, season


class Decoder(nn.Module):
    def __init__(self, d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, d_layers, dropout, pos):
        super(Decoder, self).__init__()
        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout, pos)

        #------------------------------------------------------
        #   decoder是由多个decoder-layer堆叠出来的
        #------------------------------------------------------
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_feature, d_ff, mov_avg,
                                                          n_heads, factor, dropout) for _ in range(d_layers)])
        self.norm = LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_feature, bias=True)

    def forward(self, season, trend, dec_mark, enc_out):
        season = self.embedding(season, dec_mark)
        for decoder_layer in self.decoder_layers:
            tmp_trend, season = decoder_layer(season, enc_out)
            trend = trend + tmp_trend

        season = self.norm(season)
        season = self.projection(season)
        return season + trend
