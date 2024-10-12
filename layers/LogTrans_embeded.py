import math

import torch
from torch import nn
import torch.nn.functional as F
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.pe[: x.size(1), :]
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, d_feature, d_model):
        super(TokenEmbedding, self).__init__()
        self.embed = nn.Linear(d_feature, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class TimeEmbedding(nn.Module):
    def __init__(self, d_mark, d_model):
        super(TimeEmbedding, self).__init__()
        self.embed = nn.Linear(d_mark, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class LogTrans_DataEmbedding(nn.Module):
    def __init__(self, d_feature, d_mark, d_model, dropout=0.1, pos=False):
        super(LogTrans_DataEmbedding, self).__init__()
        self.pos = pos

        self.value_embedding = TokenEmbedding(d_feature=d_feature, d_model=d_model)
        self.time_embedding = TimeEmbedding(d_mark=d_mark, d_model=d_model)
        self.context_embedding = context_embedding(d_feature=d_feature, d_model=d_model)

        if self.pos:
            self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if self.pos:
            x = self.value_embedding(x) + self.position_embedding(x) + self.time_embedding(x_mark) + self.context_embedding(x)
        else:
            x = self.value_embedding(x) + self.time_embedding(x_mark) + self.context_embedding(x)

        return self.dropout(x)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, d_feature=1, d_model=256, k=5):
        super(context_embedding, self).__init__()
        self.causal_convolution = CausalConv1d(d_feature, d_model, kernel_size=k)

    def forward(self, x):
        x = self.causal_convolution(x.permute(0,2,1))
        return F.tanh(x.permute(0,2,1))


