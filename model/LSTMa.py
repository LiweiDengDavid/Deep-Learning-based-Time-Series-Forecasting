from torch import nn
from torch.nn.utils import weight_norm
from layers.LSTMa_attention import *
import torch


class Encoder(nn.Module):

    def __init__(self, label_len,pred_len,d_feature,d_model,d_ff):
        super().__init__()

        # 一维卷积
        self.tcn1 = nn.Sequential(
            nn.Conv1d(in_channels=d_feature, out_channels=d_model, kernel_size=14, padding=2, stride=7),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.tcn2 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=7, padding=2, stride=3),
            nn.BatchNorm1d(d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.tcn3 = nn.Sequential(
            nn.Conv1d(in_channels=d_ff, out_channels=d_ff, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
    def forward(self,x):
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = torch.mean(x, dim=-1).unsqueeze(-1)

        return x

class Decoder(nn.Module):
    def __init__(self, label_len,pred_len,d_feature,d_model,d_ff):
        super().__init__()

        self.label_len = label_len
        self.pred_len = pred_len

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_r = nn.Linear(in_features=d_feature, out_features=d_ff, bias=True)
        self.input_i = nn.Linear(in_features=d_feature, out_features=d_ff, bias=True)
        self.input_n = nn.Linear(in_features=d_feature, out_features=d_ff, bias=True)

        self.hidden_r = nn.Linear(in_features=d_ff, out_features=d_ff, bias=False)
        self.hidden_i = nn.Linear(in_features=d_ff, out_features=d_ff, bias=False)
        self.hidden_h = nn.Linear(in_features=d_ff, out_features=d_ff, bias=False)

        self.out_fc1 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.out_fc2 = nn.Linear(in_features=d_model, out_features=d_model)
        self.out_fc3 = nn.Linear(in_features=d_model, out_features=d_feature)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        self.attention = hidden_attention(d_ff,d_ff)

    def step_forward(self, x, hidden, encoder_output,hidden_array
                     ):
        '''
        构建gru核心式
        '''

        B, S, D = x.size()
        msg = hidden
        msg, hidden, encoder_output = msg.permute(0, 2, 1), hidden.permute(0, 2, 1), encoder_output.permute(0, 2, 1)

        # GRU核心式
        r = torch.sigmoid(self.input_r(x) + self.hidden_r(msg))
        z = torch.sigmoid(self.input_i(x) + self.hidden_i(msg))
        n = torch.tanh(self.input_n(x) + r * self.hidden_h(msg))
        hidden = (1 - z) * n + z * hidden

        # 输出全连接
        hidd = hidden
        hidd = self.dropout1(self.leaky_relu(self.out_fc1(hidd)))
        hidd = self.dropout2(self.leaky_relu(self.out_fc2(hidd)))

        # 引入残差
        pred_ = x + pred
        hidden = hidden.permute(0, 2, 1)

        return pred_, hidden, pred





class LSTMa(nn.Module):
    def __init__(self,label_len = 48,pred_len=96, d_feature=7, d_mark=4, d_model=512, d_ff=1024,
                 dropout=0.05):
        super().__init__()
        self.label_len = label_len
        self.pred_len = pred_len

        self.dropout = dropout

        self.encoder =
        self.decoder =

    def forward(self,enc_x, enc_mark, y, y_mark):





