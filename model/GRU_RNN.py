import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD


class GRU_RNN_Model(nn.Module):
    def __init__(self, arg):
        super(GRU_RNN_Model, self).__init__()
        self.arg = arg
        self.seq_len = arg.seq_len
        self.input_dim = arg.d_feature
        self.batch_size = arg.batch_size
        self.output_dim = arg.d_feature
        self.pred_len = arg.pred_len
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=256, num_layers=1, bias=True, batch_first=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, self.output_dim)
        self.fc2 = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, batch_y, batch_x_mark, batch_y_mark):
        x, hi = self.gru(x)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        x = self.softmax(x)
        x = self.relu(x)
        hid = self.relu(hi)
        hid = self.dropout(hid)
        x = self.dropout(x)
        return x


