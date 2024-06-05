import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer


class NBeatsNet(nn.Module):
    def __init__(self,args):
        super(NBeatsNet, self).__init__()
            # stack_types = (TREND_BLOCK, SEASONALITY_BLOCK),
            # nb_blocks_per_stack,  # 先趋势堆栈，然后季节性堆栈，最后还有一个全连接堆栈
            # pred_len,  # 预测长度
            # seq_len,  # 输入的时间长度，也是每个block输出的复原学到的输入的长度
            # thetas_dim,  # 每个stack里面的theta的维度
            # # 相当于theta^(f,s)维度是4*N,theta^(f,s)维度是8*N，全连接变成4*N和8*N。
            # share_weights_in_stack,  # 是否共享权重（每个stack里面的g^b和g^f里面的全连接权重共享）
            # hidden_layer_units,  # block里面的4个FC最后弄出来的维度
            # nb_harmonics = None
        SEASONALITY_BLOCK = 'seasonality'
        TREND_BLOCK = 'trend'
        GENERIC_BLOCK = 'generic'
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.hidden_layer_units = int(args.pred_len*3)
        self.nb_blocks_per_stack = len(args.d_nbeat)
        self.share_weights_in_stack = False # 是否共享权重（每个stack里面的g^b和g^f里面的全连接权重共享）
        self.nb_harmonics = None
        self.stack_types = (TREND_BLOCK, SEASONALITY_BLOCK)
        self.stacks = []
        self.thetas_dim = args.d_nbeat
        self.parameters = []
        self.device = args.device
        # print('| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units, self.thetas_dim[stack_id],
                    self.device, self.seq_len, self.pred_len,
                    self.nb_harmonics
                )
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == 'seasonality':
            return SeasonalityBlock
        elif block_type == 'trend':
            return TrendBlock
        else:
            return GenericBlock


    def forward(self,batch_x, batch_y, batch_x_mark, batch_y_mark):
        backcast = squeeze_last_dim(batch_x).permute(0,2,1)
        forecast = torch.zeros(size=(backcast.shape[0], backcast.shape[1],self.pred_len))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b.to(self.device)
                forecast = forecast.to(self.device) + f.to(self.device)
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
        backcast = backcast.permute(0,2,1)
        forecast = forecast.permute(0,2,1)
        return forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[2], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    seasonality_output = torch.zeros(thetas.shape[0],thetas.shape[1],S.shape[-1])
    for i in range(len(thetas)):# 由于增加了batch维度，这里对batch里面的每个样本都进行一次与矩阵T的相乘
        seasonality_output[i] = thetas[i].mm(S.to(device))
    return seasonality_output


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    trend_output = torch.zeros(thetas.shape[0],thetas.shape[1],T.shape[-1])
    for i in range(len(thetas)):# 由于增加了batch维度，这里对batch里面的每个样本都进行一次与矩阵T的相乘
        trend_output[i] = thetas[i].mm(T.to(device))
    return trend_output

def linear_space(seq_len, pred_len, is_forecast=True):
    horizon = pred_len if is_forecast else seq_len
    return np.arange(0, horizon) / horizon


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, seq_len=10, pred_len=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(seq_len, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace = linear_space(seq_len, pred_len, is_forecast=False)
        self.forecast_linspace = linear_space(seq_len, pred_len, is_forecast=True)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(self.device)))  # 对时间180进行全连接，没问题
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'seq_len={self.seq_len}, pred_len={self.pred_len}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, seq_len=10, pred_len=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, seq_len,
                                                   pred_len, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, thetas_dim, device, seq_len,
                                                   pred_len, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)  # 1,24,1024
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, seq_len=10, pred_len=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, device, seq_len,
                                         pred_len, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, seq_len=10, pred_len=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, device, seq_len, pred_len)

        self.backcast_fc = nn.Linear(thetas_dim, seq_len)
        self.forecast_fc = nn.Linear(thetas_dim, pred_len)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast
