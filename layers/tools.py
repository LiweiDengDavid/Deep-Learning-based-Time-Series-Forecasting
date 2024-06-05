import torch
from torch import nn
import math

class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.norm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.mov_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        #-----------------------------------------
        #   这里的输入X【I/2，I】，只对于前半部分进行拆解
        #-----------------------------------------
        B,L,V = x.shape
        x = x.reshape(B,L,V)
        trend = self.mov_avg(x.permute(0,2,1)).permute(0,2,1)
        season = x - trend

        return trend.reshape(B,L,V), season.reshape(B,L,V)


class series_decomp_multi(nn.Module):
    #MOEdecmop模块，就是一系列大小不同的平均滤波器
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        #一系列不同大小的卷积核
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            #   等长卷积，卷积前后不改变时间维度的大小
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        #确定每个卷积核的权重，然后基于权重对于每个核的平均结果加权求和。
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        #res就是季节项
        res = x - moving_mean
        return res, moving_mean