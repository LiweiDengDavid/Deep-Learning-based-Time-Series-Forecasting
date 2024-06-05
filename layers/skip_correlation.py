import math

import torch
import torch.nn as nn
import torch.fft


class AutoCorrelation(nn.Module):
    def __init__(self, factor=1):
        super(AutoCorrelation, self).__init__()
        self.factor = factor

    def time_delay_agg(self, V, corr):
        # V:(B, H, L, d_v), corr:(B, H, L, V)
        B, H, L = V.shape[0], V.shape[1], V.shape[2]

        #------------------------------------------------------
        #   factor是为了计算topk个数的系数
        #------------------------------------------------------
        top_k = int(self.factor * math.log(L))

        #------------------------------------------------------
        #   corr应该是计算相似性后的权值，现在挑选出最大的k个corr
        #   在L维度上进行一个选取
        #   weights是权值，delays是索引
        #------------------------------------------------------
        weights, delays = torch.topk(corr, top_k, dim=2)
        weights = torch.softmax(weights, dim=2)  # (B, H, topK, V)

        #------------------------------------------------------
        #   init_index [1,1,l,1]
        #   init_index 的目的是为了给x一个初始的值
        #   后面根据topk的索引结果，计算出实验corr，原本的x = x+coor
        #   这里通过在V后面在接一个V，无缝衔接x+corr的时延效果。
        #------------------------------------------------------
        init_index = torch.arange(L).unsqueeze(-1).unsqueeze(0).unsqueeze(0)

        #------------------------------------------------------
        #   第一维度复制B次，第二维度复制H次，第三维度复制1次，第四维度复制d_v次
        #------------------------------------------------------
        init_index = init_index.repeat(B, H, 1, V.shape[3]).to(V.device)  # (B, H, L, d_v)
        delays_agg = torch.zeros_like(V).float()

        #------------------------------------------------------
        #   没看懂这里为什么要给V复制两遍
        #   复制两次是为了直接以index相加的形式把时序delay提取出来
        #------------------------------------------------------
        V = V.repeat(1, 1, 2, 1)

        for i in range(top_k):
            weight = weights[:, :, i, :].unsqueeze(2)  # (B, H, 1, V)
            delay = delays[:, :, i, :].unsqueeze(2)  # (B, H, 1, V)

            # ------------------------------------------------------
            #   没看懂这里的index和delay
            #   init_index是初始化的arange（L）
            #   dalay是topk中每一个值的索引
            # ------------------------------------------------------
            index = init_index + delay

            # ------------------------------------------------------
            #   对V的第二维度进行索引
            #   因为V还要先roll一下，才要乘相关系数，因此pattern是进行V 的roll操作
            # ------------------------------------------------------
            pattern = torch.gather(V, dim=2, index=index)
            delays_agg = delays_agg + pattern * weight
        return delays_agg

    def forward(self, Q, K, V):
        # Q:(B, H, L, d_k), K:(B, H, S, d_k), V:(B, H, S, d_v)
        B, L, S, H = Q.shape[0], Q.shape[2], K.shape[2], K.shape[1]
        # Q:(B, H, L, d_k), K:(B, H, L, d_k), V:(B, H, L, d_v)

        #---------------
        #   B:batch
        #   N:stock_number
        #   H:head
        #   L:seq_len
        #   d:dim
        #---------------



        if L > S:
            # ------------------------------------------------------
            #   预测的长度部分用0填充？
            # ------------------------------------------------------
            zeros = torch.zeros_like(Q[:, :, :(L - S), :], device=Q.device).float()
            K = torch.cat([K, zeros], dim=2)
            V = torch.cat([V, zeros], dim=2)
        else:
            V = V[:, :, :L, :]
            K = K[:, :, :L, :]

        #------------------------------------------------
        #   通过FFT计算时移后的相关性
        #   对L维度进行一个傅里叶变换
        #------------------------------------------------

        q_fft = torch.fft.rfft(Q, dim=2)
        k_fft = torch.fft.rfft(K, dim=2)
        # ------------------------------------------------------
        #   torch.conj根据paper中的公式，这里需要对K进行共轭
        # ------------------------------------------------------
        res = q_fft * torch.conj(k_fft)

        # ------------------------------------------------------
        #   corr是变换后的相似度，有多个值，需要从中挑选最大的k个
        #   因为FFT的计算就自动默认时移的stride是1，因此变换后还是总共L个结果
        # ------------------------------------------------------
        corr = torch.fft.irfft(res, dim=2)  # (B, N,H, L, V)

        V = self.time_delay_agg(V, corr)

        return V


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor):
        super(AutoCorrelationLayer, self).__init__()
        #------------------------------------------------------
        #   d_model输出维度
        #------------------------------------------------------
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.correlation = AutoCorrelation(factor)

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model)

    def forward(self, Q, K, V):
        #------------------------------------------------
        #   这里的输入QKV其实是QKV空间变换前的原始数据
        #   对于self-attention，这里的QKV全是x
        #------------------------------------------------

        # Q:(B, L, d_model), K:(B, S, d_model), V:(B, S, d_model)



        #------------------------------------------------
        #   为什么这里要统计L和S:
        #   如果是自相关的autocorrelation 那么L=S就不用统计
        #   如果不是自相关的话，L和S可能会不一样，那么就要数据对齐
        #   主要原因在于编码器和解码器中的autocorrelation中的输入可能不全是x
        #------------------------------------------------
        B, L, S, H = Q.shape[0],Q.shape[1], K.shape[1], self.n_heads

        Q = self.W_Q(Q).reshape(B, L, H, self.d_k).transpose(1, 2)  # (B, N,H, L, d_k)
        K = self.W_Q(K).reshape(B,S, H, self.d_k).transpose(1, 2)  # (B, N,H, S, d_k)
        V = self.W_Q(V).reshape(B,S, H, self.d_v).transpose(1, 2)  # (B, N,H, S, d_v)

        out = self.correlation(Q, K, V)  # (B, H, L, d_v)
        # out = out.transpose(1, 2)  # (B, L, H, d_v)
        out = out.reshape(B, L, -1)
        out = self.fc(out)
        return out
