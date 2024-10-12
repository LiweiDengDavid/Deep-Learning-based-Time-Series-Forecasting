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
        # factor is for calculating the number of topek's
        # ------------------------------------------------------
        top_k = int(self.factor * math.log(L))

        #------------------------------------------------------
        # corr should be the weights after calculating the similarity, now pick the largest k corrs
        # Make a pick in the L dimension
        # weights are weights, delays are indexes
        # ------------------------------------------------------
        weights, delays = torch.topk(corr, top_k, dim=2)
        weights = torch.softmax(weights, dim=2)  # (B, H, topK, V)

        #------------------------------------------------------
        # init_index [1,1,l,1]
        # init_index The purpose of init_index is to give an initial value for x
        # Later, based on the topk's index result, the experimental corr is calculated, originally x = x+coor
        # Here the delay effect of x+corr is seamlessly integrated by picking up a V after the V.
        # ------------------------------------------------------

        init_index = torch.arange(L).unsqueeze(-1).unsqueeze(0).unsqueeze(0)

        #------------------------------------------------------
        # Copy B times in the first dimension, H times in the second dimension, 1 time in the third dimension, d_v times in the fourth dimension
        # ------------------------------------------------------
        init_index = init_index.repeat(B, H, 1, V.shape[3]).to(V.device)  # (B, H, L, d_v)
        delays_agg = torch.zeros_like(V).float()
        #------------------------------------------------------
        # The reason for copying V twice is to extract the timing delay directly as an index sum.
        # ------------------------------------------------------
        V = V.repeat(1, 1, 2, 1)

        for i in range(top_k):
            weight = weights[:, :, i, :].unsqueeze(2)  # (B, H, 1, V)
            delay = delays[:, :, i, :].unsqueeze(2)  # (B, H, 1, V)

            # ------------------------------------------------------
            # init_index is the initialised range (L)
            # dalay is the index of each value in topk
            # ------------------------------------------------------
            index = init_index + delay

            # ------------------------------------------------------
            # Index the second dimension of V
            # Since V has to be rolled before the correlation coefficient is multiplied, the pattern is a roll operation on V
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

            zeros = torch.zeros_like(Q[:, :, :(L - S), :], device=Q.device).float()
            K = torch.cat([K, zeros], dim=2)
            V = torch.cat([V, zeros], dim=2)
        else:
            V = V[:, :, :L, :]
            K = K[:, :, :L, :]

        #------------------------------------------------
        # Calculate the correlation after time-shift by FFT
        # Perform a Fourier transform on the L dimension
        # #------------------------------------------------

        q_fft = torch.fft.rfft(Q, dim=2)
        k_fft = torch.fft.rfft(K, dim=2)
        # ------------------------------------------------------
        # torch.conj According to the formula in paper, here you need to conjugate K
        # ------------------------------------------------------
        res = q_fft * torch.conj(k_fft)

        # ------------------------------------------------------
        # corr is the transformed similarity, there are multiple values, you need to pick the biggest k from them.                            # corr is the transformed similarity, there are multiple values from which you need to pick the largest k. Since the FFT calculation automatically defaults to a time-shifted stride of 1, the transform is still a total of L results.
        # ------------------------------------------------------
        corr = torch.fft.irfft(res, dim=2)  # (B, N,H, L, V)

        V = self.time_delay_agg(V, corr)

        return V


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor):
        super(AutoCorrelationLayer, self).__init__()
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

        # Q:(B, L, d_model), K:(B, S, d_model), V:(B, S, d_model)

        B, L, S, H = Q.shape[0],Q.shape[1], K.shape[1], self.n_heads

        Q = self.W_Q(Q).reshape(B, L, H, self.d_k).transpose(1, 2)  # (B, N,H, L, d_k)
        K = self.W_Q(K).reshape(B,S, H, self.d_k).transpose(1, 2)  # (B, N,H, S, d_k)
        V = self.W_Q(V).reshape(B,S, H, self.d_v).transpose(1, 2)  # (B, N,H, S, d_v)

        out = self.correlation(Q, K, V)  # (B, H, L, d_v)
        # out = out.transpose(1, 2)  # (B, L, H, d_v)
        out = out.reshape(B, L, -1)
        out = self.fc(out)
        return out
