import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # decoder中的crossattention
        #   k[B,蒸馏后的S,8,64]
        #   q[B,L,8,64]输入的长度是不一样的，因此来自不同的地方
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 矩阵相乘相当于KQ的操作
        # scores[B,8,L,蒸馏后的S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        #   该语句相当于矩阵乘法
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)




class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # ---------------------------------------------------
        #   Q [B, H, L, D]
        #   [batch_size,num_head,seq_len,hidden_embedding]
        #   分别统计L_k,L_Q是为了区分decoder和encoder中两个attention结构输入的区别
        # ---------------------------------------------------
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K

        # ---------------------------------------------------
        #   复制了一个维度
        # ---------------------------------------------------
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        #   K_sample[B,8,S,topk,64]
        #   S个Q每个对应topk个K，K的维度是64
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # ---------------------------------------------------
        #   debug测试维度
        #   a[B,8,S,1,64]
        #   b[B,8,S,64,topk]
        #   Q_K_sample [B,8,S,topk]
        #   S个Q，每个Q和topk个K的内积值
        # ---------------------------------------------------
        a = Q.unsqueeze(-2)
        b = K_sample.transpose(-2, -1)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement

        # ---------------------------------------------------------
        #   先对dim维度进行求max得到最大的内积值
        #   Q_K_sample.max(-1)是一个列表，一个是值一个是索引 [0]取值
        #   相当于最大值减去均值
        #   M[B,8,S]
        # ---------------------------------------------------------

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # ----------------------------------------
        #   从S里面选topk个索引值，选topk个Q
        # ----------------------------------------
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        #   Q [B, H, L, D]
        #   K [B,H,L,D]

        #   Q_reduce[B,H,topk,D]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        #   Q_K [B,H,topk,S]

        #   表示topk个Q和S个K之间的相关性
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):

        B, H, L_V, D = V.shape

        #   先把S个V全部用V的均值进行初始化
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        #   scores_top [B, H, topk, S] ,index对应25的index
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        #   对行进行softmax，即对q对每个k的结果进行softmax
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        #   选取初始化V中的特定值根据index进行一个重新赋值，不在index中的值依然保持原本的初始化值
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):

        #----------------------------------------
        #   Q[B,L,8,64]
        #   K[B,L,8,64]
        #   V[B,L,8,64]
        #----------------------------------------

        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        #   scores_top [B, H, topk, L] ,index对应topk的index
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context

        #   values[batch,num_head,seq_len,dim]
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        #   [B,8,L,64]保持和V一样的维度

        #   transpose后[B,L,8,64]
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        #   把头数合在一块[B,L,8*64]
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
