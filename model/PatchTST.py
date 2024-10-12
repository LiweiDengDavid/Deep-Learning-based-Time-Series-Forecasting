from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class PatchTST(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # load parameters
        c_in = configs.d_feature
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        patch_len = configs.patch_len
        stride = configs.stride

        self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                       patch_len=patch_len, stride=stride,n_layers=n_layers, d_model=d_model,
                                       n_heads=n_heads, d_k=None, d_v=None, d_ff=d_ff, norm='BatchNorm',
                                       attn_dropout=False,dropout=dropout, act='gelu', key_padding_mask='auto',
                                       padding_var=None,attn_mask=None, res_attention=True, pre_norm=False,
                                       store_attn=False,pe='zeros', learn_pe=True, fc_dropout=dropout, head_dropout=False,
                                       padding_patch='end',pretrain_head=False, head_type='flatten', individual=False,
                                       revin=True, affine=False,subtract_last=False, verbose=False)

    def forward(self, batch_x, batch_x_mark,batch_y, batch_y_mark):
        # batch_x: [Batch, Input length, Channel]

        x = batch_x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]

        return x
