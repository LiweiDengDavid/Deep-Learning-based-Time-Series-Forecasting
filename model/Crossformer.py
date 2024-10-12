import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.cross_encoder import Encoder
from layers.cross_decoder import Decoder
from layers.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from layers.cross_embed import DSW_embedding

from math import ceil

class Crossformer(nn.Module):
    def __init__(self,configs):
        super(Crossformer, self).__init__()
        self.data_dim = configs.d_feature
        self.seg_len = configs.seg_len
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.dropout = configs.dropout
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.factor = configs.factor
        self.device = configs.device
        self.win_size = configs.win_size

        # The padding operation to handle invisible sgemnet length
        self.pad_seq_len = ceil(1.0 * self.seq_len / self.seg_len) * self.seg_len
        self.pad_pred_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.seq_len_add = self.pad_seq_len - self.seq_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_seq_len // self.seg_len), self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(self.e_layers, self.win_size, self.d_model, self.n_heads, self.d_ff, block_depth = 1, \
                                    dropout = self.dropout,in_seg_num = (self.pad_seq_len // self.seg_len), factor = self.factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_pred_len // self.seg_len), self.d_model))
        self.decoder = Decoder(self.seg_len, self.e_layers + 1, self.d_model, self.n_heads, self.d_ff, self.dropout, \
                                    out_seg_num = (self.pad_pred_len // self.seg_len), factor = self.factor)
        
    def forward(self,batch_x, batch_x_mark,batch_y, batch_y_mark):

        batch_size = batch_x.shape[0]
        x_seq = self.enc_value_embedding(batch_x)
        x_seq += self.enc_pos_embedding  # Location information for encoder
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)
        # encoder is the same module repeated multiple times (with different parameters), and is a module that spans time dimensions

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        # It is equivalent to repeating the first dimension as bachsize to facilitate subsequent participation in modular arithmetic

        predict_y = self.decoder(dec_in, enc_out)

        return predict_y[:, :self.pred_len, :]