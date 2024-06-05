import torch
from torch import nn

from layers.autoformer_encdec import Encoder,Decoder
from layers.tools import SeriesDecomp


class Autoformer(nn.Module):
    def __init__(self,label_len = 48,pred_len=96, d_feature=7, d_mark=4, d_model=512, d_ff=2048, e_layers=2, d_layers=1, mov_avg=25,
                 n_heads=4, factor=3, dropout=0.05, pos=False):
        super(Autoformer, self).__init__()
        self.pred_len = pred_len
        self.label_len = label_len

        #-------------------------------------------
        #   时间拆解序列，encoder和decoder
        #-------------------------------------------

        self.decomp = SeriesDecomp(mov_avg)
        self.encoder = Encoder(d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, e_layers, dropout, pos)
        self.decoder = Decoder(d_feature, d_mark, d_model, d_ff, mov_avg, n_heads, factor, d_layers, dropout, pos)


    def forward(self, enc_x, enc_mark, y, y_mark):
        # ---------------------------------------------
        #   模型输入：
        #   enc_x, enc_mark, dec_x, dec_mark
        #   enc_x：归一化后数据的长度为【I】
        #   enc_mark：时间索引数据 长度为【I】
        #   dec_x：归一化后的X_des [I/2+O]
        #   dec_mark:时间索引数据 长度为【I/2+O】
        # ---------------------------------------------

        dec_inp = torch.zeros_like(y[:, -self.pred_len:, :]).float()
        # ---------------------------------------------
        #   原数据矩阵的[I/2:I]拼上长度为[O]的零矩阵
        #   这样改应该更合理一点
        # ---------------------------------------------

        # dec_inp = torch.cat([batch_y[:, self.label_len:2*self.label_len, :], dec_inp], dim=1).float().to(self.device)

        dec_x = torch.cat([y[:, :self.label_len, :], dec_inp], dim=1).float().to(y.device)
        dec_mark = y_mark
        #-------------------------------------------
        #   编码器的输入只要归一化后的数据 长度为【I】
        #   enc_mark的作用仅仅是embedding来使用的
        #   而embedding这一项在时序数据中可以天然的丢弃掉
        #-------------------------------------------
        enc_out = self.encoder(enc_x, enc_mark)

        # ---------------------------------------------
        #   根据paper的部分初始化的部分，mean和zero都要填充占位
        #   mean的长度【pred_len】即[O]
        #   zero的长度【pred_len】即[O]
        # ---------------------------------------------
        mean = torch.mean(enc_x, dim=1).unsqueeze(1).repeat(1,self.pred_len, 1)
        zeros = torch.zeros([dec_x.shape[0], self.pred_len, dec_x.shape[2]], device=dec_x.device)

        #-----------------------------------------
        #   这里的输入时序拆解的数据【I/2，I】，只对于前半部分进行拆解
        #-----------------------------------------
        trend, season = self.decomp(dec_x[:,:-self.pred_len, :])

        #------------------------------------------------
        #   初始化trend和season
        #   维度【I/2+O】，其中前【I/2，I】由原始数据拆解得到，【I，O】由均值或zero拼接得到
        #------------------------------------------------
        trend = torch.cat([trend, mean], dim=1)
        season = torch.cat([season, zeros], dim=1)


        #------------------------------------------------
        #   到解码器才需要用到初始化的trend和season数据
        #   编码器是只需要原始数据作为输入的
        #------------------------------------------------
        dec_out = self.decoder(season, trend, dec_mark, enc_out)

        # ------------------------------------------------------
        #   输出最终的预测部分结果
        # ------------------------------------------------------
        return dec_out[:,-self.pred_len:, :]
