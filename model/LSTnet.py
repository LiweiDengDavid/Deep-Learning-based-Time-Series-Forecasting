import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTnet(nn.Module):
    def __init__(self,args,d_hids=5,kernel=6,skip=7,highway_window=24,dropout=0.2,activation='sigmoid'):
        super(LSTnet, self).__init__()
        # self.P = args.window
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.d_feature = args.d_feature
        self.hidR = args.d_model
        self.hidC = args.d_model
        self.hidS = d_hids
        self.Ck = kernel
        self.skip = skip
        self.pt = int((self.seq_len - self.Ck) / self.skip)
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.d_feature))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.pt * self.hidS, self.d_feature)
        else:
            self.linear1 = nn.Linear(self.hidR, self.d_feature)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (activation == 'sigmoid'):
            self.output = F.sigmoid
        if (activation == 'tanh'):
            self.output = F.tanh

    def pred_onestep(self, x):

        batch_size = x.size(0)

        # CNN
        #   c[batch,1,seq,dim]
        c = x.view(-1, 1, self.seq_len, self.d_feature)
        #   c[batch,new_dim=50,蒸馏后的seq,1]
        #   这种CNN处理有点莫名其妙
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN

        r = c.permute(2, 0, 1).contiguous()
        #   r[蒸馏后的seq,batch,new_Dim=50]
        #   把batch放中间是为了gru操作
        _, r = self.GRU1(r)
        #   还是只输出一个时间点的隐藏单元
        r = self.dropout(torch.squeeze(r, 0))

        #   r[new_seq,new_dim]
        # skip-rnn

        #   c[batch,new_dim,seq]
        if (self.skip > 0):
            #   对c的seq维度进行切片
            #   s[batch,new_dim,切片后的seq:self.pt * self.skip]
            #   以skip为周期，pt个完整周期
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            #   s[batch,dim=50,self.pt,self.skip]把一个长的seq拆成几个完整的周期
            s = s.view(batch_size, self.hidC, int(self.pt), self.skip)

            #   s[pt表示完整周期的个数,batch,skip表示一个周期的长度,dim]
            s = s.permute(2, 0, 3, 1).contiguous()
            #   s[pt,batch*skip,dim],扩增了batch-size,也就是让seq=pt，找周期和周期之间的时间先后关系，h_1来自x_1 h_2来自x_(1+skip)即下一个周期的对应值
            #   这里源代码可能写错了s[skip,batch*pt,dim]batch扩增pt倍而不是skip倍
            s = s.view(int(self.skip), batch_size * self.pt, self.hidC)
            #   gru后：s[1,batch*pt,dim]也就是每一个独立周期输出一个隐藏单元
            _, s = self.GRUskip(s)
            #   有多少个周期就输出多少个隐藏单元
            s = s.view(batch_size, self.pt * self.hidS)
            s = self.dropout(s)
            #   把普通GRU对于每一个长序列的隐藏层结果以及skip-gru对于长序列中每个周期序列的隐藏层结果全部拼接起来
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.d_feature)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        #   x[batch,seq,dim]
        #   先初始化生成一个预测矩阵：
        pred_zero = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
        # ---------------------------------------------
        #   原数据矩阵的[I/2:I]拼上长度为[O]的零矩阵
        #   这样改应该更合理一点
        # ---------------------------------------------
        x_cat_pred = torch.cat([x_enc[:, :self.seq_len, :], pred_zero], dim=1).float().to(x_dec.device)

        for i in range(self.pred_len):
            x = x_cat_pred[:,i:i+self.seq_len,:].clone()
            res = self.pred_onestep(x)
            x_cat_pred[:,self.seq_len+i,:] += res

        return x_cat_pred[:,-self.pred_len:,:]




