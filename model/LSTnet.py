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
        c = x.view(-1, 1, self.seq_len, self.d_feature)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        #   c[batch,new_dim,seq]
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()

            s = s.view(int(self.skip), batch_size * self.pt, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.pt * self.hidS)
            s = self.dropout(s)

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
        #   First initialise to generate a prediction matrix：
        pred_zero = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
        x_cat_pred = torch.cat([x_enc[:, :self.seq_len, :], pred_zero], dim=1).float().to(x_dec.device)

        for i in range(self.pred_len):
            x = x_cat_pred[:,i:i+self.seq_len,:].clone()
            res = self.pred_onestep(x)
            x_cat_pred[:,self.seq_len+i,:] += res

        return x_cat_pred[:,-self.pred_len:,:]




