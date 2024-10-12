from torch import nn
from torch.nn.utils import weight_norm
import torch


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2,self.relu2, self.dropout2,self.chomp2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self,args, kernel_size = 5, dropout = 0.2,num_channels=[30,30,30,30,30,30,30,30]):
        super(TCN, self).__init__()
        self.pre_len = args.pred_len
        self.tcn = TemporalConvNet(1, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.pre_len_linear = nn.Linear(args.seq_len,self.pre_len)
        self.args=args
        self.init_weights()


    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # Model description:
        # 1. input batchsize, seqlen, stock_num, output batchsize, pre_len,stock_num
        # 2. Input length = output length
        # 3. Tried many times, but still create an extra channel as the channel of convolution, the effect is more normal, that is, the input is converted into four dimensions, an extra dimension as the channel of convolution.
        
        # 输入 batchsize，seqlen,stock_num
        if self.pre_len<=self.args.seq_len:
            input = batch_x[:,-self.pre_len:,:]
        else:
            input = batch_x
        input = input.unsqueeze(-1)  # batchsize，seqlen,stock_num,1
        input = input.permute(2,3,1,0) # sto_num,1,seqlen,batchsize The second dimension is used for convolution, and will become 1 later after full connectivity.
        output_total = None
        for i in range(input.shape[-1]):
            output = self.tcn(input[:,:,:,i])
            output = output.permute(0,2,1)
            output = self.linear(output)
            output = output.permute(2,1,0)
            if i == 0:
                output_total = output
            else:
                output_total = torch.cat((output_total,output))  # If there is more than one batch, then stack on top of it.

        if self.pre_len<=self.args.seq_len:
            output_total=output_total
        else:
            output_total=self.pre_len_linear(output_total.transpose(-1,-2))
            output_total=output_total.transpose(-1,-2)
        return output_total