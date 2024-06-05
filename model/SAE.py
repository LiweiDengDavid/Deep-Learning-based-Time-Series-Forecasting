'''
Author:邓力玮
Date：2022.12.29
'''
import torch
import torch.nn as nn



#进行数据压缩，对数据的seq_len维度进行压缩
class SAE_encoder(nn.Module):
    def __init__(self,seq_len=96,hidden_size1=72,hidden_size2=48):
        '''
        Arg:
        seq_len=96代表输入数据的第二个维度（时间维度：已知多长的时间序列）
        hidden_size1=72是隐藏层1的神经元数量，hidden_size2=48是隐藏层2的神经元数量
        其中hidden_size2就是经过SAE_encoder压缩过后的数据的时间维度
        '''
        super(SAE_encoder, self).__init__()
        # 输入是(batch_size,in_channels,seq_len)-->输出为(batch_size,in_channels,hidden_size1)
        # 其中in_channels表示的是数据的特征维度,本任务中为7
        self.hidden1=nn.Linear(seq_len,hidden_size1)
        # 输入是(batch_size,in_channels,hidden_size1)-->输出为（batch_size,in_channels,hidden_size2）
        self.hidden2=nn.Linear(hidden_size1,hidden_size2)
    def forward(self,x):
        x=x.permute(0,2,1) # 对输入数据x进行转置，处理的是seq_len维度
        x=torch.relu(self.hidden1(x)) # 输出为(batch_size,in_channels,hidden_size1)
        x=self.hidden2(x) # 输出为（batch_size,in_channels,hidden_size2）
        return x


class LSTM(nn.Module):#输入的数据维度（seq_len，batch_size,dim),因为使用了SAE进行数据压缩，把时间维度从seq_len变成hidden_size2
    #------------------------------------------------------------------------------------------------
    #   输入的数据维度（hidden_size2，batch_size,dim)）
    #   x(hidden_size2,batch_size,dim),为了方便以后的seq_len=hidden_size2
    #   hidden_size(num_layers,batch_size,hidden_size)=(1,8,128)
    #   cell(num_layer,batch_size,hidden_size)=(1,8,128)
    #   output(seq_len,batch_size,hidden_size)=（96，1,128）
    #   已知96天通过LSTM得到输出压缩为1天的，然后把得到的预测值拼回去，滑动到下一个96天
    #------------------------------------------------------------------------------------------

    def __init__(self,seq_len,pred_len,dim,batch_size,hidden_size=128,num_layers=1,):
        '''
        Args:
        :param seq_len: 表示已知的时间维度，这里应该要和SAE_encoder的hidden_size2
        :param pred_len: 表示预测的时间序列的长度
        :param dim: 表示数据的特征维度
        :param hidden_size:表示隐藏层的单元数
        :param num_layers:使用多少层lstm网络
        :param batch_size:时间序列的batch_size
        '''
        super(LSTM, self).__init__()
        self.seq_len=seq_len # seq_len是经过数据压缩后的长度
        self.dim=dim
        self.pred_len=pred_len
        self.num_layers=num_layers
        self.hidden_sie=hidden_size
        self.batch_size=batch_size
        self.total_len=seq_len+pred_len # 一次任务总共的时间序列
        # 输入是（seq_len,batch_size,dim）-->输出是（seq_len,batch_size,hidden_size）
        # 其中seq_len应该为SAE_encoder的hidden_size2
        self.lstm=nn.LSTM(input_size=dim,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=False) # 初始化一个LSTM网络
        # 输入为（seq_len,batch_size,hidden_size)-->输出为（seq_len,batch_size,dim）
        self.fc_dim=nn.Linear(hidden_size,dim)#压缩数据特征维度
        #  输入为（dim,batch_size,seq_len）-->(dim,batch_size,1)
        self.fc_time=nn.Linear(seq_len,1)#压缩时间维度

    # 每一次只预测下一天的时间序列
    def pred_onestep(self,x,hidden,cell): # 输入为（seq_len,batch_size,dim）
        output,(hidden,cell)=self.lstm(x,(hidden,cell))#output是（seq_len,batch_size,hidden_size）
        output=self.fc_dim(output)#输出是（seq_len,batch_size，dim）
        output=output.permute(2,1,0)#变为(dim,batch_size,seq_len),为了对时间维度进行操作
        output=self.fc_time(output)#输出为（dim,batch_size,1）
        output=output.permute(2,1,0)#变回（1,batch_size,dim），1表示一个时间点
        return output # （1,batch_size,dim）

    #  -------------------------------------------------------------------------
    #  步骤：
    # （1）每一次使用pred_onestep来只预测当前已知的时间序列的下一天的时间序列
    # （2)然后得到结果在拼接回已知的时间序列中，然后滑动窗口得到新的已知的时间序列
    # （3）循环（1）~（2），直到得到完整的需要预测的时间序列就停止循环，并且返回预测的时间序列
    #  ------------------------------------------------------------------------
    def forward(self,x):
        hidden=torch.zeros(self.num_layers,self.batch_size,self.hidden_sie).to(x.device) # hidden_size=(num_layers,batch_size,hidden_size)
        cell=torch.zeros(self.num_layers,self.batch_size,self.hidden_sie).to(x.device) # 初始化hidden和cell
        # 初始化x_cat_pred 用来装已知的时间序列和预测的时间序列
        x_cat_pred=torch.zeros(self.total_len,self.batch_size,self.dim).to(x.device) # 把预测部分也放在一起 x_cat_pred=(seq_len+pred_len,batch_size,dim)
        # 把已知的时间序列放入x_cat_pred中
        x_cat_pred[:self.seq_len,:,:]=x_cat_pred[:self.seq_len,:,:].clone()+x # 把已知的序列数据也放入到x_cat_pred中

        for i in range(self.pred_len): # 滑动多少次，预测多少长的时间序列就滑动多少次
            lstm_input=x_cat_pred[i:i+self.seq_len,:,:].clone()#获取每一次的输入lstm_input (seq_len,batch_size,dim)
            pred=self.pred_onestep(lstm_input,hidden,cell)#输出是pred （1，batch_szie，dim）
            x_cat_pred[i+self.seq_len,:,:]=x_cat_pred[i+self.seq_len,:,:].clone().unsqueeze(0)+pred#把预测值也拼接回去
        return x_cat_pred[-self.pred_len:,:,:]#把预测的在返回回去，输出是(pred_len,batch_size,dim)



class SAE(nn.Module):
    def __init__(self,arg):
        super(SAE, self).__init__()
        self.arg=arg
        # 先使用SAE_encoder把数据的时间维度进行压缩，压缩为hidden_size2，即从seq_len-->hidden_size2
        self.SAE_encoder=SAE_encoder(seq_len=self.arg.seq_len,hidden_size1=arg.seq_len*2//3,hidden_size2=arg.seq_len//2)
        # 将经过SAE_encoder压缩过后的数据，放入LSTM中进行预测任务
        self.LSTM=LSTM(seq_len=arg.seq_len//2,pred_len=self.arg.pred_len,dim=self.arg.d_feature,hidden_size=128,num_layers=1,batch_size=self.arg.batch_size)

    def forward(self,enc_x, enc_mark, y, y_mark):
        '''
        :param enc_x: 已知的时间序列 （batch_size,seq_len,dim）
        以下的 param本 model未使用，不做过多介绍
        :param enc_mark: 已知的时序序列的时间对应的时间矩阵，
        :param y:
        :param y_mark:
        :return:  x 将预测的时间序列的部分返回回去 (batch_size,pred_len,dim)
        '''
        # 其中预训练是使用自编码器的方法进行预训练
        # self.SAE_encoder.load_state_dict(torch.load('./checkpoint/SAE/SAE_encoder')) # 使用预训练的SAE_encoder的权重
        x=self.SAE_encoder(enc_x) # x shape(batch_size,dim,hidden_size2),seq_len被压缩为hidden_size2, 为了方便后面的seq_len都是表示hidden_size2
        x=x.permute(2,0,1)#输出为（seq_len,batch_size,dim）
        x=self.LSTM(x)#输出是（pred_len,batch_size,dim）
        x=x.permute(1,0,2)#输出是（batch_size,pred_len,dim）
        return x


