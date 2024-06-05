'''
Authors:邓力玮
'''
import torch
import torch.nn as nn


# 使用堆叠的Autoender形成SAE网络，然后直接输出预测值
# 对时间维度进行操作 因为paper是说希望先记住时间序列然后在进行推理
# 所以是对时间维度进行操作
class Autoencoder(nn.Module): #输入是已知的时间序列，输出是预测时间序列
    def __init__(self,args,hidden_size=300):
        '''
        Arg:
        seq_len=96代表输入数据的第二个维度（时间维度：已知多长的时间序列）
        pred_len=96代表预测时间有多长,hidden_size=300是隐藏层的神经元数量
        '''
        super(Autoencoder, self).__init__()
        seq_len=args.seq_len
        pred_len=args.pred_len
        # 输入是（batch_size,dim,seq_len）-->输出为（batch_size,dim,hidden_size）
        self.fc1=nn.Linear(seq_len,hidden_size)
        # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        self.fc2=nn.Linear(hidden_size,hidden_size)
        # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        self.fc3=nn.Linear(hidden_size,hidden_size)
       # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        self.fc4=nn.Linear(hidden_size,hidden_size)
        # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,pred_len）
        self.fc5=nn.Linear(hidden_size,pred_len)
    def forward(self, enc_x, enc_mark, y, y_mark):
        '''
        Args:
        :param enc_x: 已知的时间序列 （batch_size,seq_len,dim）
        以下的 param本 model未使用，不做过多介绍
        :param enc_mark: 已知的时序序列的时间对应的时间矩阵，
        :param y:
        :param y_mark:
        :return:  x_cat_pred[:,-self.pred_len:,:] 将预测的时间序列的部分返回回去 (batch_size,pred)len,dim)
        '''
        enc_x=enc_x.permute(0,2,1) # 输入为（batch_size,seq_len,dim）--》输出为（batch_size,dim,seq_len）
        x=torch.sigmoid(self.fc1(enc_x)) # 输入是（batch_size,dim,seq_len）-->输出为（batch_size,dim,hidden_size）
        x=torch.sigmoid(self.fc2(x)) # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        x=torch.sigmoid(self.fc3(x)) # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        x=torch.sigmoid(self.fc4(x)) # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        x=self.fc5(x)  # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,pred_len）
        x=x.permute(0,2,1)#输入是（batch_size,dim,pred_len）输出是（batch_size,pred_len,dim）
        return x #返回值的shape是（batch_size，pred_len，dim）


