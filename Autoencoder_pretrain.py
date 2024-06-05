'''
Author:邓力玮
date：2022.12.29
这是Autoencoder的预训练code
预训练只需要data不需要对应的label，
是一个无监督的训练，并且训练使用的是逐层贪心算法
'''

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR



# 使用的是贪心算法，每一次训练的时候只更新一个层的参数，把其它层的参数都固定住

class SAE(nn.Module): #输入是已知的时间序列，输出是预测时间序列
    def __init__(self,seq_len,pred_len,hidden_size=300):
        '''
        Arg:
        in_channels=7代表输入数据的第一个维度（特征维度）；seq_len=96代表输入数据的第二个维度（时间维度：已知多长的时间序列）
        pred_len=96代表预测时间有多长,hidden_size=300是隐藏层的神经元数量
        '''
        super(SAE, self).__init__()
        # 输入(in_channels,seq_len)-->输出为(in_channels,hidden_size)
        hidden_size1=seq_len*2//3
        hidden_size2=seq_len//2
        self.fc1=nn.Linear(seq_len,hidden_size1) #输入是(7,96)，输出是（7,300）
        # 输入(in_channels,hidden_size-->输出为(in_channels,hidden_size)
        self.fc2=nn.Linear(hidden_size1,hidden_size2) #输入和输出都是(7,300)
        # 输入(in_channels,hidden_size-->输出为(in_channels,hidden_size)
        self.fc3=nn.Linear(hidden_size2,hidden_size)
        # 输入(in_channels,hidden_size-->输出为(in_channels,hidden_size)
        self.fc4=nn.Linear(hidden_size,hidden_size)
        # 输入(in_channels,hidden_size-->输出为(in_channels,pred_len)
        self.fc5=nn.Linear(hidden_size,pred_len)#输入是（7,300），输出是（7,96）
    def forward(self,x):
        x=torch.sigmoid(self.fc1(x))
        x=torch.sigmoid(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))
        x=torch.sigmoid(self.fc4(x))
        x=self.fc5(x)
        return x #返回值的shape是（7,96）



# 计算平均激活，就是经过当前层后的值的平均值，在计算KL散度的时候会使用
def rou_hat_cala(i,self,xx):
    '''
    Args:
        （i//2）+1 是训练哪一层fc
        self是使用哪个网络（sae）
        xx是输入数据
    '''
    if i == 0:
        pred = torch.sigmoid(self.fc1(xx))
        rou_hat1 = torch.mean(pred) + 1e-5  # 计算loss时需要使用,加上一个很小的数，防止为0
    elif i == 2:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        rou_hat1 = torch.mean(pred) + 1e-5  # 计算loss时需要使用,加上一个很小的数，防止为0
    elif i == 4:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        pred = torch.sigmoid(self.fc3(pred))
        rou_hat1 = torch.mean(pred) + 1e-5  # 计算loss时需要使用,加上一个很小的数，防止为0
    elif i == 6:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        pred = torch.sigmoid(self.fc3(pred))
        pred = torch.sigmoid(self.fc4(pred))
        rou_hat1 = torch.mean(pred) + 1e-5  # 计算loss时需要使用,加上一个很小的数，防止为0

    elif i == 8:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        pred = torch.sigmoid(self.fc3(pred))
        pred = torch.sigmoid(self.fc4(pred))
        pred = torch.sigmoid(self.fc5(pred))
        rou_hat1 = torch.mean(pred) + 1e-5  # 计算loss时需要使用,加上一个很小的数，防止为0
    else:rou_hat1=0
    return rou_hat1




#预训练,把不训练的其它层全部冻住（requires_grad=Fasle）
def pre_train(self,train_data):
    rou_hat = 0
    param_lst=[] #把每一层的权重的名字装起来
    #将所有的可训练参数全部设置为False
    for param in self.named_parameters(): #name_parameters()会返回层名字和权重
        param[1].requires_grad=False
        param_lst.append(param[0])

    for i in range(len(param_lst)):
        lst=list(self.named_parameters())#得到网络权重和名称
        if i%2==0:
            lst[i][1].requires_grad=True
            lst[i+1][1].requires_grad=True #逐层训练

            total_len= pred_len + seq_len

            #把训练集的数据都经过一遍网络，然后在计算经过隐藏层的平均激活
            for j in range(train_data.shape[0] // total_len):  # 总共可以取多少个total_len
                x = train_data[j * total_len:(j + 1) * total_len, :]  # 每一次取total长度
                xx = x[:seq_len, :].clone()  # 每一次已知的时间序列（seq_len,dim）
                xx = xx.unsqueeze(0)  # 升维度（1,seq_len,dim）
                xx = xx.permute(0, 2, 1)  # 输出维度是（1,dim,seq_len）

                #计算rou_hat，平均激活
                rou_hat+=rou_hat_cala(i,self,xx)
            rou_hat=rou_hat/(j+1)+1e-5 # +1e-5是为了为0
            for epoch in range(epoches):
                runing_loss = 0
                for j in range(train_data.shape[0] // total_len):  # 总共可以取多少个total_len
                    x = train_data[i * total_len :(i + 1) * total_len, :]  # 每一次取total_len长度
                    xx = x[:seq_len, :].clone()  # 每一次已知的时间序列（seq_len,dim）
                    xx = xx.unsqueeze(0)  # 升维度（1,seq_len,dim）
                    xx = xx.permute(0, 2, 1)  # 输出维度是（1,dim,seq_len）
                    pred = sae(xx)  # 输出是（1,dim,seq_len）
                    optimizer.zero_grad()
                    pred = pred.squeeze()
                    pred=pred.permute(1,0)#输出是(seq-len,dim)
                    kl=rou*torch.log(rou/rou_hat)+(1-rou)*torch.log((1-rou)/(1-rou_hat)) #计算KL散度
                    loss = loss_fn(pred, x[seq_len:, :])+kl  # 计算loss
                    loss.backward()
                    runing_loss += loss
                    optimizer.step()
                    rou_hat = rou_hat_cala(i, self, xx)
                print('第{0}个epoch的loss：{1}'.format(epoch + 1, round((runing_loss / (i + 1)).item(), 2)))
            lst[i][1].requires_grad = False
            lst[i + 1][1].requires_grad = False # 把训练好的层再次冻住



def train(net,loss_fn,optimizer,train_data,seq_len,pred_len,epoches):
    '''
    Args:
        net是需要训练的网络
        loss_fn是使用的损失函数，train_data是整个训练集
        seq_len是已知的时间序列的长度，pred_len是需要预测的时间序列长度
        epoches是外循环多少次，
    '''
    net.train()
    total_len=pred_len+seq_len
    for epoch in range(epoches):
        runing_loss = 0
        for i in range(train_data.shape[0] // total_len):  # 总共可以取多少个total_len
            x=train_data[i*total_len:(i+1)*total_len,:]#每一次取total_len长度
            xx=x[:seq_len,:].clone()#每一次已知的时间序列（seq_len,dim）
            xx=xx.unsqueeze(0)#升维度（1,seq_len,dim）

            xx=xx.permute(0,2,1)#输出维度是（1,dim,seq_len）
            pred=net(xx)#输出是（1,dim,pred_len）
            optimizer.zero_grad()
            pred=pred.contiguous().squeeze(0).permute(1,0)#输出是（pred_len,dim）
            loss=loss_fn(pred,x[seq_len:,:])#计算loss
            loss.backward()
            runing_loss+=loss
            optimizer.step()
        print('第{0}个epoch的loss：{1}'.format(epoch+1,round((runing_loss/(i+1)).item(),2)))


loss_fn1=nn.MSELoss()
#读数据
data=pd.read_csv('./datasets/ETT-small/ETTh1.csv')#读数据
data=data.drop(labels='date',axis=1)#把时间维扔掉
data=np.array(data)
data=torch.as_tensor(torch.from_numpy(data), dtype=torch.float32)
data=(data-torch.mean(data))/torch.std(data)#标准化

#—-----------------------------------------
#   划分数据集，0.9训练，0.1测试
#------------------------------------------
index=int(0.9*data.shape[0]) # 作为训练集和测试集的分界线
train_data=data[:index,:]#训练集
test_data=data[index:,:]#测试集

seq_len=96  # 已知的时间序列的长度
pred_len=336# 预测的时间序列的长度
sae=SAE(seq_len=seq_len,pred_len=pred_len)
optimizer=optim.Adam(sae.parameters(),lr=1e-3)
loss_fn=nn.MSELoss()
epoches=10
rou=0.005

pre_train(sae,train_data)
#预训练完毕，训练所有层
for param in sae.named_parameters():  # name_parameters()会返回层名字和权重
    param[1].requires_grad_(True)
epoches=50
train(net=sae,loss_fn=loss_fn,optimizer=optimizer,train_data=train_data,seq_len=seq_len,pred_len=pred_len,epoches=epoches)
save_files = {
    'hidden1.weight':sae.fc1.weight,
    'hidden1.bias':sae.fc1.bias,
    'hidden2.weight':sae.fc2.weight,
    'hidden2.bias':sae.fc2.bias
     }

path='SAE_encoder_96_48'
torch.save(save_files, path)



