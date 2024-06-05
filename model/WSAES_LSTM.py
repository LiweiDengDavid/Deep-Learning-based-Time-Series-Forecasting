from torch import nn
import torch
import pywt
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
class WSAES_LSTM(nn.Module):
    def __init__(self,args):
        super(WSAES_LSTM, self).__init__()
        self.seqlen = args.seq_len
        self.d_feature = args.d_feature
        self.hidden_dimension_sae = args.d_dimension
        self.hidden_dimension_lstm = args.d_dimension
        self.pre_len = args.pred_len
        self.total_pre_train_epoch = int(args.epoches*0.7)
        self.dropout = args.dropout
        self.device = args.device

        self.sae1_become_hidden = nn.Sequential(
            nn.Linear(self.d_feature,self.hidden_dimension_sae,bias=True),
            nn.Sigmoid()
        )
        self.sae1_become_original = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.d_feature, bias=True),
            nn.ReLU(inplace=True)
        )
        self.sae2_become_hidden = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.hidden_dimension_sae, bias=True),
            nn.Sigmoid(),
            nn.Dropout(p=self.dropout)
        )
        self.sae2_become_original = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.d_feature, bias=True),
            nn.Sigmoid(),
        )
        self.sae3_become_hidden = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.hidden_dimension_sae, bias=True),
            nn.Sigmoid(),
            nn.Dropout(p=self.dropout)
        )
        self.sae3_become_original = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.d_feature, bias=True),
            nn.Sigmoid(),
        )
        self.sae4_become_hidden = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.hidden_dimension_sae, bias=True),
            nn.Sigmoid(),
            nn.Dropout(p=0.2)
        )
        self.sae4_become_original = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.d_feature, bias=True),
            nn.Sigmoid(),
        )
        self.sae5_become_hidden = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.hidden_dimension_sae, bias=True),
            nn.Sigmoid(),
            nn.Dropout(p=self.dropout)
        )
        self.sae5_become_original = nn.Sequential(
            nn.Linear(self.hidden_dimension_sae, self.d_feature, bias=True),
            nn.Sigmoid(),
        )


        self.lstm_layer = torch.nn.LSTM(input_size=self.seqlen, hidden_size=int(self.seqlen/10),
                                        num_layers=5,batch_first=True, dropout=self.dropout)
        self.lstm_fc = nn.Sequential(
            nn.Linear(int(self.seqlen/10),1),
            nn.ReLU()
        )
    def Wavelet_transform(self,data):
        # 由于小波变换的包好像不能对tensor类别操作，这里将数据转成numpy，后面再转回来
        data = data.permute(0,2,1)
        input_data_length = data.shape[-1]
        batch_size = data.shape[0]
        data.reshape(-1,input_data_length)
        data = np.array(data.cpu())
        wavename = 'haar'
        cA, cD = pywt.dwt(data, wavename)
        ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
        yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component

        ya = ya.reshape(batch_size,-1,input_data_length)
        ya = torch.tensor(ya)
        data_without_noise = ya.permute(0,2,1)

        return data_without_noise

    def SAE(self,epoch,input):
        # input shape batchsize，seq——len，dim，通过sae（两次全连接，重构输入）
        # 论文中提出，sae的训练是训练了一层后才开始训练第二层，
        # 为了便于训练，我把epoch作为参数传入，总的epoches拆成5等分，
        # 第一等分主要是训练第一个sae，训练完后训练第二个sae，
        # 注意，第二个sae的输入是一个sae的隐藏层！！！！！，后面依次类推
        # 全连接对stock——number维度操作！！！！
        if epoch < int(self.total_pre_train_epoch/5 * 1):
            output = self.sae1_become_hidden(input)
            output = self.sae1_become_original(output)

        elif epoch < int(self.total_pre_train_epoch/5 * 2):
            output = self.sae1_become_hidden(input)
            output = self.sae2_become_hidden(output)
            output = self.sae2_become_original(output)
        elif epoch < int(self.total_pre_train_epoch/5 * 3):
            output = self.sae1_become_hidden(input)
            output = self.sae2_become_hidden(output)
            output = self.sae3_become_hidden(output)
            output = self.sae3_become_original(output)
        elif epoch < int(self.total_pre_train_epoch/ 5 * 4):
            output = self.sae1_become_hidden(input)
            output = self.sae2_become_hidden(output)
            output = self.sae3_become_hidden(output)
            output = self.sae4_become_hidden(output)
            output = self.sae4_become_original(output)
        elif epoch < int(self.total_pre_train_epoch/ 5 * 5):
            output = self.sae1_become_hidden(input)
            output = self.sae2_become_hidden(output)
            output = self.sae3_become_hidden(output)
            output = self.sae4_become_hidden(output)
            output = self.sae5_become_hidden(output)
            output = self.sae5_become_original(output)
        else:
            output = self.sae1_become_hidden(input)
            output = self.sae2_become_hidden(output)
            output = self.sae3_become_hidden(output)
            output = self.sae4_become_hidden(output)
            output = self.sae5_become_hidden(output)
            output = self.sae5_become_original(output)
        return output

    def LSTM_PROCEED(self,input):
        # input shape batchsize,seq_len,dim
        # 这里使用多步滚动预测，即用seqlen预测第seqlen+1天，然后1---（seqlen+1）共seqlen天去预测第seqlen+2天
        input = input.permute(0,2,1)
        # lstm的输入维度；batchsize，dim，seq——len，对seq-len操作
        prediction = torch.zeros(size=(input.shape[0],input.shape[1],self.pre_len)).to(self.device)

        for i in range(self.pre_len):
            output,(h,_) = self.lstm_layer(input[:,:,-self.seqlen:])
            output = self.lstm_fc(output)
            prediction[:,:,i] = output.squeeze(-1)
            a,b = input,output
            input = torch.cat((input,output),dim=-1)

        prediction = prediction.permute(0,2,1) # 变回batchsize，prelen，dim

        return prediction

    def forward(self,batch_x, batch_y,batch_x_mark, batch_y_mark,epoch=1):
        without_noise_data = self.Wavelet_transform(batch_x).to(self.device)
        sae_output = self.SAE(epoch,without_noise_data)
        prediction = self.LSTM_PROCEED(sae_output)
        return prediction,sae_output


def Wavelet(data):
        # 由于小波变换的包好像不能对tensor类别操作，这里将数据转成numpy，后面再转回来
    data = data.permute(0,2,1)
    input_data_length = data.shape[-1]
    batch_size = data.shape[0]
    data.reshape(-1,input_data_length)
    data = np.array(data)
    wavename = 'haar'
    cA, cD = pywt.dwt(data, wavename)
    ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
    yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component

    ya = ya.reshape(batch_size,-1,input_data_length)
    ya = torch.tensor(ya)
    data_without_noise = ya.permute(0,2,1)

    return data_without_noise