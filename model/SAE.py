import torch
import torch.nn as nn



#Perform data compression, compressing the seq_len dimension of the data
class SAE_encoder(nn.Module):
    def __init__(self,seq_len=96,hidden_size1=72,hidden_size2=48):
        '''
        Arg:
        seq_len represents the second dimension of the input data (temporal dimension: how long a time series is known)
        hidden_size1=72 is the number of neurons in hidden layer 1, hidden_size2=48 is the number of neurons in hidden layer 2
        where hidden_size2 is the time dimension of the data compressed by SAE_encoder
        '''
        super(SAE_encoder, self).__init__()
        # Input is (batch_size,in_channels,seq_len)-->output: (batch_size,in_channels,hidden_size1)
        # where in_channels represents the feature dimension of the data, in this task it is 7.
        self.hidden1=nn.Linear(seq_len,hidden_size1)
        # input is (batch_size,in_channels,hidden_size1) --> output: (batch_size,in_channels,hidden_size2)
        self.hidden2=nn.Linear(hidden_size1,hidden_size2)
    def forward(self,x):
        x=x.permute(0,2,1) # Transpose the input data x, dealing with the seq_len dimension
        x=torch.relu(self.hidden1(x)) # output：(batch_size,in_channels,hidden_size1)
        x=self.hidden2(x) # output：（batch_size,in_channels,hidden_size2）
        return x


class LSTM(nn.Module):# Input data dimensions (seq_len, batch_size,dim), because of the use of SAE for data compression, the time dimension from seq_len to hidden_size2
    # ------------------------------------------------------------------------------------------------
    # Input data dimensions (hidden_size2,batch_size,dim))
    # x(hidden_size2,batch_size,dim), for later seq_len=hidden_size2
    # hidden_size(num_layers,batch_size,hidden_size)=(1,8,128)
    # cell(num_layers,batch_size,hidden_size)=(1,8,128)
    # output(seq_len,batch_size,hidden_size)=(96,1,128)
    # known 96 days through LSTM to get output compressed to 1 day, then spell back the resulting prediction and slide it to the next 96 days
    # ------------------------------------------------------------------------------------------
    def __init__(self,seq_len,pred_len,dim,batch_size,hidden_size=128,num_layers=1,):
        '''
        Args:
        :param seq_len: denotes the known time dimension, which should be the same as the hidden_size2 of SAE_encoder.
        :param pred_len: The length of the predicted time series.
        :param dim: denotes the feature dimension of the data.
        :param hidden_size: the number of cells in the hidden layer
        :param num_layers: how many layers of the lstm network are used
        :param batch_size: batch_size of the time series
        '''
        super(LSTM, self).__init__()
        self.seq_len=seq_len # seq_len is the length after data compression
        self.dim=dim
        self.pred_len=pred_len
        self.num_layers=num_layers
        self.hidden_sie=hidden_size
        self.batch_size=batch_size
        self.total_len=seq_len+pred_len # Total time series for one mission
        # Input is (seq_len,batch_size,dim) --> Output is (seq_len,batch_size,hidden_size)
        # where seq_len should be SAE_encoder's hidden_size2
        self.lstm=nn.LSTM(input_size=dim,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=False) # Initialise an LSTM network
        # input：（seq_len,batch_size,hidden_size)-->output：（seq_len,batch_size,dim）
        self.fc_dim=nn.Linear(hidden_size,dim)#Compression of data feature dimensions
        #  input：（dim,batch_size,seq_len）-->(dim,batch_size,1)
        self.fc_time=nn.Linear(seq_len,1)#Compression time dimension

    # Each time only the next day's time series is predicted
    def pred_onestep(self,x,hidden,cell): # input：（seq_len,batch_size,dim）
        output,(hidden,cell)=self.lstm(x,(hidden,cell))#output：（seq_len,batch_size,hidden_size）
        output=self.fc_dim(output)#output：（seq_len,batch_size，dim）
        output=output.permute(2,1,0)#--》(dim,batch_size,seq_len),In order to operate on the time dimension
        output=self.fc_time(output)#output：（dim,batch_size,1）
        output=output.permute(2,1,0)#--》（1,batch_size,dim），1 indicates a point in time
        return output # （1,batch_size,dim）

    # -------------------------------------------------------------------------
    # Steps:
    # (1) Each time use pred_onestep to predict only the next day's time series for the currently known time series
    # (2) Then get the result in splicing back into the known time series, and then slide the window to get the new known time series
    # (3) Loop (1)~(2) until you get the complete time series to be predicted, then stop the loop and return the predicted time series.
    # ------------------------------------------------------------------------
    def forward(self,x):
        hidden=torch.zeros(self.num_layers,self.batch_size,self.hidden_sie).to(x.device) # hidden_size=(num_layers,batch_size,hidden_size)
        cell=torch.zeros(self.num_layers,self.batch_size,self.hidden_sie).to(x.device) # Initialise hidden and cell
        # Initialise x_cat_pred Used to load known and predicted time series
        x_cat_pred=torch.zeros(self.total_len,self.batch_size,self.dim).to(x.device) # Put the prediction part together too x_cat_pred=(seq_len+pred_len,batch_size,dim)
        # Put the known time series into x_cat_pred
        x_cat_pred[:self.seq_len,:,:]=x_cat_pred[:self.seq_len,:,:].clone()+x # Put the known sequence data into x_cat_pred as well

        for i in range(self.pred_len): # Slide as many times as it takes to predict a time series of any length
            lstm_input=x_cat_pred[i:i+self.seq_len,:,:].clone()#Get each input lstm_input (seq_len,batch_size,dim)
            pred=self.pred_onestep(lstm_input,hidden,cell)#pred （1，batch_szie，dim）
            x_cat_pred[i+self.seq_len,:,:]=x_cat_pred[i+self.seq_len,:,:].clone().unsqueeze(0)+pred#Splice the predictions back in.
        return x_cat_pred[-self.pred_len:,:,:]#Returning the predicted in back, the output is (pred_len,batch_size,dim)



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
        x=x.permute(2,0,1)#output：（seq_len,batch_size,dim）
        x=self.LSTM(x)#输出是（pred_len,batch_size,dim）
        x=x.permute(1,0,2)#输出是（batch_size,pred_len,dim）
        return x


