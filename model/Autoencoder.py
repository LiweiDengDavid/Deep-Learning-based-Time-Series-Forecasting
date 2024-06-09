import torch
import torch.nn as nn


# Use stacked Autoender to form a SAE network and then output the predictions directly
# Operate on the time dimension because the paper is saying that it wants to remember the time series first and then reason about it.
# So it's operating on the time dimension
class Autoencoder(nn.Module):
    def __init__(self,args,hidden_size=300):
        '''
        Arg:
        seq_len represents the second dimension of the input data (time dimension: how long the time series is known to be)
        pred_len represents how long the prediction time is, hidden_size=300 is the number of neurons in the hidden layer.
        '''
        super(Autoencoder, self).__init__()
        seq_len=args.seq_len
        pred_len=args.pred_len
        # （batch_size,dim,seq_len）-->（batch_size,dim,hidden_size）
        self.fc1=nn.Linear(seq_len,hidden_size)
        # （batch_size,dim,hidden_size）-->（batch_size,dim,hidden_size）
        self.fc2=nn.Linear(hidden_size,hidden_size)
        # （batch_size,dim,hidden_size）-->（batch_size,dim,hidden_size）
        self.fc3=nn.Linear(hidden_size,hidden_size)
       # （batch_size,dim,hidden_size）-->（batch_size,dim,hidden_size）
        self.fc4=nn.Linear(hidden_size,hidden_size)
        # （batch_size,dim,hidden_size）-->（batch_size,dim,pred_len）
        self.fc5=nn.Linear(hidden_size,pred_len)
    def forward(self, enc_x, enc_mark, y, y_mark):

        enc_x=enc_x.permute(0,2,1) # （batch_size,seq_len,dim）--》（batch_size,dim,seq_len）
        x=torch.sigmoid(self.fc1(enc_x)) # 输入是（batch_size,dim,seq_len）-->输出为（batch_size,dim,hidden_size）
        x=torch.sigmoid(self.fc2(x)) # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        x=torch.sigmoid(self.fc3(x)) # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        x=torch.sigmoid(self.fc4(x)) # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,hidden_size）
        x=self.fc5(x)  # 输入是（batch_size,dim,hidden_size）-->输出为（batch_size,dim,pred_len）
        x=x.permute(0,2,1)#输入是（batch_size,dim,pred_len）输出是（batch_size,pred_len,dim）
        return x #（batch_size，pred_len，dim）


