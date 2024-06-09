import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR



# A greedy algorithm is used, updating the parameters of only one layer at each training session and fixing the parameters of all other layers

class SAE(nn.Module): # The input is a known time series and the output is a predicted time series
    def __init__(self,seq_len,pred_len,hidden_size=300):
        super(SAE, self).__init__()
        # input(in_channels,seq_len) --> output as (in_channels,hidden_size)
        hidden_size1=seq_len*2//3
        hidden_size2=seq_len//2
        self.fc1=nn.Linear(seq_len,hidden_size1)
        # (in_channels,hidden_size-->(in_channels,hidden_size)
        self.fc2=nn.Linear(hidden_size1,hidden_size2)
        # (in_channels,hidden_size-->(in_channels,hidden_size)
        self.fc3=nn.Linear(hidden_size2,hidden_size)
        #(in_channels,hidden_size-->(in_channels,hidden_size)
        self.fc4=nn.Linear(hidden_size,hidden_size)
        # (in_channels,hidden_size-->(in_channels,pred_len)
        self.fc5=nn.Linear(hidden_size,pred_len)
    def forward(self,x):
        x=torch.sigmoid(self.fc1(x))
        x=torch.sigmoid(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))
        x=torch.sigmoid(self.fc4(x))
        x=self.fc5(x)
        return x



# Calculate the average activation, which is the average of the values after going through the current layer,
# and is used when calculating the KL dispersion
def rou_hat_cala(i,self,xx):
    if i == 0:
        pred = torch.sigmoid(self.fc1(xx))
        rou_hat1 = torch.mean(pred) + 1e-5  # It's needed to calculate the loss, plus a very small number to prevent it from going to zero.
    elif i == 2:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        rou_hat1 = torch.mean(pred) + 1e-5
    elif i == 4:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        pred = torch.sigmoid(self.fc3(pred))
        rou_hat1 = torch.mean(pred) + 1e-5
    elif i == 6:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        pred = torch.sigmoid(self.fc3(pred))
        pred = torch.sigmoid(self.fc4(pred))
        rou_hat1 = torch.mean(pred) + 1e-5

    elif i == 8:
        pred = torch.sigmoid(self.fc1(xx))
        pred = torch.sigmoid(self.fc2(pred))
        pred = torch.sigmoid(self.fc3(pred))
        pred = torch.sigmoid(self.fc4(pred))
        pred = torch.sigmoid(self.fc5(pred))
        rou_hat1 = torch.mean(pred) + 1e-5
    else:rou_hat1=0
    return rou_hat1




#Pre-train, freeze all other layers that are not trained (requires_grad=Fasle)
def pre_train(self,train_data):
    rou_hat = 0
    param_lst=[] #Load up the names of each layer's weights
    #Set all trainable parameters all to False
    for param in self.named_parameters(): #name_parameters() returns layer names and weights
        param[1].requires_grad=False
        param_lst.append(param[0])

    for i in range(len(param_lst)):
        lst=list(self.named_parameters())# Get network weights and names
        if i%2==0:
            lst[i][1].requires_grad=True
            lst[i+1][1].requires_grad=True # Layer by layer training

            total_len= pred_len + seq_len

            #The data from the training set is passed through the network once,
            # and then the average activation after passing through the hidden layer is calculated in the
            for j in range(train_data.shape[0] // total_len):  # How many total_len can be taken?
                x = train_data[j * total_len:(j + 1) * total_len, :]  # Take the total length each time
                xx = x[:seq_len, :].clone()  # Each known time series (seq_len,dim)
                xx = xx.unsqueeze(0)  # Ascending dimension (1,seq_len,dim)
                xx = xx.permute(0, 2, 1)  # The output dimension is (1,dim,seq_len)

                #Calculate rou_hat, average activation
                rou_hat+=rou_hat_cala(i,self,xx)
            rou_hat=rou_hat/(j+1)+1e-5
            for epoch in range(epoches):
                runing_loss = 0
                for j in range(train_data.shape[0] // total_len):
                    x = train_data[i * total_len :(i + 1) * total_len, :]
                    xx = x[:seq_len, :].clone()  # Each known time series (seq_len,dim)
                    xx = xx.unsqueeze(0)  # Ascending dimension (1,seq_len,dim)
                    xx = xx.permute(0, 2, 1)  # The output dimension is (1,dim,seq_len)
                    pred = sae(xx)  # The output is (1,dim,seq_len)
                    optimizer.zero_grad()
                    pred = pred.squeeze()
                    pred=pred.permute(1,0)#The output is (seq-len,dim)
                    kl=rou*torch.log(rou/rou_hat)+(1-rou)*torch.log((1-rou)/(1-rou_hat)) #Calculate the KL scatter
                    loss = loss_fn(pred, x[seq_len:, :])+kl
                    loss.backward()
                    runing_loss += loss
                    optimizer.step()
                    rou_hat = rou_hat_cala(i, self, xx)
                print('Loss at the {0}th epoch: {1}'.format(epoch + 1, round((runing_loss / (i + 1)).item(), 2)))
            lst[i][1].requires_grad = False
            lst[i + 1][1].requires_grad = False # Freeze the trained layer again



def train(net,loss_fn,optimizer,train_data,seq_len,pred_len,epoches):
    '''
    Args:
        net is a network that needs to be trained
        loss_fn is the loss function used and train_data is the entire training set
        seq_len is the length of the known time series, pred_len is the length of the time series to be predicted
        epoches is how many times the epoches are out of the loop.
    '''
    net.train()
    total_len=pred_len+seq_len
    for epoch in range(epoches):
        runing_loss = 0
        for i in range(train_data.shape[0] // total_len):
            x=train_data[i*total_len:(i+1)*total_len,:]
            xx=x[:seq_len,:].clone()
            xx=xx.unsqueeze(0)

            xx=xx.permute(0,2,1)
            pred=net(xx)
            optimizer.zero_grad()
            pred=pred.contiguous().squeeze(0).permute(1,0)
            loss=loss_fn(pred,x[seq_len:,:])
            loss.backward()
            runing_loss+=loss
            optimizer.step()
        print('Loss at the {0}th epoch: {1}'.format(epoch+1,round((runing_loss/(i+1)).item(),2)))


loss_fn1=nn.MSELoss()
#read data
data=pd.read_csv('./datasets/ETT-small/ETTh1.csv')
data=data.drop(labels='date',axis=1)
data=np.array(data)
data=torch.as_tensor(torch.from_numpy(data), dtype=torch.float32)
data=(data-torch.mean(data))/torch.std(data)

#—-----------------------------------------
#   Divide dataset, 0.9 training, 0.1 testing
#------------------------------------------
index=int(0.9*data.shape[0]) # is the dividing line between the training set and the test set
train_data=data[:index,:]
test_data=data[index:,:]

seq_len=96  # The length of the known time series
pred_len=336# Length of predicted time series
sae=SAE(seq_len=seq_len,pred_len=pred_len)
optimizer=optim.Adam(sae.parameters(),lr=1e-3)
loss_fn=nn.MSELoss()
epoches=10
rou=0.005

pre_train(sae,train_data)
#Pre-training complete, train all layers
for param in sae.named_parameters():  # name_parameters() will return the layer names and weights
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



