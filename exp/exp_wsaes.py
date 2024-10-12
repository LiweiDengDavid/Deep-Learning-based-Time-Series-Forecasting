import os
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import torch



class EXP_WSAES_LSTM:
    def __init__(self,args):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.pretrain = int(args.epoches*0.7)
        self.batch_size = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.lr = args.lr
        self.dim = args.d_mark
        self.args = args
        self.epoch = args.epoches

        self.train_gpu = [1,]
        self.devices = [0, ]

        self.model_name = 'WSAES_LSTM'
        self.data_name = args.data_name

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./checkpoint/'+self.model_name+'/'):
            os.makedirs('./checkpoint/'+self.model_name+'/')



        self.modelpath = './checkpoint/'+self.model_name+'/'+self.data_name+'_best_model.pkl'

        if self.args.data_name == 'ETTh1':
            self.data_path = './datasets/ETT-small/ETTh1.csv'

        if self.args.data_name == 'ETTm1':
            self.data_path = './datasets/ETT-small/ETTm1.csv'

        if self.args.data_name == 'illness':
            self.data_path = './datasets/illness/national_illness.csv'

        if self.args.data_name == 'electricity':
            self.data_path = './datasets/electricity/electricity.csv'

        if self.args.data_name == 'exchange':
            self.data_path = './datasets/exchange_rate/exchange_rate.csv'

        if self.args.data_name == 'traffic':
            self.data_path = './datasets/traffic/traffic.csv'



        self._get_data()
        self._get_model()

    def _get_data(self):

        train,valid,test,mean,scale,dim = get_data(self.data_path)

        self.mean = mean
        self.scale = scale
        self.args.data_dim = dim

        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size,shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size,shuffle=False)
        self.testloader = DataLoader(testset, batch_size=self.batch_size,shuffle=False)

        print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

        return


    def _get_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.train_gpu)
        ngpus_per_node = len(self.train_gpu)
        print('Number of devices: {}'.format(ngpus_per_node))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('------------Use of equipment---------------')
        print(self.device)

        self.model = WSAES_LSTM(self.args)

        if ngpus_per_node > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        if ngpus_per_node > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.devices)
            self.scheduler = nn.DataParallel(self.scheduler, device_ids=self.devices)
        self.early_stopping = EarlyStopping(optimizer=self.optimizer,scheduler=self.scheduler,patience=self.patience, path=self.modelpath,)
        self.criterion = nn.MSELoss()


        if self.args.resume:
            print('Loading pre-trained models')
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])


        return

    def _process_one_batch_WSAEs_LSTM(self, epoch,batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        prediction,sae_output = self.model(batch_x, batch_y,batch_x_mark, batch_y_mark,epoch)
        loss_MSE = self.criterion(prediction, batch_y[:, -self.pred_len:, :])
        return prediction,sae_output,loss_MSE


    def train(self):

        sae_loss_function = nn.MSELoss(reduction='sum')
        sae_optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []

            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):

                pred,sae_output,_ = self._process_one_batch_WSAEs_LSTM(epoch,batch_x, batch_y, batch_x_mark, batch_y_mark)
                if epoch < self.pretrain:
                    pred = pred.to(torch.float32)
                    batch_x = batch_x.to(torch.float32)

                    loss = sae_loss_function(sae_output,Wavelet(batch_x).to(self.device))
                    # 二范数正则化
                    lambd = torch.tensor(1).to(self.device)
                    l2_reg = torch.tensor(0.).to(self.device)

                    for param in self.model.parameters():
                        l2_reg += 0.5*torch.norm(param**2)
                    loss += lambd * l2_reg


                    sae_optimizer.zero_grad()
                    loss.backward()
                    sae_optimizer.step()
                    prediction_loss = self.criterion(pred,batch_y[:, -self.pred_len:, :].to(self.device))
                    train_loss.append(prediction_loss.item())
                else:
                    pred = pred.to(torch.float32)
                    result = batch_y[:, -self.pred_len:, :].to(self.device)
                    loss = self.criterion(pred.float(), result.float())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss.append(loss.item())


            if epoch >= self.pretrain:
                self.model.eval()
                valid_loss = []
                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                    pred,_,_ = self._process_one_batch_WSAEs_LSTM(epoch,batch_x, batch_y, batch_x_mark, batch_y_mark)
                    pred = pred.to(torch.float32)
                    result = batch_y[:, -self.pred_len:, :].to(torch.float32)
                    loss = self.criterion(pred.to(self.device), result.to(self.device))
                    valid_loss.append(loss.item())

                test_loss = []
                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                    pred,_,_ = self._process_one_batch_WSAEs_LSTM(epoch,batch_x, batch_y, batch_x_mark, batch_y_mark)
                    pred = pred.to(torch.float32)
                    result = batch_y[:, -self.pred_len:, :].to(torch.float32).to(self.device)
                    loss= self.criterion(pred, result)
                    test_loss.append(loss.item())

                train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(
                        test_loss)
                print(
                        "Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(epoch + 1, train_loss,
                                                                                                         valid_loss,
                                                                                                         test_loss))

                self.early_stopping(valid_loss, self.model, epoch)
                if self.early_stopping.early_stop:
                    break
                self.scheduler.step()

        checkpoint = torch.load(self.modelpath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # self.model.load_state_dict(torch.load(self.modelpath))


    def test(self):
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred, _,loss = self._process_one_batch_WSAEs_LSTM(self.epoch,batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:,  -self.pred_len:, :])

        mape_error = np.mean(self.mean)*0.1
        trues, preds = np.array(trues), np.array(preds)

        #反归一化：
        dstand_preds = preds*self.scale+self.mean
        dstand_trues = trues*self.scale+self.mean

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        mape = np.mean(np.abs(dstand_trues-dstand_preds)/(dstand_trues+mape_error))

        print('Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{0:.4f}'.format(mse, mae, mape))

        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_preds',preds)
        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_trues',trues)
        
        
        

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
        input = input.permute(0,2,1)
        prediction = torch.zeros(size=(input.shape[0],input.shape[1],self.pre_len)).to(self.device)

        for i in range(self.pre_len):
            output,(h,_) = self.lstm_layer(input[:,:,-self.seqlen:])
            output = self.lstm_fc(output)
            prediction[:,:,i] = output.squeeze(-1)
            a,b = input,output
            input = torch.cat((input,output),dim=-1)

        prediction = prediction.permute(0,2,1) # Change back to batchsize, prelen, dim

        return prediction

    def forward(self,batch_x, batch_y,batch_x_mark, batch_y_mark,epoch=1):
        without_noise_data = self.Wavelet_transform(batch_x).to(self.device)
        sae_output = self.SAE(epoch,without_noise_data)
        prediction = self.LSTM_PROCEED(sae_output)
        return prediction,sae_output


def Wavelet(data):
        # Since the wavelet transform packet doesn't seem to be able to operate on the tensor category,
        # here's the data converted to numpy, and then converted back later.
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

