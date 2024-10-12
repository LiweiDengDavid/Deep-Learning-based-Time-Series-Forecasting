import torch
import torch.nn as nn
import numpy as np


class Deep_states_model(nn.Module):
    def __init__(self,seqlen,prelen,lstm_hidden_dimension,space_hidden_dimension,stock_number=10,device='cpu'):
        super(Deep_states_model, self).__init__()

        self.seqlen = seqlen
        self.prelen = prelen
        self.stock_number = stock_number
        self.lstm_hidden_dimension = lstm_hidden_dimension
        self.space_hidden_dimension = space_hidden_dimension
        self.device = device
        self.lstm = nn.LSTM(input_size = 4,hidden_size = lstm_hidden_dimension,
                            num_layers=3,batch_first = True,dropout = 0.1,)

        self.F_linear = nn.Linear(self.lstm_hidden_dimension,self.space_hidden_dimension * self.space_hidden_dimension*self.stock_number)
        self.a_linear = nn.Linear(self.lstm_hidden_dimension,1*self.space_hidden_dimension*self.stock_number)
        self.b_linear = nn.Linear(self.lstm_hidden_dimension,1*1*self.stock_number)
        self.g_linear = nn.Linear(self.lstm_hidden_dimension,self.space_hidden_dimension*1*self.stock_number)
        self.sigmoid_linear = nn.Linear(self.lstm_hidden_dimension,1*1*self.stock_number)
        self.mu0_linear = nn.Linear(self.lstm_hidden_dimension,self.space_hidden_dimension*1*self.stock_number)
        self.sigma0_linear = nn.Linear(self.lstm_hidden_dimension, self.space_hidden_dimension*1*self.stock_number)




    def lstm_proceed(self,input_feature):

        # input_feature = input_feature.permute(0,2,1).contiguous()
        # input_feature = input_feature.view(-1,self.seqlen,1)

        output,(_,_) = self.lstm(input_feature)

        return output


    def state_space_proceed(self,input):
        # input shape:batchsize,lstm_hidden_dimension,dim
        batchsize,time_step = input.shape[0],input.shape[1]
        F = self.F_linear(input).view(batchsize,time_step,self.space_hidden_dimension , self.space_hidden_dimension,self.stock_number).permute(0,1,4,2,3)
        a = self.a_linear(input).view(batchsize,time_step,1 , self.space_hidden_dimension,self.stock_number).permute(0,1,4,2,3)
        b = self.b_linear(input).view(batchsize,time_step,1 , 1,self.stock_number).permute(0,1,4,2,3)
        g = self.g_linear(input).view(batchsize,time_step,self.space_hidden_dimension,1,self.stock_number).permute(0,1,4,2,3)
        sigmoid = self.sigmoid_linear(input).view(batchsize,time_step,1,1,self.stock_number).permute(0,1,4,2,3)
        l = self.mu0_linear(input[:,0,:].unsqueeze(-2)).view(batchsize,1,self.space_hidden_dimension,1,self.stock_number).permute(0,1,4,2,3)
        p = self.sigma0_linear(input[:,0,:].unsqueeze(-2)).view(batchsize,1,self.space_hidden_dimension,1,self.stock_number).permute(0,1,4,2,3)


        return (F,a,b,g,sigmoid,l,p)



    def step_forward(self,para,time_step,observation=None):
        # history_price shape: batchsize,seqlen,dim
        (F, a, b, g, sigmoid, l, p) = para
        pre_total = torch.zeros(size=(F.shape[0],time_step,F.shape[2]))
        log_prob_total = torch.zeros(size=(F.shape[0],time_step,F.shape[2]))

        for t in range(time_step):
            F_one,a_one,b_one,g_one,sigmoid_one =F[:,t,:].unsqueeze(1),a[:,t,:].unsqueeze(1),b[:,t,:].unsqueeze(1),g[:,t,:].unsqueeze(1),sigmoid[:,t,:].unsqueeze(1)
            l = torch.matmul(F_one,l)
            p = torch.matmul(torch.matmul(F_one,p),p.permute(0,1,2,4,3)) + torch.matmul(g_one,g_one.permute(0,1,2,4,3))
            z_pred = torch.matmul(a_one,l) + b_one
            s_one = torch.abs(torch.matmul(torch.matmul(a_one,p),a_one.permute(0,1,2,4,3)) + torch.matmul(sigmoid_one,sigmoid_one.permute(0,1,2,4,3)))
            s_one = torch.where(torch.isnan(s_one), torch.full_like(s_one, 1e-3), s_one)
            z_pred = torch.where(torch.isnan(z_pred), torch.full_like(z_pred, 1e-3), z_pred)

            s_one = s_one.squeeze(-1).squeeze(-1)
            z_pred = z_pred.squeeze(-1).squeeze(-1)
            if observation is not None:
                z = observation[:,t,:].unsqueeze(1)
                try:
                    log_prob = torch.distributions.Normal(z_pred, s_one).log_prob(z)
                except:
                    e,f,ff = z_pred, s_one,z
                log_prob = log_prob.unsqueeze(-1).unsqueeze(-1)
                log_prob_total[:, t] = log_prob.squeeze()
            else:
                z = torch.distributions.Normal(z_pred, s_one).sample()
            z = z.unsqueeze(-1).unsqueeze(-1)
            s_one = s_one.unsqueeze(-1).unsqueeze(-1)
            z_pred = z_pred.unsqueeze(-1).unsqueeze(-1)
            k_one = torch.matmul(torch.matmul(p,a_one.permute(0,1,2,4,3)),torch.linalg.inv(s_one)) # !!!
            y = z - z_pred
            l = l + torch.matmul(k_one,y)
            p = p - torch.matmul(torch.matmul(k_one,a_one),p)

            pre_total[:,t] = z_pred.squeeze()

        if observation is None:
            return pre_total
        else:
            return (F, a, b, g, sigmoid, l, p),log_prob





    def trains(self,input_feature_history, input_history_price):

        lstm_output_total = self.lstm_proceed(input_feature_history)
        para = self.state_space_proceed(lstm_output_total)
        para,loss = self.step_forward(para,self.seqlen,input_history_price)

        return loss

    def prediction(self,input_feature_future):
        lstm_output_total = self.lstm_proceed(input_feature_future)
        para = self.state_space_proceed(lstm_output_total)
        prediction = self.step_forward(para,self.prelen)

        return prediction.to(self.device)






class Deep_states(nn.Module):
    def __init__(self,args):
        super(Deep_states, self).__init__()

        self.seqlen = args.seq_len
        self.prelen = args.pred_len
        self.stock_number = args.d_feature
        self.lstm_hidden_dimension = min(int(args.d_dimension/2),32)
        self.space_hidden_dimension = min(args.d_dimension,8)
        self.device = args.device
        self.model_main = Deep_states_model(seqlen=self.seqlen,
                                   prelen=self.prelen,
                                   lstm_hidden_dimension=self.lstm_hidden_dimension,
                                   space_hidden_dimension=self.space_hidden_dimension,
                                   stock_number=self.stock_number,device=self.device)

        self.optimizer = torch.optim.SGD(self.model_main.parameters(),lr=5e-4,momentum=0.9)

    def forward(self,batch_x, batch_x_mark,batch_y, batch_y_mark):

        input_feature_history, input_history_price, input_feature_future = batch_x_mark,batch_x,batch_y_mark[:,-self.prelen:,:]
        loss = self.model_main.trains(input_feature_history, input_history_price)
        try:
            self.optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            self.optimizer.step()
        except:
            pass
        prediction = self.model_main.prediction(input_feature_future)

        return prediction.squeeze(-1).squeeze(-1)



