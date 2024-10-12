import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_time_token

'''
We define a recurrent network that predicts the future values of a time-dependent variable based on
past inputs and covariates.
'''
class Deepar(nn.Module):
    def __init__(self,args,lstm_layers=3,dropout=0.2):
        super(Deepar, self).__init__()
        self.seq_len = args.seq_len # Known length of time series
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.d_feature = args.d_feature
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.d_mark = args.d_mark
        self.lstm_layers = lstm_layers
        self.dropout = dropout


        self.lstm = nn.LSTM(input_size=self.d_model,
                            hidden_size=self.d_ff,
                            num_layers=self.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=self.dropout)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        # input：(batch_size,hidden_size*num_layers)-->output：（batch_size,d_model）
        self.distribution_mu = nn.Linear(self.d_ff * self.lstm_layers, self.d_model)
        # input：(batch_size,hidden_size*num_layers)-->output：（batch_size,d_model）
        self.distribution_presigma = nn.Linear(self.d_ff * self.lstm_layers, self.d_model)
        # Use the softplus activation function to make sigma greater than 0.
        self.distribution_sigma = nn.Softplus()
        # input：(batch_size,d_model)-->output：(batch_size,d_feature)
        self.mu_outfc = nn.Linear(self.d_model,self.d_feature)
        # input：(batch_size,d_model)-->output：(batch_size,d_feature)
        self.sigma_outfc = nn.Linear(self.d_model, self.d_feature)
        # input：（batch_size,pred_len,d_model）-->output：（batch_size,pred_len,d_feature）
        self.pred_outfc = nn.Linear(self.d_model, self.d_feature)

        self.embedding = DataEmbedding_time_token(self.d_feature, self.d_mark, self.d_model)

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)



    def pred_onestep(self,x,hidden,cell):

        #x：（seq_len,batch_size,self.d_feature）-->output (seq_len,batch_size,hidden_size)
        # hidden and cell (num_layers,batch_size,hidden_size)
        output, (hidden, cell) = self.lstm(x, (hidden, cell)) # (96,32,64)-->(96,32,128)
        # use h from all three layers to calculate mu and sigma
        # hidden_permute (batch_size,hidden_size*num_layers)
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma) # Use the softplus activation function to make sigma greater than zero.

        gaussian = torch.distributions.normal.Normal(mu, sigma) # Construct a Gaussian distribution using mu and sigma
        pred = gaussian.sample()  # Predicted values are sampled from the Gaussian distribution just constructed
        # pred and mu and sigma are (batch_size,d_model)
        return pred,mu,sigma


    def forward(self, enc_x, enc_mark, y, y_mark,mode):
        loss = torch.zeros(1, device=enc_x.device)
        B = enc_x.shape[0] # batch_size
        x_embed = self.embedding(enc_x, enc_mark)
        y_embed = self.embedding(y,y_mark)

        #  (batch_size,pred_len,d_model)
        pred_zero = torch.zeros_like(y_embed[:, -self.pred_len:, :]).float()
        input_zero = torch.zeros_like(y_embed[:, -self.pred_len:, :]).float()

        x_cat_pred = torch.cat([x_embed[:, :self.seq_len, :], pred_zero], dim=1).float().to(enc_x.device) # 把初始化的预测值和原本的时序数据拼接起来，用来装预测值
        x_cat_input = torch.cat([x_embed[:, :self.seq_len, :], input_zero], dim=1).float().to(enc_x.device) # 打算在训练的时候每一次输入LSTM的数据都是真实的数据，而不是上一个时间步预测值

        hidden = torch.zeros(self.lstm_layers,B , self.d_ff, device=enc_x.device)
        cell = torch.zeros(self.lstm_layers,B, self.d_ff, device=enc_x.device)


        for i in range(self.pred_len):
            if mode == 'train':
                lstm_input = x_cat_input[:, i:i + self.seq_len, :].permute(1, 0, 2).clone()
            else:
                lstm_input = x_cat_pred[:, i:i + self.seq_len, :].permute(1,0,2).clone()
            pred,mu,sigma = self.pred_onestep(lstm_input ,hidden,cell)
            # input：（batch_size,d_model）-->(batch_size,d_feature)
            out_mu = self.mu_outfc(mu)
            out_sigma = self.distribution_sigma(self.sigma_outfc(sigma))
            loss += self.loss_fn(out_mu,out_sigma,y[:,self.label_len+i,:])
            x_cat_pred[:, self.seq_len + i, :] = pred
            x_cat_input[:, self.seq_len + i, :] = y_embed[:, self.label_len + i,:]

        return self.pred_outfc(x_cat_pred[:,-self.pred_len:,:]),loss

    def loss_fn(self,mu, sigma,labels): # Customise a loss function with negative logarithmic loss
        '''
        Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
        Args:
            mu: (Variable) dimension [batch_size] - estimated mean at time step t
            sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
            labels: (Variable) dimension [batch_size] z_t
        Returns:
            loss: (Variable) average log-likelihood loss across the batch
        '''
        distribution = torch.distributions.normal.Normal(mu, sigma) # Reconstructing a Gaussian distribution using mu,sigma
        likelihood = distribution.log_prob(labels) # Negative log likelihood
        return -torch.mean(likelihood)




