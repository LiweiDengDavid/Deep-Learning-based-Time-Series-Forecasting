import torch
import torch.nn as nn


# -------------------------------------------------------------------
# Take the seq_len columns one at a time, sliding one column at a time.
# Input model is (batch_size,seq_len,dim), if you want to predict days after pred_len
# You can now initialise all the positions in the days after pred_len to 0, and then once you scroll, fill in the next value.
# Until you get to the end, then slice off the back pred_len, then calculate the loss with the target value.
# -------------------------------------------------------------------

class CNN_1D(nn.Module):
    def __init__(self, args, conv_kernel_size=33, conv_stride=1, out_channels1=32, out_channels2=64, out_channels3=128,
                 pool_kernel_size=4, padding=16):
        '''
        Args:
        conv_kernel_size=33 is the size of the convolution kernel, the three convolution kernels used here are all of the same size, conv_stride=1 represents the step size of the convolution kernel for a single swipe
        out_channels1=32 is the output dimension after the first convolution, out_channels2=64 is the output dimension after the second convolution, out_channels3=128 is the output dimension after the third convolution.
        padding is the number of zeros around the convolution, in order to make the convolution of the second dimension of the data, i.e., the time dimension does not change.
        padding=(conv_kernel_size-1)/2, (when string is 1)
        pool_kernel_size=4 is the kernel size of maxPool1d, and the step size of maxPooling is the same as the pooling kernel size by default.
        self.in_channels=7 represents the first dimension of the input data (feature dimension); self.seq_len=96 represents the second dimension of the input data (time dimension: how long the time series is known)
        '''

        self.seq_len = args.seq_len  
        self.pred_len = args.pred_len  
        self.in_channels = args.d_feature 

        super(CNN_1D, self).__init__()
        # （batch_size,self.in_channels，seq_len）--》（batch_size,out_channels1,seq_len）
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=out_channels1, kernel_size=conv_kernel_size,
                               padding=padding, stride=conv_stride)
        # input（batch_size,out_channels1,seq_len） output（batch_size,out_channels2,seq_len）
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=conv_kernel_size,
                               padding=padding, stride=conv_stride)
        # input（batch_size,out_channels2,seq_len），output（batch_size,out_channels3，seq_len）
        self.conv3 = nn.Conv1d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=conv_kernel_size,
                               padding=padding, stride=conv_stride)
        # input（batch_size,out_channels3，seq_len）,output（batch_size,out_channels3,seq_len//pool_kernel_size）
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        # input（batch_size,out_channels3,seq_len//pool_kernel_size），output(batch_size,out_channels3, 1）
        self.fc1 = nn.Linear(self.seq_len // pool_kernel_size, 1)
        # input (batch_size,1,out_channels3) output（batch_size,1,out_channels2） 
        self.fc2 = nn.Linear(out_channels3, out_channels2)
        # input（batch_size,1, out_channels2）-->output（batch_size,1，in_channels） 
        self.fc3 = nn.Linear(out_channels2, self.in_channels)

    def pred_onestep(self, input_x):
        # input_x [batch,seq_len,dim]
        x = self.conv1(input_x.permute(0, 2, 1))  #（batch_size,out_channels1,seq_len）
        x = torch.relu(x)
        x = self.conv2(x)  # （batch_size,out_channels2,seq_len）
        x = torch.relu(x)
        x = torch.relu(self.conv3(x))  # （batch_size,out_channels3，seq_len）
        x = self.pool(x)  # （batch_size,out_channels3,seq_len//pool_kernel_size）
        x = self.fc1(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.fc2(x))  # output（batch_size,1,out_channels2）
        x = self.fc3(x)  # output（batch_size,1，in_channels）

        return x

    def forward(self, enc_x, enc_mark, y, y_mark):
        '''
        :param enc_x: known time series (batch_size,seq_len,dim)
        The following params are not used in this model and will not be introduced in detail
        :param enc_mark: the time matrix corresponding to the time of the known time series.
        :param y.
        :param y_mark.
        :return: x_cat_pred[:,-self.pred_len:,:] return the part of the predicted time series back (batch_size,pred)len,dim)
        '''
        pred_zero = torch.zeros_like(y[:, -self.pred_len:, :]).float()  # Initialising predicted results,shape(batch_size,pred_len,dim)
        # Splicing together known time series and pred，
        x_cat_pred = torch.cat([enc_x, pred_zero], dim=1).float().to(
            enc_x.device)  # shape(batch_size , seq_len+pred_len , dim)

        for i in range(self.pred_len):  # The length of the time series predicted by the loop, since only one day at a time is predicted for the
            input_x = x_cat_pred[:, i:i + self.seq_len, :].clone()  # The length of the time series predicted by the loop, since only one day at a time is predicted for the
            pred = self.pred_onestep(input_x)  #shape(batch_size,1,dim)
            x_cat_pred[:, self.seq_len + i, :] = x_cat_pred[:, self.seq_len + i, :].clone() + pred.squeeze(
                1)  # Put this time's prediction into x_cat_pred

        return x_cat_pred[:, -self.pred_len:, :]  # (batch_size,pred_len,dim)
