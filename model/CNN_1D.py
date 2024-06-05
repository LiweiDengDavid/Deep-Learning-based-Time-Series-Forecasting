

'''
Author:邓力玮
Date:2022.12.29
'''
import torch
import torch.nn as nn


# -------------------------------------------------------------------
#   一次取seq_len列，每一次滑动一列
#   输入模型的是（batch_size,seq_len,dim）,如果想要预测后pred_len天
#   可以现在后pred_len天的位置全部初始化为0，那么滚动一次，就填上后面的值
#   直至滑动到最后，那么就把后面的pred_len切分出来，然后在和目标值计算loss
# -------------------------------------------------------------------

class CNN_1D(nn.Module):
    def __init__(self, args, conv_kernel_size=33, conv_stride=1, out_channels1=32, out_channels2=64, out_channels3=128,
                 pool_kernel_size=4, padding=16):
        '''
        Args:
        conv_kernel_size=33是卷积核的大小，这里使用的三个卷积核大小都是一样的，conv_stride=1代表卷积核一次滑动的步长
        out_channels1=32是第一个卷积后的输出维度，out_channels2=64是第二个卷积后的输出维度，out_channels3=128是第三个卷积后的输出维度
        padding是卷积时候周围补零的数目，为了使的卷积出来的数据第二个维度即时间维度不改变，
        padding=(conv_kernel_size-1)/2，（当string为1时）
        pool_kernel_size=4是maxPool1d的核大小，并且maxPooling的步长默认和pooling核大小一样，
        self.in_channels=7代表输入数据的第一个维度（特征维度）；self.seq_len=96代表输入数据的第二个维度（时间维度：已知多长的时间序列）
        '''

        self.seq_len = args.seq_len  # seq_len 已知的时间序列的长度
        self.pred_len = args.pred_len  # pred_len 是预测的时间序列的长度
        self.in_channels = args.d_feature  # in_channels表示输入模型的维度

        super(CNN_1D, self).__init__()
        # 输入是（batch_size,self.in_channels，seq_len），输出是（batch_size,out_channels1,seq_len）,扩充了特征维度
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=out_channels1, kernel_size=conv_kernel_size,
                               padding=padding, stride=conv_stride)
        # 输入是（batch_size,out_channels1,seq_len） 输出是（batch_size,out_channels2,seq_len）
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=conv_kernel_size,
                               padding=padding, stride=conv_stride)
        # 输入是（batch_size,out_channels2,seq_len），输出是（batch_size,out_channels3，seq_len）
        self.conv3 = nn.Conv1d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=conv_kernel_size,
                               padding=padding, stride=conv_stride)
        # 输入是（batch_size,out_channels3，seq_len）,输出是（batch_size,out_channels3,seq_len//pool_kernel_size） 压缩时间维度
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        # 输入是（batch_size,out_channels3,seq_len//pool_kernel_size），输出是(batch_size,out_channels3, 1） 对时间维度进行压缩
        self.fc1 = nn.Linear(self.seq_len // pool_kernel_size, 1)
        # 输入 (batch_size,1,out_channels3) 输出是（batch_size,1,out_channels2） 在特征维度上进行降维
        self.fc2 = nn.Linear(out_channels3, out_channels2)
        # 输入（batch_size,1, out_channels2）-->输出（batch_size,1，in_channels） 得到一天的预测结果
        self.fc3 = nn.Linear(out_channels2, self.in_channels)  # 在特征维度上进行降维

    def pred_onestep(self, input_x):  # 每一次只预测一天
        # input_x [batch,seq_len,dim]
        # 因为conv1d 转为 input_x[batch,dim,seq_len]
        x = self.conv1(input_x.permute(0, 2, 1))  # 对时间维度卷积,变为（batch_size,out_channels1,seq_len）
        x = torch.relu(x)
        x = self.conv2(x)  # 对时间维度卷积,变为（batch_size,out_channels2,seq_len）
        x = torch.relu(x)
        x = torch.relu(self.conv3(x))  # 对时间维度卷积，变为（batch_size,out_channels3，seq_len）
        x = self.pool(x)  # 输出变为（batch_size,out_channels3,seq_len//pool_kernel_size）
        x = self.fc1(x)  # 把预测一天的结果seq=1
        x = torch.relu(x)
        x = x.permute(0, 2, 1)  # 开始处理dim维度
        x = torch.relu(self.fc2(x))  # 输出是（batch_size,1,out_channels2）
        x = self.fc3(x)  # 输出（batch_size,1，in_channels）

        return x

    def forward(self, enc_x, enc_mark, y, y_mark):  # 每一次预测一天，然后把这一次预测的结果拼到已知的时间序列后，然后在移动窗口，预测下一天的
        '''
        :param enc_x: 已知的时间序列 （batch_size,seq_len,dim）
        以下的param本model未使用，不做过多介绍
        :param enc_mark: 已知的时序序列的时间对应的时间矩阵，
        :param y:
        :param y_mark:
        :return:  x_cat_pred[:,-self.pred_len:,:] 将预测的时间序列的部分返回回去 (batch_size,pred)len,dim)
        '''
        pred_zero = torch.zeros_like(y[:, -self.pred_len:, :]).float()  # 初始化预测的结果,shape(batch_size,pred_len,dim)
        # 将已知的时间序列和pred拼接在一起，
        x_cat_pred = torch.cat([enc_x, pred_zero], dim=1).float().to(
            enc_x.device)  # shape(batch_size , seq_len+pred_len , dim)

        for i in range(self.pred_len):  # 循环预测的时间序列的长度，因为一次只预测一天的
            input_x = x_cat_pred[:, i:i + self.seq_len, :].clone()  # 得到每一次已知的时间序列，shape(batch_size,seq_len,dim)
            pred = self.pred_onestep(input_x)  # 得到下一天的预测结果,shape(batch_size,1,dim)
            x_cat_pred[:, self.seq_len + i, :] = x_cat_pred[:, self.seq_len + i, :].clone() + pred.squeeze(
                1)  # 将这一次的预测结果放入x_cat_pred中

        return x_cat_pred[:, -self.pred_len:, :]  # 返回总的预测的时间序列的结果,shape(batch_size,pred_len,dim)
