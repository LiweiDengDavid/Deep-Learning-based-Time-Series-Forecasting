'''
Author：
'''

import numpy as np
import torch
from pmdarima.arima import auto_arima
from tqdm import tqdm
# class ARIMA():
#     def __init__(self,pred_len = 96):
#         super(ARIMA, self).__init__()
#         self.pred_len = pred_len
#
#     def forward(self, enc_x, enc_mark, y, y_mark):
#         #--------------------------------------
#         #   enc_x[batch,seq,dim]
#         #--------------------------------------
#
#         B,S,D = enc_x.shape
#         x = enc_x.numpy()
#
#         #   单独拿一个batch来做
#         for i in range(B):
#             #   单独拿一个dim来做
#             for j in range(D):
#                 ts = x[i,:,j]
#                 #ts[1,seq_len,1] = [seq_len]
#
#                 ts = pd.Series(ts)
#                 ts_log = ts
#                 # ts_log = np.log(ts)
#
#                 #   一阶拆分
#                 diff_12 = ts_log.diff(12)
#                 diff_12.dropna(inplace=True)
#                 diff_12_1 = diff_12.diff(1)
#                 diff_12_1.dropna(inplace=True)
#
#                 #   移动平均，平滑曲线
#                 rol_mean = ts_log.rolling(window=12).mean()
#                 rol_mean.dropna(inplace=True)
#                 ts_diff_1 = rol_mean.diff(1)
#                 ts_diff_1.dropna(inplace=True)
#
#                 #   二阶差分
#                 ts_diff_2 = ts_diff_1.diff(1)
#                 ts_diff_2.dropna(inplace=True)
#
#
#                 #   ARIMA模型，指标要根据数据来调整
#                 model = sm.tsa.arima.ARIMA(ts_diff_1, order=(1, 1, 1))
#                 result_arima = model.fit()
#
#                 # 预测未来10天数据
#                 forecast = result_arima.forecast(self.pred_len)
#
#                 # 一阶差分还原
#                 diff_shift_ts = ts_diff_1.shift(1)
#                 diff_recover_1 = forecast + (diff_shift_ts.values[-1] - diff_shift_ts.values[-2])
#
#                 # 再次一阶差分还原
#                 rol_shift_ts = rol_mean.shift(1)
#                 diff_recover = diff_recover_1.cumsum() + rol_shift_ts.values[-1]
#                 rol_sum = ts_log.rolling(window=11).sum()
#                 rol_sum_shift = rol_sum.shift(1)
#
#                 # 移动平均还原
#                 rol_recover = (diff_recover * 12) - rol_sum_shift.values[-1]
#
#                 # 对数还原
#                 # log_recover = np.exp(rol_recover)
#                 log_recover = rol_recover
#                 out1 = torch.from_numpy(np.array(log_recover)).unsqueeze(0)
#
#                 if j == 0:       ##将第一次第一维特征预测值记录下来
#                     pred = out1
#                     continue
#
#                 ##把每一维特征预测值拼起来
#                 pred = torch.cat([pred,out1], dim=0)
#
#             if i == 0:
#                 out2 = pred.unsqueeze(0)
#                 continue
#             ##把每个batch的预测值拼起来
#             out2 = torch.cat([out2, pred.unsqueeze(0)], dim=0)
#
#         return out2
class ARIMA():
    def __init__(self,seq_len=96,pred_len = 96):
        super(ARIMA, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len



    def forward(self, enc_x):
        #--------------------------------------
        #   enc_x[batch,seq,dim]
        #--------------------------------------

        S,D = enc_x.shape
        x = enc_x

        #   单独拿一个batch来做
        # for i in tqdm(range(B)):
        #   单独拿一个dim来做
        for j in tqdm(range(D)):
            ts = x[-365:,j]
            pre = []  # 存放预测

            for m in range(1):
                ts = np.expand_dims(ts,axis = 1)

                '''
                auto_arima部分参数解析:
                    1.start_p:p的起始值，自回归(“AR”)模型的阶数(或滞后时间的数量),必须是正整数
                    2.start_q:q的初始值，移动平均(MA)模型的阶数。必须是正整数。
                    3.max_p:p的最大值，必须是大于或等于start_p的正整数。
                    4.max_q:q的最大值，必须是一个大于start_q的正整数
                    5.seasonal:是否适合季节性ARIMA。默认是正确的。注意，如果season为真，而m == 1，则season将设置为False。
                    6.stationary :时间序列是否平稳，d是否为零。
                    6.information_criterion：信息准则用于选择最佳的ARIMA模型。(‘aic’，‘bic’，‘hqic’，‘oob’)之一
                    7.alpha：检验水平的检验显著性，默认0.05
                    8.test:如果stationary为假且d为None，用来检测平稳性的单位根检验的类型。默认为‘kpss’;可设置为adf
                    9.n_jobs ：网格搜索中并行拟合的模型数(逐步=False)。默认值是1，但是-1可以用来表示“尽可能多”。
                    10.suppress_warnings：statsmodel中可能会抛出许多警告。如果suppress_warnings为真，那么来自ARIMA的所有警告都将被压制
                    11.error_action:如果由于某种原因无法匹配ARIMA，则可以控制错误处理行为。(warn,raise,ignore,trace)
                    12.max_d:d的最大值，即非季节差异的最大数量。必须是大于或等于d的正整数。
                    13.trace:是否打印适合的状态。如果值为False，则不会打印任何调试信息。值为真会打印一些
                '''
                model = auto_arima(ts, start_p=0, start_q=0, max_p=5, max_q=12, max_d=12,
                                   seasonal=True, test='adf',
                                   error_action='ignore',
                                   information_criterion='aic',
                                   njob=-1, suppress_warnings=True)
                model.fit(ts)
                forecast = model.predict(n_periods=96)
                pre.append(forecast)
            predictions = torch.tensor(np.array(pre).reshape(-1, 1))


            # if j == 0:
            #     # out2 = predictions.squeeze(0)
            #     continue
            # ##把每个batch的预测值拼起来
            # out2 = torch.cat([out2,predictions.unsqueeze(0)], dim=0)

            if j == 0:       ##将第一次第一维特征预测值记录下来
                pred = predictions
                continue

            ##把每一维特征预测值拼起来
            pred = torch.cat([pred,predictions], dim=-1)

            # if i == 0:
            #     out2 = pred.unsqueeze(0)
            #     continue
            # ##把每个batch的预测值拼起来
            # out2 = torch.cat([out2, pred.unsqueeze(0)], dim=0)

        return pred