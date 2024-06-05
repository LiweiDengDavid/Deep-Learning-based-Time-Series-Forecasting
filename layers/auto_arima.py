import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

# 生成一个路径


def main(ts):
    # 加载数据
    # 定义将字符串时间转化成日期时间数组

    # 预处理数据-由于所给数据本身就是单变量序列，并且没有空值，因此，可以不进行这一步处理

    # 设置滚动预测的参数

    test_size = 10  # 需要预测的个数
    rolling_size = 240  # 滚动窗口大小
    ps = 1  # 每次预测的个数
    horizon = 1  # 用来消除切片的影响
    pre = []  # 存放预测值
    test = ts[-test_size:]

    # 滚动预测
    for i in range(test_size):
        train = ts[-(rolling_size + test_size - i):-(test_size + horizon - i)]

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
        model = auto_arima(train, start_p=0, start_q=0, max_p=12, max_q=12, max_d=5,
                           seasonal=True, test='adf',
                           error_action='ignore',
                           information_criterion='aic',
                           njob=-1, suppress_warnings=True)
        model.fit(train)
        forecast = model.predict(n_periods=ps)
        pre.append(forecast[-1])
    # print(train)
    print(pre)

    predictions_ = pd.Series(pre,index=test.index)
    # print(predictions)




