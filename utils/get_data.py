import pandas as pd
from sklearn.preprocessing import StandardScaler


def timefeature(dates):
    #-------------------------------------------------------------
    #   通过‘date’属性来对于时间顺序进行一个维度扩增
    #   目的是为了后续的embedding操作
    #   输入数据集必须有'date'属性，要求是年/月/日/时，如果某一个时刻没有，则修改下面的对应特征即可
    #-------------------------------------------------------------


    #---------------------------------------------
    #   经过 pd.to_datetime 自动处理年/月/日后
    #   通过行操作 .hour，.day等可以直接提取对应时间信息
    #---------------------------------------------


    dates["hour"] = dates["date"].apply(lambda row: row.hour / 23 - 0.5, 1)  # 一天中的第几小时
    dates["weekday"] = dates["date"].apply(lambda row: row.weekday() / 6 - 0.5, 1)  # 周几
    dates["day"] = dates["date"].apply(lambda row: row.day / 30 - 0.5, 1)  # 一个月的第几天
    dates["month"] = dates["date"].apply(lambda row: row.month / 11 - 0.5, 1)  # 一年的第几月
    return dates[["hour", "weekday", "day", "month"]].values




def get_data(path):
    df = pd.read_csv(path)
    #-------------------------------------------------------------
    #   提取’date‘属性中的年/月/日/时
    #-------------------------------------------------------------
    df['date'] = pd.to_datetime(df['date'])

    #---------------------------------------------
    #   标准化
    #   对各个特征数据进行预处理
    #---------------------------------------------

    scaler = StandardScaler(with_mean=True,with_std=True)
    # ---------------------------------------------
    #   特征的命名需要满足如下的条件：
    #   对于不同的数据集可以在这里进行修改。
    #   这里以后需要改进成通用的格式
    #   通过直接获取列名称的方式，改的更为通用。
    # ---------------------------------------------

    fields = df.columns.values
    # data = scaler.fit_transform(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values)
    data = scaler.fit_transform(df[fields[1:]].values)
    mean = scaler.mean_
    scale = scaler.scale_
    stamp = scaler.fit_transform(timefeature(df))



    #---------------------------------------------
    #   划分数据集
    #   data是包含除时间外的特征
    #   stamp只包含时间特征
    #---------------------------------------------
    train_data = data[:int(0.6 * len(data)), :]
    valid_data = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
    test_data = data[int(0.8 * len(data)):, :]

    train_stamp = stamp[:int(0.6 * len(stamp)), :]
    valid_stamp = stamp[int(0.6 * len(stamp)):int(0.8 * len(stamp)), :]
    test_stamp = stamp[int(0.8 * len(stamp)):, :]

    dim = train_data.shape[-1]

    return [train_data, train_stamp], [valid_data, valid_stamp], [test_data, test_stamp],mean,scale,dim
