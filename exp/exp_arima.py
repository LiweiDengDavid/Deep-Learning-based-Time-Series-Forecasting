import os
from utils import *
from model import *
import numpy as np
import datetime


class EXP_arima:
    def __init__(self, args):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.verbose = True
        self.args = args
        self.seed = args.seed
        self.model_name = args.model_name
        self.data_name = args.data_name

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./checkpoint/' + self.model_name + '/'):
            os.makedirs('./checkpoint/' + self.model_name + '/')

        self.modelpath = './checkpoint/' + self.model_name + '/' + self.data_name + '_best_model.pkl'


        if self.args.data_name == 'ETTh1':
            self.data_path = './datasets/ETT-small/ETTm1.csv'

        if self.args.data_name == 'ETTm1':
            self.data_path = './datasets/ETT-small/ETTh1.csv'

        if self.args.data_name == 'illness':
            self.data_path = './datasets/illness/national_illness.csv'

        if self.args.data_name == 'electricity':
            self.data_path = './datasets/electricity/electricity.csv'

        if self.args.data_name == 'exchange':
            self.data_path = './datasets/exchange_rate/exchange_rate.csv'

        if self.args.data_name == 'traffic':
            self.data_path = './datasets/traffic/traffic.csv'


        # self._get_data()
        self._get_model()
        self.train_data,self.valid_data,self.test_data,self.mean,self.scale,self.dim =get_data(self.data_path)



    def _get_model(self):
        if self.model_name == 'Arima':
            self.model = ARIMA(pred_len=self.pred_len)
        return

    def test(self):
        pred = self.model.forward(self.train_data[0][:-self.pred_len])
        trues, preds = np.array(self.test_data[0][-self.pred_len:]), np.array(pred)

        mape_error = np.mean(self.mean)*0.1

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        dstand_preds = preds*self.scale+self.mean
        dstand_trues = trues*self.scale+self.mean
        mape = np.mean(np.abs(dstand_trues-dstand_preds)/(dstand_trues+mape_error))


        print('Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{2:.4f}'.format(mse, mae,mape))
        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_preds',preds)
        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_trues',trues)

        # Create a csv file to record the training process
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'epoches', 'batch_size', 'seed', 'best_mae', 'mse', 'mape', 'seq_len', 'label_len',
                           'pred_len']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'seed': self.seed, 'best_mae': mae, 'mse': mse, 'mape': mape, 'seq_len': self.seq_len,
                   'pred_len': self.pred_len}]
        write_csv_dict(log_path, a_log, 'a+')
