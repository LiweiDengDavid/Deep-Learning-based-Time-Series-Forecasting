import shutil
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from copy import deepcopy
from utils import *
from layers.utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *

import datetime
from layers.Quantile_loss import *
import numpy as np
import torch
from torch import optim, nn

class EXP:
    def __init__(self, args):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        self.batch_size = args.batch_size
        self.train_batch = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.verbose = True
        self.lr = args.lr

        self.args = args

        self.train_gpu = [1, ]
        self.devices = [0, ]

        self.model_name = args.model_name
        self.data_name = args.data_name

        self.seed = args.seed

        # Calculate the current time, in order to save the subsequent results
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if self.args.data_name == 'ETTh1':
            self.data_path = './datasets/ETT-small/ETTh1.csv'

        if self.args.data_name == 'ETTh2':
            self.data_path = './datasets/ETT-small/ETTh2.csv'

        if self.args.data_name == 'ETTm1':
            self.data_path = './datasets/ETT-small/ETTm1.csv'

        if self.args.data_name == 'ETTm2':
            self.data_path = './datasets/ETT-small/ETTm2.csv'
            
        if self.args.data_name == 'illness':
            self.data_path = './datasets/illness/national_illness.csv'

        if self.args.data_name == 'exchange':
            self.data_path = './datasets/exchange_rate/exchange_rate.csv'

        if self.args.data_name == 'traffic':
            self.data_path = './datasets/traffic/traffic.csv'
        
        if self.args.data_name=='electricity':
            self.data_path = './datasets/electricity/electricity.csv'

        self._get_path()
        self._get_data()
        self._get_model()

    def _get_data(self):

        # Getting data, based on different datasets
        train, valid, test, mean, scale, dim = get_data(self.data_path)

        self.mean = mean
        self.scale = scale

        self.args.data_dim = dim

        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        self.trainloader = DataLoader(trainset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        if self.verbose:
            print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

        return

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('------------Use of equipment---------------')
        print(self.device)

        if self.model_name == 'autoformer':
            self.model = Autoformer(self.args)

        if self.model_name == 'Fedformer':
            self.model = Fedformer(self.args)

        if self.model_name == 'LSTnet':
            self.model = LSTnet(self.args)

        if self.model_name == 'Deepar':
            self.model = Deepar(self.args)

        if self.model_name == 'TCN':
            self.model = TCN(self.args)

        if self.model_name == 'informer':
            self.model = Informer(self.args)

        if self.model_name == 'TDformer':
            self.model = TDformer(self.args)

        if self.model_name == 'reformer':
            self.model = Reformer(self.args)

        if self.model_name == 'logtrans':
            self.model = LogTrans(self.args)

        if self.model_name == 'TFT':
            self.model = TFT(self.args)

        if self.model_name == 'CNN_1D':
            self.model = CNN_1D(self.args)

        if self.model_name == 'GRU_RNN':
            self.model = GRU_RNN_Model(self.args)

        if self.model_name == 'SAE':
            self.model = SAE(self.args)

        if self.model_name == 'Autoencoder':
            self.model = Autoencoder(self.args)

        if self.model_name == 'Deepssm':
            self.model = DeepSSM(self.args)

        if self.model_name == 'Pyraformer':
            self.model = Pyraformer(self.args)

        if self.model_name == 'Aliformer':
            self.model = Aliformer(self.args)

        if self.model_name == 'Transformer':
            self.model = Transformer(self.args)

        if self.model_name == 'Nbeat':
            self.model = NBeatsNet(self.args)

        if self.model_name == 'deep_states':
            self.model = Deep_states(self.args)

        if self.model_name == 'SSD':
            self.model = SSD(self.args)

        if self.model_name == 'ETSformer':
            self.model = ETSformer(self.args)

        if self.model_name == 'PatchTST':
            self.model = PatchTST(self.args)

        if self.model_name == 'Scaleformer':
            self.model = Scaleformer(self.args)
        #
        if self.model_name == 'DLinear':
            self.model = DLinear(self.args)

        if self.model_name == 'Crossformer':
            self.model = Crossformer(self.args)
        #
        if self.model_name == 'Triformer':
            self.model = Triformer(self.args)

        if self.model_name == 'NS_Transformer':
            self.model = NS_Transformer(self.args)

        if self.model_name == 'koopa':
            mask_spectrum = self._get_mask_spectrum()
            self.args.mask_spectrum = mask_spectrum
            self.model = koopa(self.args)

        if self.model_name == 'FITS':
            self.model = FITS(self.args)

        if self.model_name == 'TimeMixer':
            self.model = TimeMixer(self.args)

        if self.model_name == 'iTransformer':
            self.model = iTransformer(self.args)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.9 ** ((epoch - 1) // 1))

        self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler, patience=self.patience,
                                            verbose=self.verbose, path=self.modelpath, )

        if self.args.loss == 'quantile':
            self.criterion = QuantileLoss(self.args.quantiles)

        if self.args.loss == 'normal':

            if self.model_name == 'TFT':
                self.criterion = QuantileLoss_TFT([0.1, 0.5, 0.9])
            else:
                self.criterion = nn.MSELoss()

        if self.args.resume:
            print('Loading pre-trained models')
            self.resumepath = self.path + '/' + self.args.resume_path + '/best_model.pkl'
            checkpoint = torch.load(self.resumepath)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def _get_path(self):

        self.path = './checkpoint/' + self.model_name + '/' + self.data_name + '_best_model.pkl'

        # Build checkpoint to save training results
        self.path = './checkpoint/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.path = self.path + '/' + self.model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.path = self.path + '/' + self.data_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        exp_id = self.args.save_path
        if exp_id is not None and exp_id != 'None' and exp_id != 'none':
            self.modelpath = self.path + '/' + exp_id
            if self.args.train:
                if os.path.exists(self.modelpath):
                        shutil.rmtree(self.modelpath)
                os.makedirs(self.modelpath)

        else:
            # 如为None则自动加一
            path_list = os.listdir(self.path)
            if path_list == []:
                self.modelpath = self.path + '/exp0'

            else:
                path_list = [int(idx[3:]) for idx in path_list]
                self.modelpath = self.path + '/exp' + str(max(path_list) + 1)

            os.makedirs(self.modelpath)
        self.savepath = deepcopy(self.modelpath)
        self.modelpath = self.modelpath + '/best_model.pkl'


    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_loader =self.trainloader
        self.args.alpha=0.2
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0] * self.args.alpha)).indices
        return mask_spectrum  # as the spectrums of time-invariant component

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if self.model_name == 'Deepar':
            outputs, loss = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mode)

        elif self.model_name == 'Aliformer':
            outputs, loss = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mode)

        elif self.model_name == 'FITS':
            outputs, loss = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

        else:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])

        return outputs, loss

    def train(self):

        for e in range(self.epochs):
            self.model.train()
            train_loss = []

            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):
                # ------------------------------------------------
                #   [batch_size,seq_len,dim]
                # ------------------------------------------------

                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')

                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')
                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
                test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model, e)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        # Read the previously saved weights file (including the optimiser and the learning rate policy).
        # Read the optimal model parameters since they will be sent to the test function for testing next
        checkpoint = torch.load(self.modelpath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def test(self):
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            f_dim = -1 if self.args.features == 'MS' else 0
            pred = pred[:, -self.args.pred_len:, f_dim:].to(self.device)
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:, -self.pred_len:, :])
        mape_error = np.mean(self.mean) * 0.1
        trues, preds = np.array(trues), np.array(preds)
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        # inverse normalisation：
        dstand_preds = preds * self.scale[f_dim:] + self.mean[f_dim:]
        dstand_trues = trues * self.scale[f_dim:] + self.mean[f_dim:]
        mape = np.mean(np.abs((dstand_trues - dstand_preds) / (dstand_trues + mape_error)))

        print('Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{2:.4f}'.format(mse, mae, mape))

        np.save(self.savepath + '/test_preds', preds)
        np.save(self.savepath + '/test_trues', trues)

        # 创建csv文件记录训练过程
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'epoches', 'batch_size', 'seed', 'best_mae', 'mse', 'mape', 'seq_len', 'label_len',
                           'pred_len','d_model','d_ff','weight_decay','type','e_layers','d_layers','info']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # Get current system time
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                  'epoches': self.epochs, 'batch_size': self.batch_size,
                  'seed': self.seed, 'best_mae': mae, 'mse': mse, 'mape': mape, 'seq_len': self.seq_len,
                  'label_len': self.label_len, 'pred_len': self.pred_len,'d_model':self.args.d_model,'d_ff':self.args.d_ff,
                  'weight_decay':self.args.weight_decay,'type':self.args.features,'e_layers':self.args.e_layers,'d_layers':self.args.d_layers,'info':f'{self.modelpath}+{self.args.info}'}]
        write_csv_dict(log_path, a_log, 'a+')

