
import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
import datetime




class EXP_tft:
    def __init__(self,args):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        self.batch_size = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.lr = args.lr
        self.quantiles = args.quantiles

        self.args = args

        self.train_gpu = [1,]
        self.devices = [0, ]

        self.model_name = args.model_name
        self.data_name = args.data_name

        self.seed = args.seed

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./checkpoint/'+self.model_name+'/'):
            os.makedirs('./checkpoint/'+self.model_name+'/')


        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


        self.modelpath = './checkpoint/'+self.model_name+'/'+self.data_name+'_best_model.pkl'


        
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

            
            
        self._get_data()
        self._get_model()

    def _get_data(self):

        train,valid,test,mean,scale,dim = get_data(self.data_path)

        self.mean = mean
        self.scale = scale

        self.args.data_dim = dim

        trainset = MyDataset_tft(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset_tft(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset_tft(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size,shuffle=True,drop_last=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size,shuffle=False,drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size,shuffle=False,drop_last=True)

        print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

        return


    def _get_model(self):
        #   获取模型
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.train_gpu)
        ngpus_per_node = len(self.train_gpu)
        print('Number of devices: {}'.format(ngpus_per_node))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('------------使用设备---------------')
        print(self.device)


        quantiles = self.quantiles
        self.model = TFT(self.args)


        if ngpus_per_node > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        if ngpus_per_node > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.devices)
            self.scheduler = nn.DataParallel(self.scheduler, device_ids=self.devices)


        self.early_stopping = EarlyStopping(optimizer=self.optimizer,scheduler=self.scheduler,patience=self.patience,path=self.modelpath,)

        self.criterion = nn.MSELoss()
        self.QuantileLoss = QuantileLoss(self.quantiles)


        if self.args.resume:
            print('Loading pre-trained models')
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])


        return

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark,category,mode):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        category = category.float().to(self.device)

        static = category[:,0].unsqueeze(-1)
        for i in range(batch_x.shape[-1]):
            if i == 0:
                batch_x_mark_concat = batch_x_mark.unsqueeze(-1)
            else:
                batch_x_mark_concat = torch.cat((batch_x_mark_concat,batch_x_mark.unsqueeze(-1)),dim=-1)
        batch_x_mark_concat = batch_x_mark_concat.permute(0,1,3,2)
        past_input = torch.cat((batch_x.unsqueeze(-1),batch_x_mark_concat),dim=-1)
        future_time = batch_y_mark[:,-self.pred_len:]
        for i in range(batch_x.shape[-1]):
            if i == 0:
                future_input = future_time.unsqueeze(-1)
            else:
                future_input = torch.cat((future_input,future_time.unsqueeze(-1)),dim=-1)
        future_input = future_input.permute(0,1,3,2)
        outputs = self.model(static,past_input,future_input)
        loss = self.QuantileLoss(outputs,batch_y[:, -self.pred_len:, :])
        outputs = torch.mean(outputs,dim=-1)

        return outputs, loss


    def train(self):

        for e in range(self.epochs):
            self.model.train()
            train_loss = []

            for (batch_x, batch_y, batch_x_mark, batch_y_mark,category) in tqdm(self.trainloader):

                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,category,mode='train')
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark,category) in tqdm(self.validloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,category,mode='val')
                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark,category) in tqdm(self.testloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,category,mode='test')
                test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model,e)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        checkpoint = torch.load(self.modelpath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])






    def test(self):
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark,category) in tqdm(self.testloader):
            pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,category,mode='test')
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:,  -self.pred_len:, :])

        mape_error = np.mean(self.mean)*0.1

        trues, preds = np.array(trues), np.array(preds)
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        dstand_preds = preds*self.scale+self.mean
        dstand_trues = trues*self.scale+self.mean

        mape = np.mean(np.abs(dstand_trues-dstand_preds)/(dstand_trues+mape_error))


        print('Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{0:.4f}'.format(mse, mae,mape))

        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_preds',preds)
        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_trues',trues)

        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'epoches', 'batch_size', 'seed', 'best_mae', 'mse','mape','seq_len','label_len','pred_len']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                  'epoches': self.epochs, 'batch_size': self.batch_size,
                  'seed': self.seed, 'best_mae': mae, 'mse': mse,'mape':mape,'seq_len':self.seq_len,'label_len':self.label_len,'pred_len':self.pred_len }]
        write_csv_dict(log_path, a_log, 'a+')


class QuantileLoss(nn.Module):

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # target.shape:bs,prelen,dim
        # preds.shape bs,prelen,dim,quantiles_number
        preds = preds.view(-1, len(self.quantiles))
        target = target.flatten()
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class MyDataset_tft(Dataset):
    def __init__(self, data, seq_len=96, label_len=48, pred_len=96):
        self.data = data[0]
        self.stamp = data[1]
        self.category = data[2]

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.label_len
        d_end = e_end + self.pred_len

        seq_x = self.data[e_begin:e_end]
        seq_y = self.data[d_begin:d_end]
        seq_x_mark = self.stamp[e_begin:e_end]
        seq_y_mark = self.stamp[d_begin:d_end]
        category = self.category[e_begin:e_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,category

    def __len__(self):

        return len(self.data) - self.seq_len - self.pred_len + 1



def get_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])


    scaler = StandardScaler(with_mean=True,with_std=True)


    fields = df.columns.values
    data = scaler.fit_transform(df[fields[1:]].values)
    mean = scaler.mean_
    scale = scaler.scale_
    stamp = scaler.fit_transform(timefeature(df))

    category = np.array(df.shape[0]*[mean])

    train_data = data[:int(0.6 * len(data)), :]
    valid_data = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
    test_data = data[int(0.8 * len(data)):, :]

    train_stamp = stamp[:int(0.6 * len(stamp)), :]
    valid_stamp = stamp[int(0.6 * len(stamp)):int(0.8 * len(stamp)), :]
    test_stamp = stamp[int(0.8 * len(stamp)):, :]

    train_category = category[:int(0.6 * len(stamp)), :]
    valid_category = category[int(0.6 * len(stamp)):int(0.8 * len(stamp)), :]
    test_category = category[int(0.8 * len(stamp)):, :]

    dim = train_data.shape[-1]

    return [train_data, train_stamp,train_category], [valid_data, valid_stamp,valid_category], [test_data, test_stamp,test_category],mean,scale,dim


