import os
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
import numpy as np
import torch
from torch import optim, nn
import datetime




class EXP_gan:
    def __init__(self,args):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        self.batch_size = args.batch_size
        self.train_batch = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.verbose = True
        self.lr = args.lr
        self.lr_d = args.lr_d

        self.args = args

        self.train_gpu = [1,]
        self.devices = [0, ]

        self.model_name = args.model_name
        self.data_name = args.data_name

        self.seed = args.seed

        # Construct checkpoint to save training results
        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./checkpoint/'+self.model_name+'/'):
            os.makedirs('./checkpoint/'+self.model_name+'/')


        # Calculate the current time, in order to save the subsequent results
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


        self.modelpath = './checkpoint/'+self.model_name+'/'+self.data_name+'_best_model.pkl'


        #-------------------------------------------
        #   All data naming should be named in a consistent format
        #   And the csv should be processed into a uniform format [date,dim1,dim2......]
        #-------------------------------------------

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

        #Getting data, based on different datasets, mainly requires changes to the
        # get_data function as well as the MyDataset function
        train,valid,test,mean,scale,dim = get_data(self.data_path)

        self.mean = mean
        self.scale = scale
        self.args.data_dim = dim

        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        self.trainloader = DataLoader(trainset, batch_size=self.train_batch,shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size,shuffle=False)
        self.testloader = DataLoader(testset, batch_size=self.batch_size,shuffle=False)
        if self.verbose:
            print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

        return


    def _get_model(self):
        #  Getting the model
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.train_gpu)
        ngpus_per_node = len(self.train_gpu)
        print('Number of devices: {}'.format(ngpus_per_node))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('------------使用设备---------------')
        print(self.device)

        # -------------------------------------------------------------
        #   Select a model based on the model name
        # -------------------------------------------------------------
        if self.model_name == 'AST':
            self.model = AST(self.args)
            self.discriminator = Discriminator(self.args)

        if ngpus_per_node > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)

        self.model.to(self.device)
        self.discriminator.to(self.device)

        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=self.lr_d)

        self.optimizer_G = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer_G, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        if ngpus_per_node > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.devices)
            self.scheduler = nn.DataParallel(self.scheduler, device_ids=self.devices)


        self.early_stopping = EarlyStopping(optimizer=self.optimizer_G,scheduler=self.scheduler,patience=self.patience, verbose=self.verbose, path=self.modelpath,)
        self.criterion = nn.MSELoss()
        self.adversarial_loss = torch.nn.BCELoss()


        if self.args.resume:
            print('Loading pre-trained models')
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])

        return

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark,mode):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        outputs = self.model(batch_x, batch_x_mark,batch_y, batch_y_mark)

        return outputs


    def train(self):

        # self.model.load_state_dict(torch.load(self.modelpath))
        for e in range(self.epochs):
            self.model.train()
            train_loss = []

            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):

                B, S, D = batch_x.shape
                # Constructing True and False Samples
                valid = torch.autograd.Variable(torch.cuda.FloatTensor(B,D).fill_(1.0), requires_grad=False).unsqueeze(1)
                fake = torch.autograd.Variable(torch.cuda.FloatTensor(B,D).fill_(0.0), requires_grad=False).unsqueeze(1)

                labels = batch_y[:, -self.pred_len:,:].float().to(self.device)
                batch_labels = batch_y.clone().float().to(self.device)
                pred = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='train')

                fake_input = pred
                self.optimizer_G.zero_grad()
                loss = loss_quantile(pred[:,-self.pred_len:,:], labels, torch.tensor(0.5)) + 0.1 * self.adversarial_loss(
                    self.discriminator(fake_input), valid)
                loss.backward()
                self.optimizer_G.step()

                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(batch_labels), valid)
                fake_loss = self.adversarial_loss(self.discriminator(fake_input.detach()), fake)
                loss_d = 0.5 * (real_loss + fake_loss)
                loss_d.backward()
                self.optimizer_D.step()


            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                B, S, D = batch_x.shape
                pred = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='val')
                valid = torch.autograd.Variable(torch.cuda.FloatTensor(B, D).fill_(1.0), requires_grad=False).unsqueeze(
                    1)

                labels = batch_y[:, -self.pred_len:,:].float().to(self.device)

                fake_input = pred

                loss = loss_quantile(pred[:,-self.pred_len:,:], labels, torch.tensor(0.5)) + 0.3 * self.adversarial_loss(
                    self.discriminator(fake_input), valid)

                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                B, S, D = batch_x.shape
                pred = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='test')
                valid = torch.autograd.Variable(torch.cuda.FloatTensor(B, D).fill_(1.0), requires_grad=False).unsqueeze(
                    1)

                labels = batch_y[:, -self.pred_len:, :].float().to(self.device)

                fake_input = pred

                loss = loss_quantile(pred[:, -self.pred_len:, :], labels,torch.tensor(0.5)) + 0.1 * self.adversarial_loss(
                    self.discriminator(fake_input), valid)

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
        self.optimizer_G.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])


    def test(self):
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='test')
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:,  -self.pred_len:, :])

        mape_error = np.mean(self.mean)*0.1

        trues, preds = np.array(trues), np.array(preds)
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        dstand_preds = preds * self.scale + self.mean
        dstand_trues = trues * self.scale + self.mean

        mape = np.mean(np.abs(dstand_trues-dstand_preds)/(dstand_trues+mape_error))


        print('Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{0:.4f}'.format(mse, mae, mape))

        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_preds',preds)
        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_trues',trues)

        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'epoches', 'batch_size', 'seed', 'best_mae', 'mse','mape','seq_len','label_len','pred_len']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                  'epoches': self.epochs, 'batch_size': self.batch_size,
                  'seed': self.seed, 'best_mae': mae, 'mse': mse,'mape':mape,'seq_len':self.seq_len,'label_len':self.label_len,'pred_len':self.pred_len }]
        write_csv_dict(log_path, a_log, 'a+')



