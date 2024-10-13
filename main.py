import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from exp.exp import EXP
from exp.exp_arima import EXP_arima
from exp.exp_gan import EXP_gan
from exp.exp_wsaes import EXP_WSAES_LSTM
from exp.exp_tft import EXP_tft
from utils import read_config
from utils.setseed import set_seed
import torch
import argparse

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
def get_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    # Parameters relating to the experimental model
    parser.add_argument('--model_name', default='autoformer', type=str, help='[autoformer,Fedformer,LSTnet,Deepar,TCN,informer,TDformer,reformer,logtrans,TFT\
        ,CNN_1D,GRU_RNN,SAE,Autoencoder,Deepssm,Pyraformer,Aliformer,Transformer,Nbeat,deep_states,SSD,ETSformer,Arima,AST,WSAES_LSTM,\
        PatchTST,Scaleformer,DLinear,Crossformer,Triformer,NS_Transformer,koopa,iTransformer,FITS,TimeMixer]')

    # Call different experiments, some models require special experimental manipulation
    parser.add_argument('--exp', default='deep_learning', type=str, help='[deep_learning,arima,tft,wases,gan:AST]')

    parser.add_argument('--train', default=True, type=str, help='if train')
    parser.add_argument('--resume', default=False, type=str, help='resume from checkpoint')
    parser.add_argument('--save_path', default=None,
                        help='If path is None, exp_id add 1 automatically:if train, it wiil be useful')
    parser.add_argument('--resume_path', default=None,
                        help='if resume is True, it will be useful')
    parser.add_argument('--loss', default='normal', type=str, help='quantile,normal')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    # Parameters related to experimental data
    parser.add_argument('--data_name', default='ETTh1', type=str,help='[data:ETTh1,electricity,exchange,ETTm1,illness,traffic]')
    parser.add_argument('--seq_len', type=int, help='Automatically specified based on dataset')
    parser.add_argument('--label_len', type=int, help='Automatically specified based on dataset')
    parser.add_argument('--pred_len', default=48, type=int, help='prediction len')
    parser.add_argument('--d_mark', default=4, type=int, help='date embed dim')
    parser.add_argument('--d_feature', type=int,help='Automatically specified based on dataset')
    parser.add_argument('--c_out', type=int, help='Automatically specified based on dataset')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # hyperparameter
    parser.add_argument('--d_model', default=510, type=int, help='feature dim in model，must be divisible by nhead')
    parser.add_argument('--n_heads', type=int, default=3, help='num of heads')
    parser.add_argument('--d_ff', default=1024, type=int, help='feature dim2 in model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--lr', default=0.00005, type=float, help='initial learning rate')
    parser.add_argument('--lr_d', default=0.05, type=float, help='initial learning rate for discriminator of gan')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epoches', default=100, type=int, help='Train Epoches')
    parser.add_argument('--patience', default=5, type=int, help='Early Stop patience')

    # Different parameters needed for specific models
    parser.add_argument('--data_dim', default=0, type=int, help='data_dim dont need change')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    parser.add_argument('--info', type=str, default=None, help='information')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    args = parser.parse_args()
    return args

def run_exp(args):
    if args.exp == 'deep_learning':
        exp = EXP(args)
    elif args.exp == 'gan':
        exp = EXP_gan(args)
    elif args.exp == 'wases':
        exp = EXP_WSAES_LSTM(args)
    elif args.exp == 'tft':
        exp = EXP_tft(args)
    elif args.exp == 'arima':
        exp = EXP_arima(args)
        args.train=False
    else:
        raise print(f'没有名字为{args.exp}的exp文件')
    if args.train:
        exp.train()
    exp.test()

def get_dataset_param(args):
    num_feature_dict = {'ETTh1':7,'ETTh2':7,'electricity':321,'exchange':8,'ETTm1':7,'ETTm2':7,'illness':7,'traffic':862}
    num_seq_dict = {'ETTh1':96,'ETTh2':96,'electricity':96,'exchange':96,'ETTm1':96,'ETTm2':96,'illness':36,'traffic':96}
    num_label_dict = {'ETTh1':48,'ETTh2':48,'electricity':48,'exchange':48,'ETTm1':48,'ETTm2':48,'illness':24,'traffic':48}
    args.seq_len=num_seq_dict[args.data_name]
    args.label_len = num_label_dict[args.data_name]
    args.d_feature = num_feature_dict[args.data_name]
    args.c_out=args.d_feature
    return args

if __name__ == '__main__':
    args=get_args() # 得到公共的参数
    args=get_dataset_param(args)
    args=read_config.get_model_params(args) # 得到Model特有参数，并且封装回args中
    set_seed(args.seed)
    print(f"|{'=' * 101}|")
    # 使用__dict__方法获取参数字典，之后遍历字典
    for key, value in args.__dict__.items():
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")
    run_exp(args) # Run Experiment











