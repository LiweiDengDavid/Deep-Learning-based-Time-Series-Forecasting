from exp.exp import EXP
from exp.exp_arima import EXP_arima
from exp.exp_gan import EXP_gan
from exp.exp_wsaes import EXP_WSAES_LSTM
from exp.exp_tft import EXP_tft
from utils.setseed import set_seed
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':

    import argparse

    #---------------------------------------------------------------------------
    #   Load Basic Parameters
    #   resume：Whether to load pre-trained models
    #   seq_len：Input historical data length
    #   label_len：Historical data length for decoder (used by transfomer series)
    #   pred_len：Predicted length
    #   lr：learning rate
    #   batch size：Batch size
    #   model name：Model name
    #   data name：Use the dataset name
    #   patience：Training uses an early stopping mechanism, so PATIENCE controls the number of rounds of judgement for early stopping
    #   seed：random seed
    #---------------------------------------------------------------------------


    parser = argparse.ArgumentParser(
        description=__doc__)

    # Parameters relating to the experimental model
    parser.add_argument('--model_name',default='Transformer',type=str,help='[autoformer,Fedformer,LSTnet,Deepar,TCN,informer,TDformer,reformer,logtrans,TFT\
    ,CNN_1D,GRU_RNN,SAE,Autoencoder,Deepssm,Pyraformer,Aliformer,Transformer,Nbeat,deep_states,SSD,ETSformer,Arima,AST,WSAES_LSTM,\
    PatchTST,Scaleformer,DLinear,Crossformer,Triformer,NS_Transformer,koopa]')

    # Call different experiments, some models require special experimental manipulation
    parser.add_argument('--exp',default='deep_learning',type=str,help='[deep_learning,arima,tft,wases,gan:AST]')

    parser.add_argument('--train',default=True,type=str,help='if train')
    parser.add_argument('--resume', default=False, type=str, help='resume from checkpoint')
    parser.add_argument('--loss', default='normal', type=str, help='quantile,normal')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    # Parameters related to experimental data
    parser.add_argument('--data_name', default='exchange', type=str, help='[data:ETTh1,electricity,exchange,ETTm1,illness,traffic]')
    parser.add_argument('--seq_len',default=96,type=int,help='input sequence len')
    parser.add_argument('--label_len',default=48,type=int,help='transfomer decoder input part')
    parser.add_argument('--pred_len',default=96,type=int,help='prediction len')
    parser.add_argument('--d_mark', default=4, type=int, help='date embed dim')
    parser.add_argument('--d_feature',default=8,type=int,help='input data feature dim without date :[Etth1:7 , electricity:321,exchange:8,ETTm1:7,illness:7,traffic:862]')
    parser.add_argument('--c_out', type=int, default=8, help='output size')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')


    # hyperparameter
    parser.add_argument('--d_model', default=510, type=int, help='feature dim in model，must be divisible by nhead')
    parser.add_argument('--n_heads', type=int, default=3, help='num of heads')
    parser.add_argument('--d_ff', default=1024, type=int, help='feature dim2 in model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--lr',default=0.001,type=float,help='initial learning rate')
    parser.add_argument('--lr_d',default=0.05,type=float,help='initial learning rate for discriminator of gan')
    parser.add_argument('--batch_size',default=32,type=int,help='batch size')
    parser.add_argument('--epoches',default=100,type=int,help='Train Epoches')
    parser.add_argument('--patience',default=5,type=int,help='Early Stop patience')



    # Different parameters needed for specific models
    parser.add_argument('--data_dim',default=0,type=int,help='data_dim dont need change')


    # pyraformer
    parser.add_argument('--window_size', type=int, default=[4, 4, 4])  # The number of children of a parent node.
    parser.add_argument('--CSCM', type=str, default='Bottleneck_Construct')
    parser.add_argument('--embed_type', type=str,
                        default='DataEmbedding')  #Select Embedding Method：  DataEmbedding  or  CustomEmbedding
    parser.add_argument('--truncate', action='store_true',
                        default=False)  # Whether to remove coarse-scale nodes from the attention structure
    parser.add_argument('--use_tvm', action='store_true', default=False)  # Whether to use TVM.
    parser.add_argument('--decoder', type=str, default='FC')  # selection: [FC, attention]
    parser.add_argument('--inner_size', type=int, default=3)  # Number of similar nodes in the same layer that can transmit information
    parser.add_argument('--device', type=int, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--d_inner_hid', type=int, default=256)
    parser.add_argument('--d_k', type=int, default=128)  #  Number of key-value pairs
    parser.add_argument('--d_v', type=int, default=128)  #  value Number of key-value pairs
    parser.add_argument('--d_bottleneck', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=4)

    # Nbeat
    # N_beat has three stacks, each containing a certain number of dimensional variations.
    parser.add_argument('--d_dimension',type=int,default=64,help='the hidden dimension in model')
    parser.add_argument('--stack_dimension',type=int,default=[2,8,3],help='the dimension of each stack in Nbeat')
    # The tft loss Korean uses a similar approach to quartiles, where the values of the quartiles are set as a list.
    parser.add_argument('--quantiles',default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],type=int,help='the quantiles used in the loss of TFT')
    # Nbeat involves dimensional transformations inside each stack, set the transformed dimensions here
    parser.add_argument('--d_nbeat',default=[2,8,3],type=int,help='the dimension of each stack in Nbeat')

    # fedformer
    parser.add_argument('--version', default='WSAES_LSTM', type=str, help='fourier or wavelet')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # informer
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    # ETSformer
    parser.add_argument('--K', type=int, default=3, help='Top-K Fourier bases')
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--std', type=float, default=0.2)

    # patchTST
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')

    # scaleformer
    parser.add_argument('--scale_factor', type=int, default=2, help='scale_factor')

    # Crossformer
    parser.add_argument('--seg_len', type=int, default=6, help='seg_len')
    parser.add_argument('--win_size', type=int, default=2, help='win_size')

    # Triformer
    parser.add_argument('--patch_sizes', default=[6, 4, 4], help='the patch_len in each layer for Triformer'
                                                                 'advice:24:(4,3,2,2)    96:(6,4,4)    192:(6,4,4,2)'
                                                                 '       336:(7,4,3,2,2)  720:(6,4,4)')

    # de-stationary projector params (NS_Transformer)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--info', type=str, default=None, help='information')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    args = parser.parse_args()

    for i in range(1,6):
        args.seed=i
        set_seed(i)
        print('exp:'+args.exp)
        if args.exp == 'deep_learning':
            exp = EXP(args)
            if args.train:
                exp.train()
            exp.test()
        elif args.exp == 'arima':
            exp = EXP_arima(args)
            exp.test()
        elif args.exp == 'gan':
            exp = EXP_gan(args)
            if args.train:
                exp.train()
            exp.test()
        elif args.exp == 'wases':
            exp = EXP_WSAES_LSTM(args)
            if args.train:
                exp.train()
            exp.test()
        elif args.exp == 'tft':
            exp = EXP_tft(args)
            if args.train:
                exp.train()
            exp.test()









