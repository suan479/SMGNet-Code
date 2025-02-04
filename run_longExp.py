import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='family for Time Series Forecasting')
parser.add_argument('--is_training', type=int,  default=1)
parser.add_argument('--train_only', type=bool,  default=False)
parser.add_argument('--model_id', type=str,  default='test')
parser.add_argument('--model', type=str,default='SMGNet')
parser.add_argument('--data', type=str, default='ETTh2')
parser.add_argument('--root_path', type=str, default='./Data/')
parser.add_argument('--data_path', type=str, default='ETTh2.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--seq_len', type=int, default=720)
parser.add_argument('--seg_len', type=int, default=24)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--individual', action='store_true', default=False)
parser.add_argument('--embed_type', type=int, default=0)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--distil', action='store_false')
parser.add_argument('--dropout', type=float, default=0.05)  
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--pct_start', type=float, default=0.3)
parser.add_argument('--use_amp', action='store_true')
parser.add_argument('--patchvechidden', type=int, default=1)
parser.add_argument('--anti_ood', type=int, default=1)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--pj_subgraph_size', type=int, default=10)
parser.add_argument('--scale_subgraph_size', type=int, default=2)
parser.add_argument('--gc_layer', type=int, default=1)
parser.add_argument('--down_sampling_layers', type=int, default=1)
parser.add_argument('--down_sampling_window', type=int, default=2)
parser.add_argument('--use_multi_scale', type=int, default=1)
parser.add_argument('--scale_seg_len', type=int, default=24)
parser.add_argument('--e_layers', type=int, default=1)
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--dt_rank', type=int, default=32)
parser.add_argument('--d_state', type=int, default=16)
parser.add_argument('--dt_init', type=str, default='random')
parser.add_argument('--dt_scale', type=float, default=1.0)
parser.add_argument('--dt_max', type=float, default=0.1)
parser.add_argument('--dt_min', type=float, default=0.001)
parser.add_argument('--dt_init_floor', type=float, default=1e-4)
parser.add_argument('--pscan', action='store_true')
parser.add_argument('--avg', action='store_true')
parser.add_argument('--max', action='store_true')
parser.add_argument('--reduction', type=int, default=2)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--use_multi_gpu', action='store_true')
parser.add_argument('--devices', type=str, default='0,1,2,3')
parser.add_argument('--test_flop', action='store_true', default=False)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_sg{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_lr{}_ood{}_loss{}_layer{}_hidden{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.seg_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii,
            args.learning_rate,
            args.anti_ood,
            args.loss,
            args.e_layers,
            args.hidden,
            )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_sg{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.seg_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
