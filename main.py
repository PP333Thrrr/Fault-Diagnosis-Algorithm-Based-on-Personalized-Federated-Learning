# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import argparse

from datautil.prepare_data import *
from util.config import img_param_init, normalize_dataset_name, set_random_seed
from util.evalandprint import evalandprint
from alg import algs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed ]')
    parser.add_argument('--datapercent', type=float,
                        default=1e-1, help='data percent to use')
    parser.add_argument('--dataset', type=str, default='pacs',
                        help='[vlcs | pacs | officehome | pamap | covid | medmnist]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    # 通信轮数
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    # 学习率
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=20, help='number of clients')
    # 数据非独立同分布程度（越小，分布约不均匀）
    parser.add_argument('--non_iid_alpha', type=float,
                        default=0.1, help='data split for label shift')
    # 数据分区方式
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way')
    # 特征类型
    parser.add_argument('--plan', type=int,
                        default=1, help='choose the feature type')
    # 预训练轮数
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # 本地训练轮数
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    # 不共享BN层
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')

    # algorithm-specific parameters
    # fedprox 算法的正则化参数
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    # metafed 算法的阈值
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    # MetaFed 算法的初始化 lambda
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    # FedAP 算法的动量参数
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    if args.lr is None:
        default_lrs = {
            'base': 1e-2,
            'fedavg': 1e-2,
            'fedprox': 5e-3,
            'fedbn': 1e-2,
            'fedap': 5e-3,
            'metafed': 1e-3,
        }
        args.lr = default_lrs.get(args.alg, 1e-2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available, falling back to CPU.')
        args.device = 'cpu'

    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    if args.dataset in ['vlcs', 'pacs', 'officehome', 'office-caltech']:
        args = img_param_init(args)
        args.n_clients = 4

    exp_folder = f'fed_{args.dataset}_{args.alg}_{args.datapercent}_{args.non_iid_alpha}_{args.mu}_{args.model_momentum}_{args.plan}_{args.lam}_{args.threshold}_{args.iters}_{args.wk_iters}'
    if args.nosharebn:
        exp_folder += '_nosharebn'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, args.alg)

    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)
    if len(train_loaders) == 0:
        raise RuntimeError(
            'No available client data. Try reducing --n_clients or adjusting the dataset split settings.'
        )
    args.n_clients = len(train_loaders)

    algclass = algs.get_algorithm_class(args.alg)(args)

    if args.alg == 'fedap':
        algclass.set_client_weight(train_loaders)
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters-1
        print('Common knowledge accumulation stage')

    best_changed = False

    best_acc = [0] * args.n_clients
    best_tacc = [0] * args.n_clients
    start_iter = 0
    last_eval_iter = 0

    for a_iter in range(start_iter, args.iters):
        last_eval_iter = a_iter
        print(f"============ Train round {a_iter} ============")

        if args.alg == 'metafed':
            # MetaFed 特殊处理
            for c_idx in range(args.n_clients):
                algclass.client_train(
                    c_idx, train_loaders[algclass.csort[c_idx]], a_iter)
            algclass.update_flag(val_loaders)
        else:
            # local client training
            for wi in range(args.wk_iters):
                for client_idx in range(args.n_clients):
                    algclass.client_train(
                        client_idx, train_loaders[client_idx], a_iter)

            # server aggregation
            algclass.server_aggre()

        best_acc, best_tacc, best_changed = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed)

    # metafed 特殊处理
    if args.alg == 'metafed':
        print('Personalization stage')
        for c_idx in range(args.n_clients):
            algclass.personalization(
                c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
        best_acc, best_tacc, best_changed = evalandprint(
            args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, last_eval_iter, best_changed)

    final_accs = []
    final_precisions = []
    final_recalls = []
    final_f1s = []
    metric_lines = ['Personalized test metrics for each client:']
    for client_idx in range(args.n_clients):
        _, acc, precision, recall, f1 = algclass.client_eval(
            client_idx, test_loaders[client_idx]
        )
        final_accs.append(acc)
        final_precisions.append(precision)
        final_recalls.append(recall)
        final_f1s.append(f1)
        metric_lines.append(
            f' Client-{client_idx:02d} | Acc: {acc:.4f} | Precision: {precision:.4f} | '
            f'Recall: {recall:.4f} | F1: {f1:.4f}'
        )

    mean_acc_test = np.mean(np.array(final_accs))
    mean_precision_test = np.mean(np.array(final_precisions))
    mean_recall_test = np.mean(np.array(final_recalls))
    mean_f1_test = np.mean(np.array(final_f1s))
    metric_lines.append(f'Average accuracy: {mean_acc_test:.4f}')
    metric_lines.append(f'Average precision: {mean_precision_test:.4f}')
    metric_lines.append(f'Average recall: {mean_recall_test:.4f}')
    metric_lines.append(f'Average F1: {mean_f1_test:.4f}')
    print('\n'.join(metric_lines))
