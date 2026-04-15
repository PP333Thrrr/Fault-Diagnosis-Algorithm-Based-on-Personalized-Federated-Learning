# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import copy

from util.modelsel import modelsel
from util.traineval import trainwithteacher, test


class metafed(torch.nn.Module):
    def __init__(self, args):
        super(metafed, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        args.sort = ''
        for i in range(args.n_clients):
            args.sort += '%d-' % i
        args.sort = args.sort[:-1]
        self.args = args
        self.csort = [int(item) for item in args.sort.split('-')]
        self.teacher_models = None
        self.teacher_snapshot_round = None
        self.personalization_teacher_models = None

    def _is_personalized_head_key(self, model, key):
        if hasattr(model, 'classifier') and key.startswith('classifier.fc3.'):
            return True
        if hasattr(model, 'fc3') and key.startswith('fc3.'):
            return True
        if hasattr(model, 'fc2') and not hasattr(model, 'fc3') and key.startswith('fc2.'):
            return True
        return False

    def _copy_teacher_weights(self, student_model, teacher_model):
        if teacher_model is None:
            return
        with torch.no_grad():
            for key in teacher_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    continue
                if self.args.nosharebn and 'bn' in key:
                    continue
                if self._is_personalized_head_key(student_model, key):
                    continue
                student_model.state_dict()[key].data.copy_(
                    teacher_model.state_dict()[key].data
                )

    def _refresh_teacher_models(self, round_idx):
        if self.teacher_snapshot_round == round_idx and self.teacher_models is not None:
            return
        self.teacher_models = [
            copy.deepcopy(model).to(self.args.device)
            for model in self.client_model
        ]
        self.teacher_snapshot_round = round_idx

    def _refresh_personalization_teacher_models(self):
        if self.personalization_teacher_models is not None:
            return
        self.personalization_teacher_models = [
            copy.deepcopy(model).to(self.args.device)
            for model in self.client_model
        ]

    def init_model_flag(self, train_loaders, val_loaders):
        client_num = self.args.n_clients
        self.flagl = [False for _ in range(client_num)]
        self.teacher_models = None
        self.teacher_snapshot_round = None
        self.personalization_teacher_models = None
        base_state = copy.deepcopy(self.server_model.state_dict())
        for idx in range(client_num):
            model = copy.deepcopy(self.server_model).to(self.args.device)
            model.load_state_dict(base_state)
            optimizer = optim.SGD(params=model.parameters(), lr=self.args.lr)
            warmup_iters = max(1, self.args.wk_iters)
            for _ in range(warmup_iters):
                trainwithteacher(
                    model,
                    train_loaders[idx],
                    optimizer,
                    self.loss_fun,
                    self.args.device,
                    None,
                    0,
                    self.args,
                    False
                )
            _, val_acc, _, _, _ = test(
                model, val_loaders[idx], self.loss_fun, self.args.device
            )
            self.flagl[idx] = val_acc > self.args.threshold

        for idx in range(client_num):
            self.client_model[idx].load_state_dict(copy.deepcopy(base_state))

        if self.args.dataset in ['vlcs', 'pacs']:
            self.thes = 0.4
        elif 'medmnist' in self.args.dataset:
            self.thes = 0.5
        elif 'pamap' in self.args.dataset:
            self.thes = 0.5
        else:
            self.thes = 0.5

    def update_flag(self, val_loaders):
        for client_idx, model in enumerate(self.client_model):
            _, val_acc, _, _, _ = test(
                model, val_loaders[client_idx], self.loss_fun, self.args.device)
            self.flagl[client_idx] = val_acc > self.args.threshold

    def client_train(self, c_idx, dataloader, round):
        self._refresh_teacher_models(round)
        # 确保c_idx不超出范围
        if c_idx >= len(self.csort):
            c_idx = c_idx % len(self.csort)
        client_idx = self.csort[c_idx]
        
        # 处理第一个客户端的情况
        if round == 0 and c_idx == 0:
            tmodel = None
        else:
            # 确保索引不越界
            tidx_idx = c_idx - 1 if c_idx > 0 else len(self.csort) - 1
            tidx = self.csort[tidx_idx]
            tmodel = self.teacher_models[tidx]

        model, train_loader, optimizer = self.client_model[
            client_idx], dataloader, self.optimizers[client_idx]
        if tmodel is not None and not self.flagl[client_idx]:
            self._copy_teacher_weights(model, tmodel)
        for _ in range(self.args.wk_iters):
            train_loss, train_acc = trainwithteacher(
                model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, self.args.lam, self.args, self.flagl[client_idx])
        return train_loss, train_acc

    def personalization(self, c_idx, dataloader, val_loader):
        client_idx = self.csort[c_idx]
        self._refresh_personalization_teacher_models()
        tidx_idx = c_idx - 1 if c_idx > 0 else len(self.csort) - 1
        tidx = self.csort[tidx_idx]
        model, train_loader, optimizer, tmodel = self.client_model[
            client_idx], dataloader, self.optimizers[client_idx], self.personalization_teacher_models[tidx]

        with torch.no_grad():
            _, v1a, _, _, _ = test(model, val_loader, self.loss_fun, self.args.device)
            _, v2a, _, _, _ = test(tmodel, val_loader, self.loss_fun, self.args.device)

        if v2a <= v1a and v2a < self.thes:
            lam = 0
        else:
            lam = (10**(min(1, (v2a-v1a)*5)))/10*self.args.lam

        if lam == 0:
            tmodel = None
        for _ in range(self.args.wk_iters):
            train_loss, train_acc = trainwithteacher(
                model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, lam, self.args, True)
        return train_loss, train_acc

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc, precision, recall, f1 = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc, precision, recall, f1

    def server_aggre(self):
        # MetaFed 不需要传统的服务器聚合
        # 它使用教师模型的方式进行知识传递
        pass
