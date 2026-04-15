import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datautil.datasplit import define_pretrain_dataset
from datautil.prepare_data import get_whole_dataset


class TrainingCancelled(Exception):
    pass


def check_cancel(args):
    cancel_checker = getattr(args, 'cancel_checker', None)
    if callable(cancel_checker) and cancel_checker():
        raise TrainingCancelled()


def train(model, data_loader, optimizer, loss_fun, device, args=None):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        check_cancel(args)
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    # 用于计算精确率、召回率、F1分数
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
            
            # 收集预测和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        accuracy = correct/total
        
        # 计算精确率、召回率、F1分数
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        return loss_all / len(data_loader), accuracy, precision, recall, f1


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        check_cancel(args)
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        w_diff = torch.tensor(0., device=device)
        for w, w_t in zip(server_model.parameters(), model.parameters()):
            w_diff += torch.sum((w_t - w) ** 2)
        loss += args.mu / 2. * w_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def trainwithteacher(model, data_loader, optimizer, loss_fun, device, tmodel, lam, args, flag):
    check_cancel(args)
    model.train()
    if tmodel:
        tmodel.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        check_cancel(args)
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        f1 = model.get_sel_fea(data, args.plan)
        loss = loss_fun(output, target)
        if flag and tmodel:
            f2 = tmodel.get_sel_fea(data, args.plan).detach()
            loss += (lam*F.mse_loss(f1, f2))
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def pretrain_model(args, model, filename, device='cuda'):
    print('===training pretrained model===')
    data = get_whole_dataset(args.dataset)(args)
    predata = define_pretrain_dataset(args, data)
    traindata = torch.utils.data.DataLoader(
        predata, batch_size=args.batch, shuffle=True)
    loss_fun = nn.CrossEntropyLoss()
    opt = optim.SGD(params=model.parameters(), lr=args.lr)
    for _ in range(args.pretrained_iters):
        check_cancel(args)
        _, acc = train(model, traindata, opt, loss_fun, device, args)
    torch.save({
        'state': model.state_dict(),
        'acc': acc
    }, filename)
    print('===done!===')
