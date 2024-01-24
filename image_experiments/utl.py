# -*- coding: utf-8 -*-

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

def CELoss():
    criterion = nn.CrossEntropyLoss()
    return criterion


def Evalute(model,data_loader,criterion,device='cuda'):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    loss_avg = 0.
    with torch.no_grad():    
        for idx,[data, target] in enumerate(data_loader):
            data,target=data.to(device),target.long().to(device)   
            output = model(data)
            loss = criterion(output, target)
            loss_avg += loss.item()
            pred = torch.max(output.detach(), 1)[1]
            total += target.size(0)
            correct += (pred == target.detach()).sum().float().item()
    val_acc = correct / total
    print('\n Test set: Error:{}  Accuracy: {}\n'.format(loss_avg/(idx+1), val_acc))
    return loss_avg/(idx+1), val_acc

def EvaluteTop5(model,data_loader,device='cuda'):
    model.eval()
    correct = 0
    total = 0.
    for x, y in data_loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            nul, pred = logits.topk(maxk, 1, True, True)
            total += y.size(0)
            correct += torch.eq(pred, y_resize).sum().float().item()
    val_acc = correct / total
    print('\nTest set: Top5 Accuracy: {}\n'.format(val_acc))
    return val_acc

def EpochTrain(epoch,model, optimizer,train_loader,criterion,device='cuda',dataset='cifar10'):

    model.train()
    correct = 0.
    total = 0.
    loss_avg = 0.
    accuracy = 0.
    since = time.time()
    for idx,[data, target] in enumerate(train_loader):
        data, target = data.to(device), target.long().to(device) 
        loss = 0.
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
        pred = torch.max(output.detach(), 1)[1]
        total += target.size(0)
        correct += (pred == target.detach()).sum()
        accuracy = correct.item() / total
        if idx % 100 ==0: 
            print('epoch %4.1f'% epoch,'iter %5.1f'% idx,'train loss %9.8f'% (loss_avg/ (idx+1.)),'train acc %5.4f' % accuracy)   
    print('epoch %4.1f'% epoch,'iter %5.1f'% idx,'train loss %9.8f'% (loss_avg/ (idx+1.)),'train acc %5.4f' % accuracy)   
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return loss_avg / (idx + 1.), correct.item() / total
