# -*- coding: utf-8 -*-
"""
"""
from tqdm import tqdm
import argparse
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
from ann import HouseholderNet
from LaNet import NetworkCIFAR as LaNet
from nasnet_set import *


parser = argparse.ArgumentParser(description=' test')
parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='gpu device number')

args = parser.parse_args()
if args.device == 'gpu':
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")


arch='[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]'
net = eval(arch)
code = gen_code_from_list(net, node_num=int((len(net) / 4)))
genotype = translator([code, code], max_node=int((len(net) / 4)))
model = LaNet(C=128,num_classes=10,layers=24,auxiliary=True,genotype=genotype) 
# print(model)
pretrained_dict=torch.load('model/LaNet.pt')
model.load_state_dict(pretrained_dict['model_state_dict'])
model.fc_backup = model.classifier
classifier =  model.classifier
model.classifier = nn.Sequential()
model.to(device)
d_input = 2048
print('model load：successful')
data_classes = 10

nnstructurehouse = [30 for i in range(3)]
hanmodel = HouseholderNet(nnstructure=nnstructurehouse,input=d_input,output=data_classes)
hanmodel.to(device)
print(hanmodel)

criterion = nn.CrossEntropyLoss()


Xtrain = torch.load('feature/lanettraindata.pt')
Ytrain = torch.load('feature/lanettrainlabel.pt')
Xtest = torch.load('feature/lanettestdata.pt')
Ytest = torch.load('feature/lanettestlabel.pt')

print('Extracting feature succeeds.')

def test(model,data,target):
    model.eval()    
    correct = 0.
    total = 0.
    with torch.no_grad():    
        data,target=data.to(device),target.to(device)   
        output = model(data)
        loss = criterion(output, target)

        pred = torch.max(output.detach(), 1)[1]
        total += target.size(0)
        correct += (pred == target.detach()).sum().float().item()
    val_acc = correct / total
 
    return loss.item(), val_acc


pre_trainloss,pre_trainacc = test(classifier,Xtrain,Ytrain)
pre_testloss,pre_testacc = test(classifier,Xtest,Ytest)

print('LaNet training loss %9.8f'% pre_trainloss,' train acc %5.4f' % pre_trainacc, 
      ' test loss %7.6f'%pre_testloss , ' test acc %5.4f' %pre_testacc)

hanmodel = torch.load('model/han.pkl')
hanmodel.to(device)
trainloss,trainacc = test(hanmodel,Xtrain,Ytrain)
testloss,testacc = test(hanmodel,Xtest,Ytest)

print('HanNet training loss %9.8f'% trainloss,' train acc %5.4f' % trainacc, 
      ' test loss %7.6f'%testloss , ' test acc %5.4f' %testacc)
