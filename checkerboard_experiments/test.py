# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from utl import datashow3D,correct,problem,datasplit,modelparameter,datashow2D
import random
from ann import Net,ResNet,HouseholderNet
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Model Testing')
model_options = ['fcnet','hannet','resnet']
parser.add_argument('--model', default='hannet',choices=model_options)
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

###################prepare data ##################

trainset = np.loadtxt('dataset/trainset.txt', delimiter=' ') 
testset = np.loadtxt('dataset/testset.txt', delimiter=' ') 

Xtrain,Ytrain = torch.tensor(trainset[:,0:2]).float(), torch.tensor(trainset[:,2:4]).float()
Xtest,Ytest = torch.tensor(testset[:,0:2]).float(), torch.tensor(testset[:,2:4]).float()  
X0,Y0 = torch.cat([Xtrain,Xtest]),torch.cat([Ytrain,Ytest])

if 0<-1:              
    datashow3D(X0,Y0)
    datashow2D(X0,Y0)
    
#%%
################### hidden layers ##################
nnstructurehouse = [30 for i in range(20)]
nnstructurenet = [100 for i in range(6)]
criterion = nn.MSELoss()

################### test function ##################
def test(model):
    model.eval()
    Yctrain = model(Xtrain)
    trainloss = criterion(Yctrain, Ytrain).detach().item()
    traincor = correct(Yctrain.detach(),Ytrain.detach())
    
    Yctest = model(Xtest)
    testloss = criterion(Yctest, Ytest).detach().item()
    testcor = correct(Yctest.detach(),Ytest.detach()) 
    
    return trainloss,traincor,testloss,testcor


if args.model == 'hannet':
    nnstructurehouse = [30 for i in range(20)]
    model = HouseholderNet(nnstructurehouse,input=2,activation='ABS')
    model.load_state_dict(torch.load('model/hannet.pkl')) 
elif args.model == 'fcnet':
    nnstructurenet = [100,100,100,100,100,100]
    model = Net(nnstructurenet,input=2,activation='ReLU',bn=False)
    model.load_state_dict(torch.load('model/net.pkl'))
elif args.model == 'resnet':
    model = ResNet(channels=100,blocknumber=10)
    model.load_state_dict(torch.load('model/resnet.pkl'))
    

trainloss,traincor,testloss,testcor = test(model)
print('train loss %9.8f'% trainloss,'train acc %5.4f' % traincor, 
  'test loss %7.6f'% testloss, 'test acc %5.4f \n' %testcor)   
Yc = model(X0)
datashow3D(X0,Yc)
datashow2D(X0,Yc)

