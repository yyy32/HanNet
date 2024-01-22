# -*- coding: utf-8 -*-

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from utl import modelparameter,datasplit,train,test
from ann import Net,HouseholderNet,ResNet
import argparse
import os
import time
import random

parser = argparse.ArgumentParser(description='Neural network')
initial_options = ['xavier', 'kaiming','orth']

model_options = ['fcnet','hannet','resnet']
parser.add_argument('--model', default='fcnet',choices=model_options)
parser.add_argument('--act', type=str, default='ReLU')
parser.add_argument('--initial', default='kaiming',choices=initial_options)
parser.add_argument('--width', type=int, default=200)
parser.add_argument('--depth', type=int, default=20)

parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 100)')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--rho', type=float, default=0.8)
parser.add_argument('--prob', type=str, default='calhousing')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 0)')
parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

if args.device == 'gpu':
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


experiment_name = '{}{}width{}depth{}seed{}{}'.format(args.model,args.act,args.width,args.depth,args.seed,args.rho)


###################prepare data ##################
storage_folder = 'file/'+args.prob+'/'
model_storage_folder = storage_folder
dataset = np.loadtxt('dataset/'+args.prob+'.csv', delimiter=',') 
X0,Y0 = torch.tensor(dataset[:,0:-1]).float(), torch.tensor(dataset[:,-1]).float()
Y0 =Y0.view(1,-1).t()

X0 = X0[:,X0.std(dim=0)>0]
X0 = (X0- X0.mean(dim=0))/X0.std(dim=0)
print('dataset:',X0.size())


Xtrain,Ytrain,Xtest,Ytest = datasplit(X0,Y0,rho=args.rho)
Xtest,Ytest,Xval,Yval = datasplit(Xtest,Ytest,rho=0.6) 

Xtrain,Ytrain = Xtrain.to(device), Ytrain.to(device)
Xtest,Ytest = Xtest.to(device), Ytest.to(device)
Xval,Yval = Xval.to(device), Yval.to(device)


#%%
################### loss function ##################
criterion = nn.MSELoss()

################### batch size and max iterations ##################
batch_size = args.batch_size
maxiter = int(args.epoch*Xtrain.size(0)/args.batch_size)
print(maxiter)
if args.opt == 'sgd':
    lrist = [0.025,0.05,0.075]
elif args.opt == 'adam':
    lrist = [0.001,0.003]

################### model ##################
def constructmodel(nn):
    nnstructure = [args.width for i in range(args.depth)]
    if nn == 'fcnet':
        single_model = Net(nnstructure,input=Xtrain.size(1),output=Ytrain.size(1),activation=args.act,init=args.initial)
    elif nn == 'hannet':    
        single_model = HouseholderNet(nnstructure,input=Xtrain.size(1),output=Ytrain.size(1),activation=args.act,init=args.initial)
    single_model.to(device)            
    return single_model


def constructoptimizer(model,opt,lr):
    if opt =='sgd':
        optimizer = optim.SGD(single_model.parameters(), lr=lr,momentum=0.9)    
    elif opt =='adam':
        optimizer = optim.Adam(single_model.parameters(), lr=lr)
    return optimizer



###############################################
runs = 5
for irun in range(runs):
    irun_experiment_name = experiment_name+'run'+str(irun)
    outputfile = storage_folder+irun_experiment_name
    outputwriter = open(outputfile+'.txt', 'a+')

    total_valacc = 0.
    for idx,ilr in enumerate(lrist):
        model_name = os.path.join(model_storage_folder,irun_experiment_name+'lr_'+str(ilr))
        single_model = constructmodel(args.model)
        if idx==0:        
            outputwriter.write('model_param  '+str(modelparameter(single_model))+'\n')
            print(single_model)
            print(modelparameter(single_model))
            print('learning rate:',lrist)
            
        optimizer = constructoptimizer(single_model,args.opt,ilr)
        print(optimizer)
        scheduler = MultiStepLR(optimizer, milestones=[0.5*args.epoch,0.7*args.epoch],gamma=0.2)

        print('initial learning rate:',ilr)
        trainloss,valloss =  train(model=single_model,
                                    lr=ilr,
                                    Xtrain=Xtrain,
                                    Ytrain=Ytrain,
                                    Xtest=Xval,
                                    Ytest=Yval,
                                    maxiter=maxiter,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    batch_size=args.batch_size,
                                    target='test',
                                    model_name=model_name,
                                    interval= int(Xtrain.size(0)/args.batch_size),
                                    outputwriter=outputwriter)
        print('best  train loss %9.8f'% trainloss,
            'val loss %7.6f'% valloss,file=outputwriter)  
    outputwriter.close()
