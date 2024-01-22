# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from utl import modelparameter,datasplit
from torch.utils.data import TensorDataset, DataLoader
import random
from ann import Net,HouseholderNet
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Regression')
model_options = ['fcnet','hannet']
dataset_options = ['elevators','calhousing']

parser.add_argument('--model', default='fcnet',choices=model_options)
parser.add_argument('--prob', default='elevators',choices=dataset_options)
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of iterations to train (default: 300)')
parser.add_argument('--rho', type=float, default=0.8, 
                    help='train set/ data set (default: 0.8)')
parser.add_argument('--use_bn', default=False, type=bool,
                    help='whether use batch normalization')

args = parser.parse_args()

torch.manual_seed(1)    
random.seed(1)  
device = torch.device("cpu")


batch_size = 100

###################prepare data ##################
dataset = np.loadtxt('dataset/'+str(args.prob)+'.csv', delimiter=',') 
X0,Y0 = torch.tensor(dataset[:,0:-1]).float(), torch.tensor(dataset[:,-1]).float()
Y0 =Y0.view(1,-1).t()
X0 = (X0- X0.mean(dim=0))/X0.std(dim=0)
print('dataset:',X0.size())

###################prepare model ##################
nnstructurehouse = [200 for i in range(20)] 
nnstructurenet = [200 for i in range(5)]

###################prepare loss function ##################
criterion = nn.MSELoss()

###################train and test ##################

def test(model,X,Y):
    model.eval()
    X,Y = X.to(device),Y.to(device)
    Yc = model(X)
    Y = Y.mean(dim=1)
    
    Yc = Yc.mean(dim=1)
    loss = criterion(Yc, Y).detach().sqrt()/(Y.max()-Y.min())
    
    return loss.item()

def train(model):
    model.train()
    best_testloss = 1e+10
    best_trainloss = 1e+10
    for i in range(args.epochs):
        for idx,[inputs, labels] in enumerate(train_loader):
            loss = 0.
            model.train()
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        trainloss = test(model,Xtrain,Ytrain)
        testloss = test(model,Xtest,Ytest)
        print('epoch: %4.1f' %i,'train loss %9.8f'% trainloss,'test loss %7.6f'% testloss)   
        filename.write(' trainloss %7.6f testloss %7.6f \n'
                       %(trainloss,testloss))
        if best_testloss > testloss:
            best_testloss = testloss
            best_model = model
        if best_trainloss > trainloss:
            best_trainloss = trainloss
    print( 'best train loss %9.8f'% best_trainloss,
          'best test loss %7.6f'%best_testloss)

    return best_model,best_trainloss,best_testloss

tic = datetime.now()
run = 5


for irun in range(run):
    fname = 'prob-'+str(args.prob)+'model'+args.model+str(nnstructurenet)+'-rho'+str(args.rho)
    filename = open('file/irun-'+str(irun)+fname+'.txt', 'w')

    Xtrain,Ytrain,Xtest,Ytest = datasplit(X0,Y0,rho=args.rho)
    train_loader = DataLoader(dataset=TensorDataset(Xtrain, Ytrain), batch_size=batch_size, shuffle=True) 
    testfeature_loader = DataLoader(dataset=TensorDataset(Xtest, Ytest), batch_size=batch_size, shuffle=True) 

    if args.model == 'fcnet':
        model = Net(nnstructurenet,input=X0.size(1),output=Y0.size(1),activation='ReLU',
                    bn=args.use_bn,init='kaiming')
    elif args.model == 'hannet':    
        model=HouseholderNet(nnstructurehouse,input=X0.size(1),
                     output=Y0.size(1))
    model.to(device)
    if irun==0:
        print(model)
        print(modelparameter(model))
        
###################prepare optimizer and loss function ##################
    param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(param, lr=0.001) 
    model,trainloss,testloss = train(model)

    filename.write('irun %2.1f thebesttrainloss %7.6f testloss %7.6f \n'
                   %(irun,trainloss,testloss))
    filename.close()
