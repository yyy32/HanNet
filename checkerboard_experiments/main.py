# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from utl import datashow3D,correct,modelparameter,datashow2D
import random
from ann import Net,HouseholderNet,ResNet
import argparse
import os

parser = argparse.ArgumentParser(description='Neural network')
initial_options = ['xavier', 'kaiming','orth']
model_options = ['fcnet','hannet','resnet']
parser.add_argument('--model', default='hannet',choices=model_options)
parser.add_argument('--activation', type=str, default='ABS')
parser.add_argument('--initial', default='orth',choices=initial_options)
parser.add_argument('--use_bn', default=False, type=bool,
                    help='whether use batch normalization or net in FCNet')
parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 100)')
parser.add_argument('--maxiters', type=int, default=40000, 
                    help='number of iterations to train (default: 40000)')
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

#########show data ###############
if 0<-1:              
    datashow3D(torch.cat([Xtrain,Xtest]),torch.cat([Ytrain,Ytest]))
    datashow2D(Xtrain,Ytrain)
    datashow2D(torch.cat([Xtrain,Xtest]),torch.cat([Ytrain,Ytest]))


################### hidden layers ##################
nnstructurehouse = [30 for i in range(20)]
nnstructurenet = [100 for i in range(6)]

################### loss function ##################
batch_size = args.batch_size
maxiter = args.maxiters
criterion = nn.MSELoss()

# lrist = [0.001,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,1]
if args.model == 'hannet':
    lrist = [0.025,0.05,0.075,0.1]
else:
    lrist = [0.001,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25]
irun = 1

################### batch size and max iterations ##################
batch_size = args.batch_size
maxiter = args.maxiters

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

################### train function ##################
def train(model,model_name,lr):

    model.train()
    best_testacc = 0.
    train_range = [i for i in range(Xtrain.size(0))]

    for iter in range(maxiter):
        loss = 0.
        model.train()
        batch_index = random.sample(train_range,batch_size)
        inputs = Xtrain[batch_index,:]
        labels = Ytrain[batch_index,:]

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iter % 100 == 0:
            trainloss,traincor,testloss,testcor = test(model)
            if  iter % 2000 ==0 or iter==1:
                print('learning rate: ',lr ,
                      'iter %i /40000 '%iter,
                      'train loss: %9.8f'% trainloss,
                      'train acc: %5.4f' % traincor, 
                      'test loss: %7.6f'% testloss,
                      'test acc: %5.4f \n' %testcor)   
            if best_testacc < testcor:
                best_trainloss = trainloss
                best_trainacc = traincor
                best_testloss = testloss
                best_testacc = testcor
                torch.save(model.state_dict(), model_name+'.pkl')

        if np.isnan(trainloss):
            print('Warning: train loss is NaN.')
            return best_trainloss,best_trainacc,best_testloss,best_testacc
    print('learning rate %4.3f' %lr,
          'best train loss %9.8f'% best_trainloss,
          'best train acc %5.4f' % best_trainacc,
          'best test loss %7.6f'%best_testloss,
          'best test acc %5.4f' %best_testacc)
    return best_trainloss,best_trainacc,best_testloss,best_testacc



for i in range(irun):
    total_testacc = 0.
    for idx,ilr in enumerate(lrist):
        if args.model == 'fcnet':
            single_model = Net(nnstructurenet,input=2,activation=args.activation,
                        bn=args.use_bn,init=args.initial)
        elif args.model == 'hannet':    
            single_model = HouseholderNet(nnstructurehouse,activation=args.activation,input=2,
                         init=args.initial)
        elif args.model == 'resnet':    
            single_model = ResNet(channels=100,blocknumber=10,activation=args.activation)
        if i==0 and idx==0:        
            print(single_model)
            print(modelparameter(single_model))
            print('learning rate:',lrist)

        model_name = os.path.join('model/','irun'+str(i)+'lr_'+str(ilr)+args.model)
        
    ################### optimizer and step sceduler ##################
        optimizer = optim.SGD(single_model.parameters(), lr=ilr,momentum=0.9)      
        scheduler = MultiStepLR(optimizer, milestones=[0.5*maxiter,0.7*maxiter,0.9*maxiter],gamma=0.2)
        
    ################### model training ##################
        print('run:',i, ',initial learning rate:',ilr)
        trainloss,traincor,testloss,testcor = train(single_model,model_name,ilr)

        if total_testacc < testcor:
            total_trainloss = trainloss
            total_trainacc = traincor
            total_testloss = testloss
            total_testacc = testcor
            total_lr = ilr
    print('run: ',i,'best train loss %9.8f'% total_trainloss,'best train acc %5.4f' % total_trainacc, 
          'best test loss %7.6f'%total_testloss , 'best test acc %5.4f' %total_testacc,
          'best learning rate %4.3f' %total_lr)



