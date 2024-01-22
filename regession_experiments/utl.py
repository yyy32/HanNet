# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import math
import pandas as pd
import random
from ann import ABS,HouseholderLayer


################### test function ##################
def test(model,X,Y0,criterion):
    model.eval()
    Yc = model(X)
    n = Yc.size(0)
    loss = criterion(Y0, Yc).detach().sqrt().item()
    return loss



################### train function ##################
def train(model,lr,Xtrain,Ytrain,Xtest,Ytest,maxiter,
          criterion,optimizer,scheduler,model_name,batch_size,outputwriter,interval=100,target='test'):
    model.train()
    best_trainloss = 1000000. 
    best_testloss = 1000000. 

    train_range = [i for i in range(Xtrain.size(0))]
    for iter in range(maxiter):
        loss = 0.
        model.train()
        batch_index = random.sample(train_range,batch_size)

        inputs = Xtrain[batch_index,:]
        if Ytrain.ndim ==1:
            labels = Ytrain[batch_index].long()
        else:
            labels = Ytrain[batch_index,:]
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if iter % interval == 0:
            trainloss = test(model,Xtrain,Ytrain,criterion)
            testloss = test(model,Xtest,Ytest,criterion)
            print(iter,'train loss %9.8f'% trainloss,
                 'test loss %7.6f'% testloss)  
            print(iter,'train loss %9.8f'% trainloss,
                'test loss %7.6f'% testloss,file=outputwriter)  

            if best_trainloss > trainloss and target.startswith('train'):
                best_trainloss = trainloss
                best_testloss = testloss
                torch.save(model.state_dict(), model_name+'.pkl')

            elif best_testloss > testloss and target.startswith('test'):
                best_trainloss = trainloss
                best_testloss = testloss
                torch.save(model.state_dict(), model_name+'.pkl')
                
 
        if np.isnan(trainloss):
            print('Warning: train loss is NaN.')
            return 1000000.,1000000.
    print('learning rate %4.3f' %lr,
          'best train loss %9.8f'% best_trainloss,
          'best test loss %7.6f'%best_testloss)
    return best_trainloss,best_testloss



###################  train and test data set spliting ##################
def datasplit(X0,Y0,rho=0.25):
    Dataset = TensorDataset(X0,Y0)
    train_size = int(len(Dataset) * rho)
    print(train_size)
    test_size = len(Dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(Dataset, [train_size, test_size])
    train_index = train_dataset.indices
    test_index = test_dataset.indices
    
    Xtrain = X0[train_index,:]
    Xtest = X0[test_index,:]
    if Y0.ndim ==1:
        Ytrain,Ytest = Y0[train_index],Y0[test_index]
    else: 
        Ytrain = Y0[train_index,:]
        Ytest = Y0[test_index,:]
    
    return Xtrain,Ytrain,Xtest,Ytest


####################################################
def modelparameter(model):
    return sum([param.nelement() for param in model.parameters() ])
