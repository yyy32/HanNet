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

def modelparameter(model):
    return sum([param.nelement() for param in model.parameters() ])


################### the difficulty of the problem ##############3###
def problem(t=4):
    x = np.linspace(-1, 1, num=81)
    Z,U = np.meshgrid(x,x)
    X0 = np.concatenate(([U.flatten()],[Z.flatten()]),axis=0).transpose()
    
    X0 = torch.tensor(X0).float()
    B = np.kron([[0, 1] * t, [1, 0] * t] * t, np.ones((int(36/t), int(36/t))))

    T = np.zeros([81,81])
    T[4:76,4:76] = B
        
    Y0 = T.flatten()
    Y0 = np.concatenate(([Y0.flatten()],[Y0.flatten()]),axis=0).transpose()
    Y0 = torch.tensor(Y0).float()
    return X0,Y0

################### data noising ##############3###
def datanoise(Ytrain):
    Ytrainnoise = (torch.rand(Ytrain.size(0))-0.9).clamp(0).sign().view(1,-1).t()    
    Ytrain = (Ytrain+Ytrainnoise)%2  
    return Ytrain

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

################### absolute-value as an activation function ##################
class ABS(nn.Module):
    def __init__(self,inplace=True):
        super(ABS, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        if self.inplace:
            return x.abs_()
        else:
            return x.abs()

###################  2D-data figure##################
def datashow2D(Xtrain,Ytrain):
    fig = plt.figure('2D')
    Ytrain[Ytrain.mean(dim=1)>0.5] =1
    Ytrain[Ytrain.mean(dim=1)<0.5] =0
    Xtrain1 = Xtrain[Ytrain.mean(dim=1) == 1,:]
    Xtrain0 = Xtrain[Ytrain.mean(dim=1) == 0,:]
    
    x2 = Xtrain0[:,0].detach().numpy().reshape((1,-1))[0]
    y2 = Xtrain0[:,1].detach().numpy().reshape((1,-1))[0]
    x1 = Xtrain1[:,0].detach().numpy().reshape((1,-1))[0]
    y1 = Xtrain1[:,1].detach().numpy().reshape((1,-1))[0]
    
    plt.scatter(x1,y1,color='orangered',s=15,marker='s',linewidths=0.)
    plt.scatter(x2,y2,color='lightskyblue',s=15,marker='s',linewidths=0.)
    plt.axis('square')
    plt.show()

###################  3D-data figure##################
def datashow3D(X0,Y0):
    fig = plt.figure('3D')
    if 0 > -1:
        ax = fig.gca(projection='3d')
        df = pd.DataFrame({'x': X0[:,0].detach().numpy().reshape((1,-1))[0],
                           'y': X0[:,1].detach().numpy().reshape((1,-1))[0],
                           'z': Y0.mean(dim=1).detach().numpy().reshape((1,-1))[0]})
        
        surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.rainbow,
                           linewidth=1.4, antialiased=True)
        plt.xticks([])  
        plt.yticks([])
        plt.show()
        
    else:
        ax = fig.gca(projection='3d')
        x = X0[:,0].detach().numpy().reshape((1,-1))[0]
        y = X0[:,1].detach().numpy().reshape((1,-1))[0]
        n = int(np.sqrt(np.size(x)))
        x = x.reshape((n,n))
        y = y.reshape((n,n))
        z = Y0.mean(dim=1).detach().numpy().reshape((1,-1))[0]
        z = z.reshape((n,n))
        surf = ax.plot_surface(x, y, z, cmap=cm.rainbow,
                           linewidth=1.4, antialiased=True)
        plt.xticks([])  
        plt.yticks([])
        plt.show()
#    
################### Y vs Y0 correcting ##################
def correct(Y,Y0):
    n = Y.size(0)
    Y =  Y.mean(dim=1) 
    Y[Y>0.5]=1
    Y[Y<0.5]=0
    Y0 = Y0.mean(dim=1)   
    cor = (Y0 == Y).sum().item()/n
    return cor



