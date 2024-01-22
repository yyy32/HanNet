# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import math

def modelparameter(model):
    return  sum([param.nelement() for param in model.parameters()])
                
# ###################Absolute as an activation function##################

class ABS(nn.Module):
    def __init__(self,inplace=True):
        super(ABS, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        if self.inplace:
            return x.abs_()
        else:
            return x.abs()


class HouseholderLayer(nn.Module):
    def __init__(self, input_features, bias=True):
        super(HouseholderLayer, self).__init__()
        self.input_features = input_features

        self.vector = nn.Parameter(torch.Tensor(input_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_features))
            bound = 1 / math.sqrt(input_features)

            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.register_buffer('I', torch.eye(input_features))
        self.eps = 1e-14
    def forward(self, input):
        self.normedvector = self.vector/(self.vector.pow(2).sum().sqrt()+self.eps)
        self.weight = self.I- 2*(self.normedvector*self.normedvector.t())

        output = input.mm(self.weight)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output

    def extra_repr(self):
        return 'input_features={}, bias={}'.format(
            self.input_features, self.bias is not None
        )

###################Spliting train and test data##################
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





