import torch
import torch.nn as nn
import numpy as np
import math


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
        self.eps = 1e-16
    def forward(self, input):
        self.weight = self.I- 2*(self.vector*self.vector.t()+self.eps)/(self.vector.pow(2).sum()+self.eps)
        output = input.mm(self.weight)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output

    def extra_repr(self):
        return 'input_features={}, bias={}'.format(
            self.input_features, self.bias is not None
        )

class Hanblock(nn.Module):
    def __init__(self,in_channels,nnstructure,init = 'orth'):
        super(Hanblock, self).__init__()
        self.features = self._make_layers(in_channels,nnstructure)
    def forward(self, x):
        out = self.features(x)
        return out
    def _make_layers(self,in_channels,nnstructure):
        layers = []        
        for idx,x in  enumerate(nnstructure):   
            if x != in_channels:
                layers += [nn.Linear(in_channels, x,bias=True),
                           ABS(inplace=True)]
            else:
                layers += [HouseholderLayer(x,bias=True),
                           ABS(inplace=True)]
            in_channels = x
        return nn.Sequential(*layers)
    

class HouseholderNet(nn.Module):

    def __init__(self,nnstructure,block=Hanblock,input=784,output=10,init='orth'):
        super(HouseholderNet, self).__init__()

        if init == 'xavier':
            self.inital = nn.init.xavier_normal_
        elif init == 'kaiming':
            self.inital = nn.init.kaiming_normal_
        elif init == 'orth':
            self.inital = nn.init.orthogonal_
        self.feature = block(input,nnstructure,init=self.inital)  
        self.classifer = nn.Linear(nnstructure[-1],output)
        self.initialize()

    def forward(self, x):
        x = x.reshape(x.size(0) , -1)
        x = self.feature(x)
        x = self.classifer(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.inital(m.weight) 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, HouseholderLayer):
                self.inital(m.vector) 
