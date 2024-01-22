# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from utl import ABS
import math

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
#        self.register_buffer('I', torch.eye(input_features))
        self.eps = 1e-14
        
    def forward(self, input):
        self.normedvector = self.vector/(self.vector.norm(p=2)+self.eps)
#        self.weight = self.I- 2*(self.normedvector*self.normedvector.t())
#        output = input.mm(self.weight)
        output = input - 2*(input.mm(self.normedvector))*self.normedvector.t()
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output

    def extra_repr(self):
        return 'input_features={}, bias={}'.format(
            self.input_features, self.bias is not None
        )

class HouseholderNet(nn.Module):
    def __init__(self,nnstructure,input=2,output=2,activation='ABS',init='orth'):
        super(HouseholderNet, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU
        elif activation == 'ABS':
            self.activation = ABS

        if init == 'xavier':
            self.inital = nn.init.xavier_normal_
        elif init == 'kaiming':
            self.inital = nn.init.kaiming_normal_
        elif init == 'orth':
            self.inital = nn.init.orthogonal_
            
        self.feature = self._make_layers(input,nnstructure)  
        self.classifer = nn.Linear(nnstructure[-1],output)
        self.initialize()

    def forward(self, x):
        x = self.feature(x)
        x = self.classifer(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.inital(m.weight) 
            elif isinstance(m, HouseholderLayer):
                self.inital(m.vector) 
    def _make_layers(self,in_channels,nnstructure):
        layers = []

        for idx,x in  enumerate(nnstructure):   
            if x != in_channels or idx==1 or idx==len(nnstructure)-1:
                layers += [nn.Linear(in_channels, x,bias=True),
                           self.activation(inplace=True)]
            else:
                layers += [HouseholderLayer(x,bias=True),
                           self.activation(inplace=True)]
            in_channels = x
        return nn.Sequential(*layers)

class Net(nn.Module):

    def __init__(self,nnstructure,input=2,output=2,activation='ReLU',init='kaiming',bn=True):
        super(Net, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU
        elif activation == 'ABS':
            self.activation = ABS
        elif activation == 'LReLU':
            self.activation = nn.LeakyReLU
        elif activation == 'ELU':
            self.activation = nn.ELU    
            
        if init == 'xavier':
            self.inital = nn.init.xavier_normal_
        elif init == 'kaiming':
            self.inital = nn.init.kaiming_normal_
            
        self.feature = self._make_layers(input,nnstructure,bn)  
        self.classifer = nn.Linear(nnstructure[-1],output)
        self.initialize()

    def forward(self, x):
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
    def _make_layers(self,in_channels,nnstructure,bn):
        layers = []

        for x in nnstructure:
            if bn:
                layers += [nn.Linear(in_channels, x),
                           nn.BatchNorm1d(num_features=x),
                           self.activation(inplace=True)]
            else:        
                layers += [nn.Linear(in_channels, x),
                           self.activation(inplace=True)]
            in_channels = x
        return nn.Sequential(*layers)



class BasicBlock(nn.Module):
    def __init__(self, channels,activation):
        super(BasicBlock, self).__init__()

        self.channels = channels
        self.ln1 = self.completelayer()
        self.ln2 = self.completelayer()
        self.activation = activation(inplace=True)
        self.downsample = lambda x: x

    def forward(self, x):
        residual = self.downsample(x)
        out = self.ln1(x)
        out = self.activation(out)
        out = self.ln2(out)
        out += residual
        out = self.activation(out)
        return out

    def completelayer(self):
        layer = nn.Sequential(nn.Linear(self.channels, self.channels),
                              nn.BatchNorm1d(num_features=self.channels))

        return layer
    
class ResNet(nn.Module):

    def __init__(self,block=BasicBlock,channels=20,blocknumber=10,input=2,output=2,activation='ReLU'):
        super(ResNet, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU
        elif activation == 'ABS':
            self.activation = ABS
        elif activation == 'LReLU':
            self.activation = nn.LeakyReLU
        elif activation == 'ELU':
            self.activation = nn.ELU    
        
        self.ln1 = nn.Sequential(nn.Linear(input,channels),
                                 nn.BatchNorm1d(num_features=channels),
                                 self.activation(inplace=True))

        self.feature = self._make_layers(block,channels,blocknumber)  
        self.classifer = nn.Linear(channels, output)
        self.initialize()

    def forward(self, x):
        out = self.ln1(x)
        out = self.feature(out)
        out = self.classifer(out)
        return out
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layers(self,block,channels,blocknumber):
        layers = []
        for stride in range(blocknumber):
            layers.append(block(channels,self.activation))
        return nn.Sequential(*layers)
    



