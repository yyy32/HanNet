# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from utl import ABS,HouseholderLayer

class HouseholderNet(nn.Module):

    def __init__(self,nnstructure,input=2,output=1):
        super(HouseholderNet, self).__init__()
        self.activation = ABS
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, HouseholderLayer):
                self.inital(m.vector) 
    def _make_layers(self,in_channels,nnstructure):
        layers = []

        for idx,x in  enumerate(nnstructure):   
            if x != in_channels:
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
                if self.inital is nn.init.orthogonal_ and self.activation is nn.ReLU:
                    self.inital(m.weight,gain=math.sqrt(2)) 
                else:
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

