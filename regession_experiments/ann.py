# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np 

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
        self.eps = 1e-14
        
    def forward(self, input):
        # self.normedvector = self.vector/(self.vector.norm(p=2)+self.eps)
        # output = (input.matmul(self.normedvector))*self.normedvector.t()
        output = input - 2/( self.vector.t() @ self.vector) * (input @ self.vector) @ self.vector.t()
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output
    def extra_repr(self):
        return 'input_features={}, bias={}'.format(
            self.input_features, self.bias is not None
        )


class HouseholderNet(nn.Module):
    def __init__(self,nnstructure,input=2,output=2,activation='ABS',init='orth',bias=True):
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
        self.classifier = nn.Linear(nnstructure[-1],output)
        self.initialize()

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.inital(m.weight) 
            if isinstance(m, HouseholderLayer):
                try:
                    self.inital(m.vector) 
                except:
                    continue
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layers(self,in_channels,nnstructure):
        layers = []

        for idx,x in  enumerate(nnstructure):   
            if x != in_channels: 
                layers += [nn.Linear(in_channels, x,bias=True)]
            else:
                layers += [HouseholderLayer(x,bias=True)]
            if self.activation != -1 :
                layers += [self.activation()]

            in_channels = x
        if self.activation == -1:
            layers += [ABS(inplace=True)]
        return nn.Sequential(*layers)





class Net(nn.Module):
    def __init__(self,nnstructure,input=2,output=2,activation='ReLU',init='kaiming'):
        super(Net, self).__init__()
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
        elif init == 'eye':
            self.inital = nn.init.eye_

        self.feature = self._make_layers(input,nnstructure)  
        self.classifier = nn.Linear(nnstructure[-1],output)
        self.initialize()

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                if m.weight.size(0)==m.weight.size(1):
                    self.inital(m.weight) 
                else:
                    nn.init.orthogonal_(m.weight) 


    def _make_layers(self,in_channels,nnstructure):
        layers = []
        for x in nnstructure:
            layers += [nn.Linear(in_channels, x),
                       self.activation()]
            in_channels = x
        return nn.Sequential(*layers)



class BasicBlock(nn.Module):
    def __init__(self, channels,activation,bn=False):
        super(BasicBlock, self).__init__()

        self.channels = channels
        self.bn = bn
        
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
        layer = [nn.Linear(self.channels, self.channels)]
        if self.bn:
            layer += [nn.BatchNorm1d(num_features=self.channels)]
        return nn.Sequential(*layer)
    
class ResNet(nn.Module):

    def __init__(self,block=BasicBlock,channels=20,blocknumber=10,input=2,output=2,activation='ReLU',bn=False):
        super(ResNet, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU
        elif activation == 'ABS':
            self.activation = ABS

        self.bn = bn
        if self.bn:
            self.ln1 = nn.Sequential(nn.Linear(input,channels),
                                     nn.BatchNorm1d(num_features=channels),
                                     self.activation(inplace=True))
        else:
            self.ln1 = nn.Sequential(nn.Linear(input,channels),
                                     self.activation(inplace=True))
            
        self.feature = self._make_layers(block,channels,blocknumber)  
        self.classifier = nn.Linear(channels, output)
        self.initialize()
        
    def forward(self, x):
        out = self.ln1(x)
        out = self.feature(out)
        out = self.classifier(out)
        return out
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layers(self,block,channels,blocknumber):
        layers = []
        for stride in range(blocknumber):
            layers.append(block(channels,self.activation,self.bn))
        return nn.Sequential(*layers)
    


# nnstructure = [100 for i in range(20)]
# single_model = Net(nnstructure,input=16,output=1,activation='ABS',init='orth')
# print(single_model)