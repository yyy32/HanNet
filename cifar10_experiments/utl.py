# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:33:23 2018
"""
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np


def normalCE(output,label):
    normoutput = output.norm(p=2, dim=1)
    print(normoutput)
    print(output/normoutput.view(-1, 1) )
    scaledoutput = output/normoutput.view(-1, 1)
    z = nn.CrossEntropyLoss(scaledoutput,label)
    return z


def datasplit(setname):
    if setname == 'mnist':
        num_classes = 10
        traindataset = datasets.MNIST(root='data/mnist', train=True, download=False,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1,))
                                ]))
        testdataset = datasets.MNIST(root='data/mnist', train=False, download=False,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1,))

                                ]))
    elif setname == 'fashionmnist':
        num_classes = 10
        traindataset = datasets.FashionMNIST(root='data/FashionMNIST', train=True, download=False,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))
        testdataset = datasets.FashionMNIST(root='data/FashionMNIST', train=False, download=False,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    elif setname == 'emnist':
        num_classes = 47 
        traindataset = datasets.EMNIST(root='data/EMNIST', train=True, download=True,split='balanced',
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))
        testdataset = datasets.EMNIST(root='data/EMNIST', train=False, download=True,split='balanced',
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    elif setname == 'cifar10':
        num_classes = 10
        traindataset = datasets.CIFAR10(root='data/',train=True,download=False, 
                                 transform=transforms.Compose([ 
                                 transforms.RandomCrop(32, padding=4),  
                                 transforms.Scale(32),  
                                 transforms.RandomHorizontalFlip(),  
                                 transforms.ToTensor(),  
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
                                ]))
        testdataset = datasets.CIFAR10(root='data/',train=False,download=False, 
                                 transform=transforms.Compose([ 
                                 transforms.ToTensor(),  
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]))
    elif setname == 'cifar100':
        num_classes = 100
        traindataset = datasets.CIFAR100(root='data/',train=True,download=False, 
                                 transform=transforms.Compose([ 
                                 transforms.RandomCrop(32, padding=4),  
                                 transforms.Scale(32),  
                                 transforms.RandomHorizontalFlip(),  
                                 transforms.ToTensor(),  
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
                                ]))
        testdataset = datasets.CIFAR100(root='data/',train=False,download=False, 
                                 transform=transforms.Compose([ 
                                 transforms.ToTensor(),  
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]))
    return traindataset,testdataset,num_classes

def dataset_loader(setname,batch_size):
    traindata,testdata,num_classes = datasplit(setname)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
        
    return train_loader, test_loader, num_classes   

    
    
