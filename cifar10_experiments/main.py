# -*- coding: utf-8 -*-
"""
"""
from tqdm import tqdm
import argparse
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
from ann import HouseholderNet
from LaNet import NetworkCIFAR as LaNet
from nasnet_set import *

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='input batch size for training (default: 64)')
parser.add_argument('--maxiter', type=int, default=20000,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.25, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='gpu device number')

args = parser.parse_args()
if args.device == 'gpu':
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)


arch='[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]'
net = eval(arch)
code = gen_code_from_list(net, node_num=int((len(net) / 4)))
genotype = translator([code, code], max_node=int((len(net) / 4)))
model = LaNet(C=128,num_classes=10,layers=24,auxiliary=True,genotype=genotype) 
# print(model)
pretrained_dict=torch.load('model/LaNet.pt')
model.load_state_dict(pretrained_dict['model_state_dict'])
model.fc_backup = model.classifier
classifier =  model.classifier
model.classifier = nn.Sequential()
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
model.to(device)
d_input = 2048
print('model load：successful')

nnstructurehouse = [30 for i in range(3)]
hanmodel = HouseholderNet(nnstructure=nnstructurehouse,input=d_input,output=10)
hanmodel.to(device)
print(hanmodel)

optimizer = optim.SGD(hanmodel.parameters(), lr=args.lr, weight_decay=0.0001, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[0.5*args.maxiter,0.7*args.maxiter,0.9*args.maxiter],gamma=0.2)
criterion = nn.CrossEntropyLoss()


def extract(model,data_loader):
    model.eval()
    F = []
    L = []
    for batch_idx, (data, target) in enumerate(data_loader):

        data, target = data.to(device), target.to(device)
        features,_ = model(data)
        features = features.view(features.size(0), -1).cpu().detach()
        target = target.cpu().detach()
        
        if F == []:
            F = features
            L = target
        else:
            F = torch.cat((features,F),dim=0)
            L =  torch.cat([target,L])
        print(F.size())
    return F,L


if 0<-1:
    traindataset = datasets.CIFAR10(root='data/',train=True,download=True, 
                         transform=transforms.Compose([ 
                         transforms.ToTensor(),  
                         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  
                        ]))
    testdataset = datasets.CIFAR10(root='data/',train=False,download=True, 
                             transform=transforms.Compose([ 
                             transforms.ToTensor(),  
                             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  
                            ]))
    data_classes = 10
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=384, shuffle=True, num_workers=4,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=384, shuffle=True, num_workers=4,pin_memory=True)
    Xtrain,Ytrain = extract(model,train_loader)
    Xtest,Ytest = extract(model,test_loader)
else:
    Xtrain = torch.load('feature/lanettraindata.pt')
    Ytrain = torch.load('feature/lanettrainlabel.pt')
    Xtest = torch.load('feature/lanettestdata.pt')
    Ytest = torch.load('feature/lanettestlabel.pt')

print('Extracting feature succeeds.')



def test(model,data,target):
    model.eval()    
    correct = 0.
    total = 0.
    with torch.no_grad():    
        data,target=data.to(device),target.to(device)   
        output = model(data)
        loss = criterion(output, target)

        pred = torch.max(output.detach(), 1)[1]
        total += target.size(0)
        correct += (pred == target.detach()).sum().float().item()
    val_acc = correct / total
    return loss.item(), val_acc


def train(model):
    best_testacc = 0.
    train_range = [i for i in range(Xtrain.size()[0])]
    for iter in range(args.maxiter):
        model.train()
        loss = 0.
        model.train()
        batch_index = random.sample(train_range,args.batch_size)
        
        inputs = Xtrain[batch_index,:]
        labels = Ytrain[batch_index]
        inputs,labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iter % 100 == 0:
            trainloss,traincor = test(model,Xtrain,Ytrain)
            testloss,testcor = test(model,Xtest,Ytest )

            print(iter,'train loss %9.8f'% trainloss,'train acc %5.4f' % traincor,
                  'test loss %7.6f'% testloss, 'test acc %5.4f \n' %testcor)   
            if best_testacc < testcor:
                best_trainloss = trainloss
                best_trainacc = traincor
                best_testloss = testloss
                best_testacc = testcor
#                 torch.save(model,'model/'+fname+'.pkl')
        if np.isnan(trainloss):
            break
    print(iter,'best train loss %9.8f'% best_trainloss,'best train acc %5.4f' % best_trainacc, 
          'best test loss %7.6f'%best_testloss , 'best test acc %5.4f' %best_testacc)
    return best_trainloss,best_trainacc,best_testloss,best_testacc


pre_trainloss,pre_trainacc = test(classifier,Xtrain,Ytrain)
pre_testloss,pre_testacc = test(classifier,Xtest,Ytest)

print('LaNet train loss %9.8f'% pre_trainloss,'LaNet train acc %5.4f' % pre_trainacc, 
      'LaNet test loss %7.6f'%pre_testloss , 'LaNet test acc %5.4f' %pre_testacc)


trainloss,traincor,testloss,testcor = train(hanmodel)

    