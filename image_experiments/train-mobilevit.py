import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR

import torch.nn as nn
import numpy as np
import os
import argparse
import random
import time

from mobilevit import mobilevit_xs,mobilevit_xxs,mobilevithan_xs,mobilevithan_xxs

from DatasetProcess import data_loader
from utl import CELoss,Evalute,EvaluteTop5,EpochTrain




#%%
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--augment', type=int, default=1)
parser.add_argument('--model', type=str, default='tiny')
parser.add_argument('--depth', type=int, default=10)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=512,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epoch', type=int, default=300,
                    help='number of epochs to train (default: 300)')
parser.add_argument('--reg_alpha', type=float, default=0.)
parser.add_argument('--t', type=float, default=1.)

parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='gpu device number')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 0)')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.device == 'gpu':
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")

train_loader,test_loader,num_classes = data_loader(args.dataset,args.augment,args.batch_size,image_size=256)

filehead = './filevitmobile/{}augment{}'.format(args.dataset,args.augment)

if args.model == 'xs':
    model =  mobilevit_xs(num_classes)
elif args.model == 'xxs':
    model =  mobilevit_xxs(num_classes)
elif args.model == 'hxs':
    print('ok')
    model =  mobilevithan_xs(num_classes)
elif args.model == 'hxxs':
    print('ok')
    model =  mobilevithan_xxs(num_classes)
    
filename = '{}depth{}randseed{}epoch{}{}t{}bs{}lr{}'.format(args.model,args.depth,args.seed,args.epoch,args.optim,args.t,args.batch_size,args.lr)
print(filename)
    
if not os.path.exists(filehead):
    os.makedirs(filehead)
    
filewrite = open(filehead+'/'+str(time.ctime())+filename+'.txt','w')

model.to(device)
param = sum([param.nelement() for param in model.parameters()])
print(model)
print(model,file=filewrite)
print(param)
print(param,file=filewrite)

if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005)
elif args.optim == 'sgdm':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=0.0005)
elif args.optim == 'adam':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

print(optimizer)

scheduler = MultiStepLR(optimizer, milestones=[0.7*args.epoch,0.9*args.epoch],gamma=0.2)


print(scheduler)
print(scheduler,file=filewrite)
criterion = CELoss()

best_testacc = 0.
for epoch in range(1, args.epoch + 1):
    trainloss, trainacc =  EpochTrain(epoch,model,
                                        optimizer,
                                        train_loader,
                                        criterion,
                                        device=device,
                                        dataset=args.dataset)
    valloss, valacc = Evalute(model,
                             test_loader,
                             criterion,
                             device=device)
    
    if args.dataset == 'imagenet32':
        torch.save(model.state_dict(),modelfile_name+'/epoch'+str(epoch)+'.pkl')

    filewrite.write(' epoch '+str(epoch)+
                    ' trainloss '+ str(trainloss)+
                    ' valloss ' +str(valloss)+
                    ' trainacc '+str('%.4f'%trainacc)+
                    ' test_acc '  +str('%.4f'%valacc)+'\n')

    if np.isnan(trainloss):
        break
    if best_testacc < valacc:
        best_trainloss = trainloss
        best_trainacc = trainacc
        best_valloss = valloss
        best_testacc = valacc
    scheduler.step()

print('best train loss %9.8f'% best_trainloss,'best train acc %5.4f' % best_trainacc, 
      'best test loss %7.6f'%best_valloss , 'best test acc %5.4f' %best_testacc)

filewrite.close()

