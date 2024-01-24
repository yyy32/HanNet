import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np

import os
from tqdm import tqdm
import argparse
import random
import time

from hanmixer import  CnnStemHanMixer
from resnet import ResNet

from DatasetProcess import data_loader
from utl import CELoss,Evalute,EvaluteTop5,EpochTrain



#%%
parser = argparse.ArgumentParser(description='train')

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--augment', type=int, default=1)
parser.add_argument('--model', type=str, default='resnet32')
parser.add_argument('--channel', type=int, default=1024)
parser.add_argument('--hanblock', type=int, default=12)
parser.add_argument('--mixerblock', type=int, default=0)
parser.add_argument('--depth', type=int, default=10)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch-size', type=int, default=512,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epoch', type=int, default=300,
                    help='number of epochs to train (default: 300)')
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

train_loader,test_loader,num_classes = data_loader(args.dataset,args.augment,args.batch_size)

if args.dataset.startswith('stl'):
    image_size=96
    stride = [1,2,2,2]
    tokens_mlp_dim = 144
    patch_size = 8
else:
    image_size=32
    stride = [1,2,1,2]
    tokens_mlp_dim = 64
    patch_size = 4 
    
filehead = './file/{}augment{}'.format(args.dataset,args.augment)

if args.model == 'hanmixer':
    model = CnnStemHanMixer(num_classes=num_classes,
                            num_mixerblocks=args.mixerblock,
                            num_hanblocks=args.hanblock, 
                            patch_size=patch_size,
                            hidden_dim=args.channel, 
                            tokens_mlp_dim=tokens_mlp_dim,
                            channels_mlp_dim=args.channel,
                            image_size=image_size,
                            stem_stride=stride)
    filename = 'hanmixer'+args.optim+'_mixerblock'+str(args.mixerblock)+'_hanblock'+str(args.hanblock)+'randseed'+str(args.seed)+'batchsize'+str(args.batch_size)+'channel'+str(args.channel)
elif args.model == 'resnet32':
    model = ResNet(args.dataset, 32, num_classes)
    filename = 'resnet32'+args.optim+'randseed'+str(args.seed)+'epoch'+str(args.epoch)+'batchsize'+str(args.batch_size)+'lr'+str(args.lr)
    
if not os.path.exists(filehead):
    os.makedirs(filehead)
    
filewrite = open(filehead+'/'+str(time.ctime())+filename+'.txt', 'w')


model.to(device)
param = sum([param.nelement() for param in model.parameters()])
print(model)
print(model,file=filewrite)

if args.dataset == 'imagenet32':
    modelfile_name = filehead+'/modelfile/'+filename
    if not os.path.exists(modelfile_name):
        os.makedirs(modelfile_name)


if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005)
elif args.optim == 'sgdm':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=0.0005)
elif args.optim == 'adam':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

print(optimizer)

scheduler = MultiStepLR(optimizer, milestones=[0.7*args.epoch,0.9*args.epoch],gamma=0.2)
criterion = CELoss()

best_testacc = 0.
best_top5 = 0.
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
    
    top5_acc = EvaluteTop5(model,
                           test_loader,
                           device=device)

    if args.dataset == 'imagenet32':
        torch.save(model.state_dict(),modelfile_name+'/epoch'+str(epoch)+'.pkl')

    filewrite.write(' epoch '+str(epoch)+
                    ' trainloss '+ str(trainloss)+
                    ' valloss ' +str(valloss)+
                    ' trainacc '+str('%.4f'%trainacc)+
                    ' test_acc '  +str('%.4f'%valacc)+
                    ' top5_acc '  +str('%.4f'%top5_acc)+'\n')

    if np.isnan(trainloss):
        break
    if best_testacc < valacc:
        best_trainloss = trainloss
        best_trainacc = trainacc
        best_valloss = valloss
        best_testacc = valacc
    if best_top5 < top5_acc:
        best_top5 = top5_acc
    scheduler.step()

print('best train loss %9.8f'% best_trainloss,'best train acc %5.4f' % best_trainacc, 
      'best test loss %7.6f'%best_valloss , 'best test acc %5.4f' %best_testacc)

filewrite.write('best_trainloss '+ str(best_trainloss)+
               ' best_trainacc ' +str(best_trainacc)+
               ' best_valloss '+str('%.4f'%best_valloss)+
               ' best_testacc '  +str('%.4f'%best_testacc)+
               ' best_top5acc '  +str('%.4f'%best_top5)+'\n')
filewrite.close()

