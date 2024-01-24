import torch
from torchvision import datasets, transforms
from imagenetdataloader import Imagenet32
from randaugment import RandAugment, Cutout,CIFAR10Policy, ImageNetPolicy

def data_loader(dataset,augment=0,batch_size=256,image_size=32):
    if dataset.startswith('stl'):
        trainprocess = [transforms.RandomCrop(96, padding=12),  
                        transforms.RandomHorizontalFlip()
                       ]
    else:
        trainprocess = [transforms.RandomCrop(32, padding=4),  
                        transforms.RandomHorizontalFlip()
                       ]
    if augment>0:    
        if dataset.startswith('stl'):
            trainprocess.extend([CIFAR10Policy(),
                                Cutout(size=32)
                                ])
        else:
            trainprocess.extend([CIFAR10Policy(),
                                Cutout(size=16)
                                ])
    if image_size>32:
        tensorprocess = [ transforms.Resize(image_size),  
                         transforms.ToTensor(),  
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]
    else:
        tensorprocess = [transforms.ToTensor(),  
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]

    trainprocess.extend(tensorprocess)
    
    if dataset == 'imagenet32':
        traindata = Imagenet32(root='../data/imagenet32/',train=True,
                               transform=transforms.Compose(trainprocess))
        testdata = Imagenet32(root='../data/imagenet32/',train=False,
                                    transform=transforms.Compose(tensorprocess))
        num_classes = 1000
    elif dataset == 'cifar100':
        traindata = datasets.CIFAR100(root='../data/',train=True,download=False, 
                                    transform=transforms.Compose(trainprocess))
        testdata = datasets.CIFAR100(root='../data/',train=False,download=False, 
                                    transform=transforms.Compose(tensorprocess))
        num_classes = 100
    elif dataset == 'cifar10':
        traindata = datasets.CIFAR10(root='../data/',train=True,download=False, 
                                 transform=transforms.Compose(trainprocess))
        testdata = datasets.CIFAR10(root='../data/',train=False,download=False, 
                                 transform=transforms.Compose(tensorprocess))
        num_classes = 10
    elif dataset == 'stl10':
        traindata = datasets.STL10(root='../data/',split='train',download=False, 
                                    transform=transforms.Compose(trainprocess))
        testdata = datasets.STL10(root='../data/',split='test',download=False, 
                                    transform=transforms.Compose(tensorprocess))
        num_classes = 10
    
    elif dataset =='Imagenet':
        trainprocess = [transforms.RandomResizedCrop(256),  
                        transforms.RandomHorizontalFlip(),
                        RandAugment(),
                        Cutout(size=16), 
                        transforms.ToTensor(),  
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ]
        traindata = datasets.ImageFolder(root='../../Imagenet/train',
                                    transform=transforms.Compose(trainprocess))

        testprocess = [transforms.Resize(256),  
                        transforms.ToTensor(),  
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ]
        testdata = datasets.ImageFolder(root='../../Imagenet/val',
                            transform=transforms.Compose(testprocess))
        num_classes = 1000

    train_loader = torch.utils.data.DataLoader(traindata,batch_size=batch_size, shuffle=True, 
                                               num_workers=4,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True, 
                                              num_workers=4,pin_memory=True)
    return train_loader,test_loader,num_classes