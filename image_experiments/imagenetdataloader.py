from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class Imagenet32(Dataset):
    """`Imagenet32 Dataset.

    """
    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10'],
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, train=True, download=None,transform=None):
        self.train = train  # training set or test set

        self.transform = transform
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list

        self.data = []
        self.targets = []
        self.root = root
        # now load the picked numpy arrays
        for file_name in data_list:
            file_path = os.path.join(self.root,file_name[0])
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, np.array(target)-1

    def __len__(self):
        return len(self.data)

