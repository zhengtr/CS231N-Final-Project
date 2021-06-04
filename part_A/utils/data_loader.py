
import sys
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.utils import *
import random
import torchvision.transforms.functional as TF

DATA_FOLDER_PATH = './dataset/'

data_file_name = {
    'nyu' : 'nyu_depth_v2_labeled.mat'
}


class MyDataset(Dataset):
    ''''
    Characterizes a dataset for PyTorch
    '''
    def __init__(self, inputs, labels, flip=True):
        'Initialization'
        self.images = inputs
        self.labels = labels
        self.N = self.images.shape[0]
        self.flip = flip


    def __len__(self):
        'Denotes the total number of samples'
        return self.N
    
    def transform(self, image, mask):

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        if self.flip:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            if random.random() > 0.6:
                r = np.random.uniform(-5, 5)
                image = TF.rotate(image, r)
                mask = TF.rotate(mask, r)               
                
        # normalize
        # image = TF.normalize(image, (0.4806, 0.4109, 0.3923), (0.2637, 0.2721, 0.2821))
        return image, mask

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        label = self.labels[index, :]
        img = self.images[index, :]
        img = np.moveaxis(img, 0, -1)
        img, label = self.transform(img, label)

        sample = {'image': img, 'label': label}

        return sample


def load_nyu_data(path):
    """
    Load NYU depth v2 dataset

    Inputs:
    - path

    Returns:
    data_x: (N, 3, H, W) numpy array of pixel value
    data_y: (N, H, W) numpy array of depth value
    """
    with h5py.File(path, 'r') as f:
        data_x = np.swapaxes(np.array(f.get('images')), 2, 3) # f shape: (N, 3, W, H)
        data_y = np.swapaxes(np.array(f.get('depths')), 1, 2) 

    return data_x, data_y

def load_data(dataset, small=False):
    """
    Load dataset

    Inputs:
    - dataset: name of the dataset, which should match the key of `data_file_name`.

    Returns:
    - data_x: (N, C, H, W) numpy array of pixel value
    - data_y: (N, H, W) numpy array of depth value
    """
    data_path = os.path.join(DATA_FOLDER_PATH, data_file_name[dataset])

    data_x, data_y = load_nyu_data(data_path) # (1449, 3, 480, 640) & (1449, 480, 640)
    
    if small:
        train_data = MyDataset(data_x[:32, ], data_y[:32, ])
        val_data = MyDataset(data_x[32:40, ], data_y[32:40, ])
        test_data = MyDataset(data_x[40:50,], data_y[40:50,], flip=False)
    else:
        train_data = MyDataset(data_x[:1040, ], data_y[:1040, ])
        val_data = MyDataset(data_x[1040:1300, ], data_y[1040:1300, ])
        test_data = MyDataset(data_x[1300:,], data_y[1300:,], flip=False)

    return train_data, val_data, test_data
