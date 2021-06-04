
import sys
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image

DATA_FOLDER_PATH = './dataset/'

data_file_name = {
    'nyu' : 'nyu_depth_v2_labeled.mat'
}


class MyDataset(Dataset):
  ''''
  Characterizes a dataset for PyTorch
  '''
  def __init__(self, inputs, labels, transform=None, target_transform=None):
        'Initialization'
        self.images = inputs
        self.labels = labels
        self.N = self.images.shape[0]
        self.transform = transform
        self.target_transform = target_transform

  def __len__(self):
        'Denotes the total number of samples'
        return self.N

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        label = self.labels[index, :]
        img = self.images[index, :]
 
        #print("")
        #print(label.shape)
        #print(img.shape)
        org_size = list(img.shape)[0:2]
        img = np.moveaxis(img,0,-1)
        #img = T.ToTensor()(img)
        #label = T.ToTensor()(label)
        #print(img.shape)
        #print(label.shape)

        img = T.ToPILImage()(img)
        label = T.ToPILImage()(label)


        #print(img.size)
        #print(label.size)

        #img = np.transpose(img,(1,2,0))

        #img = T.ToPILImage()(img)
        #label = T.ToPILImage()(label)


        if self.transform is not None:
           
            #color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
            #img = T.RandomApply([color_jitter],p=0.6)(img)
            img = T.RandomApply([T.Grayscale(num_output_channels=3)],p=0.2)(img)
            if np.random.rand() > .5:
                #print("rot")
                r = np.random.uniform(-5,5)
                img = TF.rotate(img,r)
                label = TF.rotate(label,r)

            if np.random.rand() > 1:
                #print("scale")
                s = np.random.randint(1,3)
            
                size = [x*s for x in org_size]
                img = TF.resize(img,size)
                img = TF.center_crop(img,org_size)
                
                label = TF.resize(label,size)
                label = TF.center_crop(label,org_size)
                label = T.ToPILImage()(np.array(label)/s)

            if np.random.rand() > .4:
                #print("flip")
                img = TF.hflip(img)
                label = TF.hflip(label)

        else:
            pass
            #print("CRAP")
      
        img = T.ToTensor()(img)
        label = T.ToTensor()(label)

        #if self.transform is not None:
        img = T.Normalize([0.5145, 0.4555, 0.4275],[0.2541, 0.2605, 0.2748])(img)*255
        #else:
        #    img = img

        label = torch.squeeze(label)
        #print(img.shape)
        #print(label.shape)
        
        
        sample = {'image': img, 'label': label}
        

        return sample

  def set_transform(self, transform):
        self.transform = transform


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

    x_train_data = data_x[:960, ]
    y_train_data = data_y[:960, ]
    x_val_data = data_x[960:1200, ]
    y_val_data = data_y[960:1200, ]
    x_test_data = data_x[1200:,]
    y_test_data = data_y[1200:,]

    

    # transform = T.Compose([
    #         T.ToTensor(),
    #         T.Normalize((125.0117, 107.3517, 101.7845), (67.4572, 69.5167, 71.9448))
    #     ])
    # data_x = np.zeros((1449, 3, 480, 640))
    # data_y = np.zeros((1449, 480, 640))

    train_data = MyDataset(data_x[:1040, ], data_y[:1040, ])
    val_data = MyDataset(data_x[1040:1300, ], data_y[1040:1300, ])
    test_data = MyDataset(data_x[1300:,], data_y[1300:,]) # use first image for visualization! 

    #loader = DataLoader(train_data, batch_size=1, num_workers=1)
    #data = next(iter(loader))
    
    #print(np.array(data['image'].view(x_train_data.shape[0],x_train_data.shape[1],-1)*1.0).shape)
    #xmean = torch.sum((T.ToTensor()(np.array(data['image'].view(x_train_data.shape[0],x_train_data.shape[1],-1)*1.0))).mean(1),0)/x_train_data.shape[0]
    #xstd = torch.sum((T.ToTensor()(np.array(data['image'].view(x_train_data.shape[0],x_train_data.shape[1],-1)*1.0))).std(1),0)/x_train_data.shape[0]
    #ymean = torch.sum((T.ToTensor()(np.array(data['label'].view(y_train_data.shape[0],-1)*1.0))).mean(0),0)/y_train_data.shape[0]
    #ystd = torch.sum((T.ToTensor()(np.array(data['label'].view(y_train_data.shape[0],-1)*1.0))).std(0),0)/y_train_data.shape[0]

    #print(xmean)
    #print(ymean)
    #print(xstd)
    #print(ystd)
    #stop

    return train_data, val_data, test_data

def load_data(dataset):
    """
    Load dataset

    Inputs:
    - dataset: name of the dataset, which should match the key of `data_file_name`.

    Returns:
    - data_x: (N, C, H, W) numpy array of pixel value
    - data_y: (N, H, W) numpy array of depth value
    """
    data_path = os.path.join(DATA_FOLDER_PATH, data_file_name[dataset])
    train_data, val_data, test_data = load_nyu_data(data_path)
    return train_data, val_data, test_data



def calMeanStd(dataset):
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for d in loader:
        data = d['image'].to(dtype=torch.float32)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std
