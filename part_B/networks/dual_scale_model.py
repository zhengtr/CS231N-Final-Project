import sys
import torch
import torch.nn as nn
import torch.optim as optim

class ScaleModel(nn.Module):
    def __init__(self):
        super(ScaleModel, self).__init__()
        
        layers = nn.Sequential(*[nn.Conv2d(3, 96,5,1,2),
                                nn.MaxPool2d(2),
                                nn.ReLU(),
                                torch.nn.BatchNorm2d(96),
                                nn.Conv2d(96, 100, 3,1,1),
                                nn.MaxPool2d(2),
                                nn.ReLU(),
                                torch.nn.BatchNorm2d(100),
                                nn.Conv2d(100, 100, 3,1,1),
                                nn.MaxPool2d(2),
                                nn.ReLU(),
                                torch.nn.BatchNorm2d(100),
                                nn.Conv2d(100, 256, 3,1,1),
                                torch.nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 1, 3,1,1),
                                nn.Flatten(),
                                ])
        self.model1 = layers
        self.fc = nn.Sequential(*[nn.Linear(4800,9600),
                                  torch.nn.ReLU(),
                                  nn.Linear(9600,19200)
                                  ])
        
        layers = nn.Sequential(*[nn.Conv2d(3,100, 11,1,5),
                                torch.nn.BatchNorm2d(100),
                                nn.ReLU(),
                                ])
        self.model2 = layers
        layers = nn.Sequential(*[nn.Conv2d(101, 101,5,1,2),
                                torch.nn.BatchNorm2d(101),
                                nn.ReLU(),
                                nn.Conv2d(101, 40, 3,1,1),
                                torch.nn.BatchNorm2d(40),
                                nn.ReLU(),
                                nn.Conv2d(40, 1, 3,1,1)])
        self.model3 = layers
        
    def forward(self,x):
        
        coarse_img = self.model1(x)
        coarse_img = self.fc(coarse_img)
        coarse_img = coarse_img.reshape(-1,1,120,160)
        m = nn.Upsample(scale_factor=4, mode='bilinear')
        coarse_img = m(coarse_img)

        fine_img = self.model2(x)
        fine_img = torch.cat((fine_img,coarse_img),1)
        fine_img = self.model3(fine_img)
    

        return fine_img, coarse_img
        
        
