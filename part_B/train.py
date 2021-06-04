import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import torch.nn.functional as F  # useful stateless functions
import matplotlib.pyplot as plt



def check_accuracy(loader, model, device, dtype, batch_size=4,mode='none'):
    criterion = nn.SmoothL1Loss(beta=2)#nn.MSELoss()

    tot_loss = 0.0
    num_samples = 0.0
    model.eval()  # set model to evaluation mode
    
    with torch.no_grad():
        for dataiter in iter(loader):
            x = dataiter['image']
            y = dataiter['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            if mode == 'dual':
                scores, coarse_scores = model(x)
                coarse_scores = torch.squeeze(coarse_scores, 1)
            else:
                scores = model(x)
            scores = torch.squeeze(scores, 1)
            tot_loss = tot_loss + criterion(scores, y)*batch_size #+ criterion2(scores, y)
            num_samples += 1
        avg_loss = float(tot_loss) / num_samples
        print(f'Average val loss: {avg_loss}')

    return avg_loss


def train_nn(model, optimizer, loader_train, loader_val, device=torch.device('cpu'), dtype=torch.float32, epochs=1, batch_size=4,mode='none'):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    criterion1 = nn.SmoothL1Loss(beta=2) #nn.MSELoss()
    #criterion1 = nn.L1Loss()
    #criterion3 = nn.KLDivLoss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    
    train_log, val_log = [], []
    t = 0
    for e in range(epochs):
        total_train_loss, total_num = 0, 0
        loader_train.dataset.set_transform(True)
        num_iter = 0
        for dataiter in iter(loader_train):

            model.train()  # put model to training mode
            x = dataiter['image']
            y = dataiter['label']

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            x.require_grad = True
            
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            if mode == 'dual':
                scores, coarse_scores = model(x)
                coarse_scores = torch.squeeze(coarse_scores, 1)
            else:
                scores = model(x)
                
            scores = torch.squeeze(scores, 1)

            if mode is 'dual':
                loss = criterion1(scores,y) + criterion1(coarse_scores,y)
            else:
                loss = criterion1(scores, y)
                
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
        
            t += 1
            total_num += batch_size
            total_train_loss += loss.item()*batch_size
            avg_train_loss = total_train_loss/total_num

            if t % 5 == 0:
                print('___________________________')
                print(f'Epoch: #{e}')
                print('Iteration %d, avg train loss = %.4f' % (t, avg_train_loss))
                train_log.append((t, avg_train_loss))

                if num_iter%50 ==0:

                    plt.figure()
                    plt.imsave("imgs/pred_"+str(e)+"_"+str(t)+".png",scores[0,:,:].cpu().detach().numpy())
                    plt.xlabel('X Pixel')
                    plt.ylabel('Y Pixel')
                    plt.close()
                  
                    if mode is 'dual':
                        plt.figure()
                        plt.imsave("imgs/pred_coarse_"+str(e)+"_"+str(t)+".png",coarse_scores[0,:,:].cpu().detach().numpy())
                        plt.xlabel('X Pixel')
                        plt.ylabel('Y Pixel')
                        plt.close()
                    
                    plt.figure()
                    plt.imsave("imgs/gt_"+str(e)+"_"+str(t)+".png",y[0,:,:].cpu().detach().numpy())
                    plt.xlabel('X Pixel')
                    plt.ylabel('Y Pixel')
                    plt.close()

                    plt.figure()
                    plt.imsave("imgs/input_"+str(e)+"_"+str(t)+".png",np.transpose(x[0,:,:,:].cpu().detach().numpy(),(1,2,0)).astype(np.uint8))
                    plt.xlabel('X Pixel')
                    plt.ylabel('Y Pixel')
                    plt.close()
                    #stop
                
        avg_val_loss = check_accuracy(loader_val, model, device, dtype, batch_size=batch_size, mode=mode)
        val_log.append((e,avg_val_loss))
                
        torch.save(model, 'checkpoint.pth')
    return train_log, val_log



















