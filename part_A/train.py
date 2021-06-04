import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import torch.nn.functional as F  # useful stateless functions


def check_accuracy(loader, model, device, dtype, batch_size=4):
    criterion = nn.SmoothL1Loss()
    tot_loss = 0.0
    num_samples = 0.0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for dataiter in iter(loader):
            x = dataiter['image']
            y = dataiter['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            y = torch.squeeze(y, 1)

            scores = model(x)
            scores = torch.squeeze(scores, 1)
            loss = criterion(scores, y)
            tot_loss = tot_loss + loss.item() * batch_size
            num_samples += 1
        avg_loss = float(tot_loss) / num_samples

    return avg_loss


def train_nn(model, optimizer, loader_train, loader_val, batch_size=4, device=torch.device('cpu'), dtype=torch.float32, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    criterion = nn.SmoothL1Loss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    train_log, val_log = [], []
    num_iter = 0
    for e in range(epochs):
        total_train_loss, total_num = 0.0, 0
        for dataiter in iter(loader_train):
            model.train()  # put model to training mode
            x = dataiter['image']
            y = dataiter['label']

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            scores = model(x)
            scores = torch.squeeze(scores, 1)
            y = torch.squeeze(y, 1)

            # raise NotImplementedError()
            loss = criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
            num_iter += 1
            total_num += batch_size
            total_train_loss += loss.item() * batch_size
            avg_train_loss = total_train_loss / total_num
            if num_iter % 5 == 0:
                print('-------------------------------------')
                print(f'Epoch: #{e}')
                print('Iteration %d, avg train loss = %.4f' % (num_iter, avg_train_loss))
                train_log.append((num_iter, avg_train_loss))
        avg_val_loss = check_accuracy(loader_val, model, device, dtype)
        val_log.append((e, avg_val_loss))
        print('========================================')
        print(f'Epoch #{e} Average val loss: {avg_val_loss}')
        torch.save(model, 'model_backup/checkpoint_epoch_'+ str(e) + '.pth')
        print('========================================')
    return train_log, val_log
