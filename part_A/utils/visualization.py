#!/usr/bin/env python

#imports:
import matplotlib.pyplot as plt
import numpy as np
import torch

def display_depth_map(pix_input, h=None,w=None, save_path=None):
    """
    This function creates a depth map visual given a pixel input vector or image. 
    If a vector is given, the desired shape (h,w) must be given as well. 
    If a save path is provided, the function will save an image of the depth map
    at that location.
    INPUTS:
        pix_input: depth of each pixel as either a vector or a 2D matrix (assumes type Numpy)
        h: (optional) specifies the pixel height of the desired image. Required if pix_input is a vector.
        w: (optional) specifies the pixel width of the desired image. Required if pix_input is a vector.
        save_path: (optional) specifies location in file directory to save the desired image.
    OUTPUT:
        NONE
    """

    if pix_input is None:
        print("A depth input vector was not given. However this is required. Given 'None' expected 'numpy vector'.")
        return
    else:
        if isinstance(pix_input, np.ndarray):
            pass

        elif isinstance(pix_input, torch.Tensor):
            pix_input = pix_input.cpu().detach().numpy()
            print("converted into numpy array")

        else:
            print("invalid vector was given! check pix_input")
            return

    if pix_input.ndim > 2:
        print("pix imput has too many dimensions, please flatten down to 2 or less dimensions")
        return

    elif pix_input.ndim < 2:
        if h is None or w is None:
            print("A dimension was not specified! the current dimensions are: (" + str(h) +","+ str(w) + ")")
            return
        
        try:
            pix_input = pix_input.reshape(h,w)
            print("success!") 
        except:
            print("could not reshape to desired (h,w)! please check your dimension math")
            return

    plt.figure()
    plt.imshow(pix_input)
    plt.colorbar(label = 'Distance to Camera')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

    return 

def display_accuracy_plot(accuracy, save_path=None):
    """
    This function plots the accuracy of a model with respect to the number of epochs.
    It does this by taking a history vector of calculated accuracy and plotting it against
    the number of epochs.
    INPUT:
        accuracy: history vector of calculated accuracy values sum(predicted pix - ground truth)/num of samples
        save_path: specifies location to save figure in the directory if desired
    OUTPUT:
        NONE
    """
    if accuracy is None:
        print("An accuracy vector was not given. However this is required. Given 'None' expected 'numpy vector'.")
        return
    else:
        if isinstance(accuracy, np.ndarray):
            if accuracy.ndim != 1:
                print("accuracy vector has too many dimensions! please flatten first.")
                return
        elif isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.cpu().detach().numpy()
            print("converted into numpy array")
            if accuracy.ndim != 1:
                print("accuracy vector has too many dimensions! please flatten first.")
                return

    epochs = np.arange(0,accuracy.shape[0])

    plt.figure()
    plt.plot(epochs,accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Model Accuracy %')

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()