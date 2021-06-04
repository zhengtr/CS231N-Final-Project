# CS 231N (Convolutional Neural Networks for Visual Recognition, Spring 2021) Final Project.
## Title: Estimating Depth From RGB Monocular images

This is the repository CS231N final project.  
Created by *Tanran Zheng*, *Gadiel Sznaier Camps* and *Raymond Guo*.

# Report
For more details, please check our final report `CS231N_Final_Report.pdf`.

# Setup
You can use pipenv to install all the requirement modules in the pipfile.

# Part A
Part A implements the training, predicting, and testing of UNet Transfer model and FPN transfer model.

**Data**  
Download the NYU Depth V2 labeled dataset (~2.8GB) from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html and save the data file `nyu_depth_v2_labeled.mat` into the 'dataset' directory.


**Scripts**  
`main.py`: contains the main function to run the implementation  

`train.py`: contains the functions to train and check model's loss

`utils/data_loarder.py`: contains function to load the NYU Depth V2 dataset into pytorch DataLoader

`utils/utils.py`: contains auxiliary functions

`utils/visualization.py`: contains functions to plot the depth map

`networks/VGG.py`: contains the code of a simple toy CNN model

`networks/seg_models.py`: contains the code of UNet Transfer model and FPN transfer model. Models are imported from module `segmentation_models_pytorch`, see https://smp.readthedocs.io/en/latest/ for details.

**To run**  
Run
```
pipenv run python main.py --model fpn --dataset nyu --action train 
```
under ./part_A.

The `--model` argument specifies which model to be used, choices: `vgg`, `unet`, `fpn`, `unet++`;

The `--dataset` argument specifies which dataset to be used, choices: `small` (use a very small subset of the NYUD2 dataset) and `full` (use the full NYUD2 dataset);

The `--action` argument specifies the task, choices: `train` (train model), `predict` (load model weights and visualize the predict images), and `test` (calculate and print the test accuracy using test dataset);

# Part B
Part B implements the training, predicting, and testing of ResNet Transfer model and Multi-Scaled model.

**Data**  
Download the NYU Depth V2 labeled dataset (~2.8GB) from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html and save the data file `nyu_depth_v2_labeled.mat` into the 'dataset' directory.


**Scripts**  
`main.py`: contains the main function to run the implementation  

`train.py`: contains the functions to train and check model's loss

`utils/data_loarder.py`: contains function to load the NYU Depth V2 dataset into pytorch DataLoader

`utils/visualization.py`: contains functions to plot the depth map

`networks/dual_scale_model.py`: contains the code of the Multi-Scaled model

`networks/seg_models.py`: contains the code of ResNet Transfer model

**To run**  
Run
```
pipenv run python main.py --model ResMod --dataset nyu --action train 
```
under ./part_B.

The `--model` argument specifies which model to be used, choices: `ResMod` (ResNet Transfer model), `dual` (Multi-Scaled model);

The `--dataset` argument specifies which dataset to be used, choices: `nyu`;

The `--action` argument specifies the task, choices: `train` (train model), `predict` (load model weights and visualize the predict images), and `test` (calculate and print the test accuracy using test dataset);

# Major reference
For detailed reference, please check our final report `CS231N_Final_Report.pdf`.

Dataset: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html  
UNet, UNet++, FPN: https://smp.readthedocs.io/en/latest/  
Multi-Scaled model: [David Eigen, Christian Puhrsch, and Rob Fergus. Depth mapprediction from a single image using a multi-scale deep net-work.arXiv preprint arXiv:1406.2283, 2014.](https://arxiv.org/abs/1406.2283)

# Disclaimer
**All rights reserved by the authors and Stanford University.**
