import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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

def write_log(filepath, logs):
    logs = [','.join([str(i) for i in list(t)]) + '\n' for t in logs]
    with open(filepath, 'w') as fp:
        fp.writelines(logs)

def plot_log(train_log, val_log):
    val_log = list(zip(*val_log))
    train_log = list(zip(*train_log))

    plt.figure(figsize=(10, 6))
    plt.plot(val_log[0], val_log[1], 'o-', color='r', label='Val')
    plt.plot(np.linspace(1, val_log[0][-1], len(train_log[0])), train_log[1], 'o-', color='b', label='Train')  # TODO: change color
    plt.xticks(val_log[0])
    plt.legend()
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')  
    # plt.show()
    plt.savefig("result/loss_curve.png")

def plot_comparison(img_gt, img_predict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.tight_layout()

    axes[0].imshow(img_gt)
    axes[0].set_xlabel('Ground Truth', fontsize=14)
    axes[1].imshow(img_predict)
    axes[1].set_xlabel('Prediction', fontsize=14)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.3, 0.03, 0.4])
    fig.colorbar(axes[1].imshow(img_predict), cax=cbar_ax)
    plt.show()