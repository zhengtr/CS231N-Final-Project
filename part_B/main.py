import sys
import argparse
from utils.data_loader import *
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from utils import visualization
import pickle
from torch.utils.data import DataLoader

from torchsummary import summary

from networks.VGG import *
from networks.ResNet_transfer import *
from networks.dual_scale_model import *
import train


USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dtype = torch.float32

def parse_command_line():
    """
        Parses the command line arguments.
    """

    parser = argparse.ArgumentParser(description='User args')
    parser.add_argument("--action", choices=['train', 'test','predict'], required=True, help="Choose action.")
    parser.add_argument("--model", choices=['ResMod','dual'], required=True, help="Choose model.")
    parser.add_argument("--dataset", choices=['nyu'], required=True, help="Choose dataset.")

    # Copy previous line to include additional transformations

    return parser.parse_args()


def write_log(filepath, logs):
    logs = [','.join([str(i) for i in list(t)]) + '\n' for t in logs]
    with open(filepath, 'w') as fp:
        fp.writelines(logs)

def main():
    args = parse_command_line()

    print(f'Training {args.model} with {args.dataset} dataset...')
    device = torch.device('cuda')
    print(device)
    batch_size = 4

    raw_data_train, raw_data_val, raw_data_test = load_data(args.dataset)

    loader_train = DataLoader(raw_data_train, batch_size=batch_size,shuffle=True)
    loader_val = DataLoader(raw_data_val, batch_size=batch_size)
    loader_test = DataLoader(raw_data_test, batch_size=batch_size)
    
    in_channel = 3
    if (args.model == 'ResMod'):
        model = TransferResNet([512,512,128,4800,76800])
    elif (args.model == 'dual'):
        model = ScaleModel()

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if (args.action == 'train'):
        train_log, val_log = train.train_nn(model, optimizer, loader_train, loader_val, epochs=20, device=device, dtype=dtype, batch_size=batch_size,mode=args.model)

        write_log("result/train_log.txt", train_log)
        write_log("result/val_log.txt", val_log)

        torch.save(model, 'model_backup/final.pth')
        final_test_loss = train.check_accuracy(loader_test, model, device, dtype,mode=args.model)
        print(f'Final test loss: {final_test_loss}')


    if (args.action == 'test'):
        with torch.no_grad():
            test_img = next(iter(loader_test))
            img_x = test_img['image'][0]
            img_y = test_img['label'][0]

            img_x = img_x.to(device=device, dtype=dtype)
            img_y = img_y.to(device=device, dtype=dtype)

            model = torch.load('model_backup/final.pth', map_location=torch.device('cpu'))
            model = model.to(dtype=dtype, device=device)
            model.eval()

            visualization.display_depth_map(img_y, save_path="./ground_map.png")
            
            img = img_x.permute(1,2,0)
            plt.figure()
            plt.imshow(img.cpu().detach().numpy())
            plt.savefig("./input.png")
            plt.show()

            if (args.model == 'dual'):
                predict, coarse_img = model(img_x.unsqueeze_(0))
                predict=model(img_x.unsqueeze_(0))
                coarse_img = np.squeeze(coarse_img, axis=0)
                coarse_img = np.squeeze(coarse_img, axis=0)
                visualization.display_depth_map(coarse_img, save_path="./coarse.png")
            else:
                predict = model(img_x.unsqueeze_(0))

            visualization.display_depth_map(img_y, save_path="./ground_map.png")
            predict = np.squeeze(predict, axis=0)
            predict = np.squeeze(predict, axis=0)
            visualization.display_depth_map(predict,save_path="./test.png")

    elif (args.action == 'predict'):
        with torch.no_grad():
            test_img = next(iter(loader_test))
            for i in range(batch_size):
                img_x = test_img['image'][i]
                img_y = test_img['label'][i]

                img_x = img_x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                img_y = img_y.to(device=device, dtype=dtype)

                model = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
                model = model.to(dtype=dtype, device=device)

                model.eval()

                visualization.display_depth_map(img_y, save_path="./ground_map.png")

                if (args.model == 'dual'):
                    predict,coarse = model(img_x.unsqueeze_(0))
                    coarse_img = np.squeeze(coarse_img, axis=0)
                    coarse_img = np.squeeze(coarse_img, axis=0)
                    visualization.display_depth_map(coarse_img, save_path="./coarse.png")
                else:
                    predict = model(img_x.unsqueeze_(0))

        
                predict = np.squeeze(predict, axis=0)
                predict = np.squeeze(predict, axis=0)
                visualization.display_depth_map(predict,save_path="./test.png")




if __name__ == "__main__":
    main()
