import sys
import argparse
import pickle
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchsummaryX import summary

from utils.data_loader import *
from utils import visualization
from utils.utils import *
from torch.utils.data import DataLoader
from networks.VGG import *
import train
from networks.seg_models import *


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
    parser.add_argument("--action", choices=['train', 'predict', 'demo', 'test'], required=True, help="Choose action.")
    parser.add_argument("--model", choices=['vgg', 'unet', 'fpn'], required=True, help="Choose model.")
    parser.add_argument("--dataset", choices=['full', 'small'], required=True, help="Choose dataset.")

    return parser.parse_args()


def main():
    args = parse_command_line()

    print(f'Training {args.model} with {args.dataset} dataset...')
    batch_size = 1

    if args.dataset == 'full':
        raw_data_train, raw_data_val, raw_data_test = load_data('nyu', small=False)
    else:
        raw_data_train, raw_data_val, raw_data_test = load_data('nyu', small=True)


    loader_train = DataLoader(raw_data_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(raw_data_val, batch_size=batch_size)
    loader_test = DataLoader(raw_data_test, batch_size=batch_size)

    if (args.action == 'train'):
        in_channel = 3

        if (args.model == 'vgg'):
            model = MyVGGNet('vgg', in_channel=in_channel)
        else:
            model = get_models(args.model)
            print(f'Using model: {args.model}')
        model = model.to(dtype=dtype, device=device)
        summary(model, torch.rand(1, 3, 480, 640))

        optimizer = optim.Adam(model.decoder.parameters(), lr=1e-4)
        train_log, val_log = train.train_nn(model, optimizer, loader_train, loader_val, batch_size=batch_size, epochs=10, device=device, dtype=dtype)

        write_log("result/train_log.txt", train_log)
        write_log("result/val_log.txt", val_log)

        torch.save(model, 'model_backup/final.pth')
        
        final_test_loss = train.check_accuracy(loader_test, model, device, dtype)
        print(f'Final test loss: {final_test_loss}')
        plot_log(train_log, val_log)

    else:

        with torch.no_grad():
            model = torch.load('model_backup/best_model_FPN_Last.pth', map_location=torch.device('cpu'))
            model = model.to(dtype=dtype, device=device)
            model.eval()
            if (args.action == 'demo'):

                rgb = Image.open('demo_nyud_rgb.jpg')
                
                # build depth inference function and run
                rgb_imgs = np.asarray(rgb)
                rgb_imgs=rgb_imgs.reshape((3, 480, 640))
                
                img_x = torch.from_numpy(np.array(rgb_imgs))
                img_x = img_x.to(dtype=dtype, device=device)
                predict = model(img_x.unsqueeze_(0))

                predict = np.squeeze(predict, axis=0)
                predict = np.squeeze(predict, axis=0)
                visualization.display_depth_map(predict)

            elif (args.action == 'predict'):

                test_img = next(iter(loader_test))
                for i in range(batch_size):
                    img_x = test_img['image'][i]
                    img_y = test_img['label'][i]

                    img_x = img_x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                    img_y = img_y.to(device=device, dtype=dtype)

                    # visualize ground truth
                    # visualization.display_depth_map(img_y.squeeze_())
                    predict = model(img_x.unsqueeze_(0))
                    predict = np.squeeze(predict, axis=0)
                    predict = np.squeeze(predict, axis=0)

                    # visualize predict image
                    # visualization.display_depth_map(predict)


                    # visualize ground truth and predict image side by side
                    plot_comparison(img_y.squeeze_(), predict)

            elif (args.action == 'test'):
                final_test_loss = train.check_accuracy(loader_test, model, device, dtype, batch_size=batch_size)
                print(f'Final test loss: {final_test_loss}')



if __name__ == "__main__":
    main()