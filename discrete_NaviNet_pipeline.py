'''
Retrieval Network Testing in Discrete World, Written by Xiao
For robot localization in a dynamic environment.
'''
# Import params and similarity from lib module
import torch
import argparse, os, copy, pickle, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from progress.bar import Bar
from torchvision import transforms
from torch.utils.data import DataLoader
from termcolor import colored
from lib.scene_graph_generation import Scene_Graph
from Network.navigation_network.params import *
from Network.navigation_network.datasets import NaviDataset
from Network.navigation_network.networks import NavigationNet
from Network.navigation_network.losses import CrossEntropyLoss
from Network.navigation_network.trainer import Training
from os.path import dirname, abspath

# ------------------------------------------------------------------------------
# -------------------------------Training Pipeline------------------------------
# ------------------------------------------------------------------------------
def training_pipeline(Dataset, Network, LossFcn, Training, checkpoints_prefix, is_only_image_branch=False):
    dataset_sizes = {}
    # ---------------------------Loading training dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading training dataset...')
    train_dataset = Dataset(DATA_DIR, is_train=True, load_only_image_data=is_only_image_branch)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'train': len(train_dataset)})

    # --------------------------Loading validation dataset--------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading validation dataset...')
    val_dataset = Dataset(DATA_DIR, is_val=True, load_only_image_data=is_only_image_branch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'val': len(val_dataset)})

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    model = Network(only_image_branch=is_only_image_branch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model training on: ", device)
    print("Cuda is_available: ", torch.cuda.is_available())

    model.to(device)

    # Uncomment to see the summary of the model structure
    # summary(model, input_size=[(3, IMAGE_SIZE, IMAGE_SIZE), (3, IMAGE_SIZE, IMAGE_SIZE)])

    # ----------------------------Set Training Critera------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Set Training Critera...')
    # Define loss function
    loss_fcn = LossFcn
    # Observe that all parameters are being optimized
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # optimizer = torch.optim.Adam(model.parameters())
    # Decay LR by a factor of GAMMA every STEP_SIZE epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # --------------------------------Training--------------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Training with dataset size --> ', dataset_sizes)
    data_loaders = {'train': train_loader, 'val': val_loader}
    model_best_fit = Training(data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS, checkpoints_prefix=checkpoints_prefix)

    # ------------------------------------------------------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Done... Best Fit Model Saved')
    print('----'*20)

    return model_best_fit

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train network", action="store_true")
    parser.add_argument("--test", help="test network", action="store_true")
    parser.add_argument("--heatmap", help="test network", action="store_true")
    parser.add_argument("--image", help="network image branch", action="store_true")
    parser.add_argument("--all", help="entire network", action="store_true")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # Train corresponding networks
    if args.train:
        Dataset = NaviDataset
        LossFcn = CrossEntropyLoss()
        if args.image and not args.all:
            Network = NavigationNet
            checkpoints_prefix = CHECKPOINTS_DIR + 'image_'
        elif args.all and not args.image:
            Network = NavigationNet
            checkpoints_prefix = CHECKPOINTS_DIR
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        TraningFcn = Training
        model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn, checkpoints_prefix, is_only_image_branch=args.image)
        torch.save(model_best_fit.state_dict(), checkpoints_prefix + 'best_fit.pkl')
