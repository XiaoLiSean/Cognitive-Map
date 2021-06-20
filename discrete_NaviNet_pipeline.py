'''
Retrieval Network Testing in Discrete World
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
from Network.navigation_network.losses import Cross_Entropy_Loss
from Network.navigation_network.trainer import Training
from Network.retrieval_network.trainer import plot_training_statistics
from discrete_RNet_pipeline import show_testing_histogram, show_testing_histogram_comparison
from os.path import dirname, abspath

# ------------------------------------------------------------------------------
# -------------------------------Testing Pipeline-------------------------------
# ------------------------------------------------------------------------------
def testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=False, benchmark=None):
    # ---------------------------Loading testing dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading testing dataset...')
    test_dataset = Dataset(DATA_DIR, is_test=True, load_only_image_data=is_only_image_branch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    model = Network(only_image_branch=is_only_image_branch, benchmarkName=benchmark)
    model.load_state_dict(torch.load(checkpoints_prefix + 'best_fit.pkl'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model testing on: ", device)
    print("Cuda is_available: ", torch.cuda.is_available())
    print('----'*20)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    # ------------------------------Testing Main------------------------------------
    testing_statistics = {}
    bar = Bar('Processing', max=len(test_loader))
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = tuple(input.to(device) for input in inputs)
        targets = targets.to(device)
        outputs = model(*inputs)
        _, is_correct = LossFcn(outputs, targets, batch_average_loss=True)
        iFloorPlan = test_dataset.trajectories[batch_idx][0].split('/')[4]
        if iFloorPlan in testing_statistics:
            testing_statistics[iFloorPlan]['total'] += 1
            testing_statistics[iFloorPlan]['corrects'] += is_correct.item()
        else:
            testing_statistics.update({iFloorPlan:dict(total=1, corrects=is_correct.item())})
        bar.next()

    bar.finish()
    print('----'*20)
    np.save(checkpoints_prefix + 'testing_statistics.npy', testing_statistics)

# ------------------------------------------------------------------------------
# -------------------------------Training Pipeline------------------------------
# ------------------------------------------------------------------------------
def training_pipeline(Dataset, Network, LossFcn, Training, checkpoints_prefix, is_only_image_branch=False, benchmark=None):
    dataset_sizes = {}
    # ---------------------------Loading training dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading training dataset...')
    train_dataset = Dataset(DATA_DIR, is_train=True, load_only_image_data=is_only_image_branch)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[benchmark], shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'train': len(train_dataset)})

    # --------------------------Loading validation dataset--------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading validation dataset...')
    val_dataset = Dataset(DATA_DIR, is_val=True, load_only_image_data=is_only_image_branch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE[benchmark], shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'val': len(val_dataset)})

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    model = Network(only_image_branch=is_only_image_branch, benchmarkName=benchmark)
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
    model_best_fit = Training(data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS, checkpoints_prefix=checkpoints_prefix, batch_size=BATCH_SIZE[benchmark])

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
    parser.add_argument("--benchmark", help="network image branch", action="store_true")
    parser.add_argument("--name", type=str, default='none',  help="benchmark network name: vgg16, resnet50, resnext50_32x4d, googlenet")
    parser.add_argument("--rnet", help="entire network", action="store_true")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    show_testing_histogram_comparison(parent_dir=CHECKPOINTS_DIR,filename='testing_statistics.npy')

    # --------------------------------------------------------------------------
    # Train corresponding networks
    if args.train:
        Dataset = NaviDataset
        Network = NavigationNet
        LossFcn = Cross_Entropy_Loss()
        if args.benchmark and not args.rnet:
            checkpoints_prefix = CHECKPOINTS_DIR + args.name + '/'
        elif args.rnet and not args.benchmark:
            checkpoints_prefix = CHECKPOINTS_DIR + 'rnet/'
            args.name = 'rnet'
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        TraningFcn = Training
        model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn, checkpoints_prefix, is_only_image_branch=args.benchmark, benchmark=args.name)
        torch.save(model_best_fit.state_dict(), checkpoints_prefix + 'best_fit.pkl')
        plot_training_statistics(parent_dir=CHECKPOINTS_DIR, filename='training_statistics.npy')

    # --------------------------------------------------------------------------
    # Test corresponding networks
    if args.test:
        Dataset = NaviDataset
        Network = NavigationNet
        LossFcn = Cross_Entropy_Loss()
        if args.benchmark and not args.rnet:
            checkpoints_prefix = CHECKPOINTS_DIR + args.name + '/'
        elif args.rnet and not args.benchmark:
            checkpoints_prefix = CHECKPOINTS_DIR + 'rnet/'
            args.name = 'rnet'
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=args.benchmark, benchmark=args.name)
        show_testing_histogram(checkpoints_prefix+'testing_statistics.npy')
        show_testing_histogram_comparison(parent_dir=CHECKPOINTS_DIR,filename='testing_statistics.npy')
