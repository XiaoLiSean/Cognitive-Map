'''
Retrieval Network, Written by Xiao
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
from torchsummary import summary
from termcolor import colored
from lib.scene_graph_generation import Scene_Graph
from Network.retrieval_network.params import *
from Network.retrieval_network.sanity_checker import SanityChecker, SanityDataset
from Network.retrieval_network.losses import TripletLoss
from Network.retrieval_network.trainer import Training, plot_training_statistics
from network_pipeline import show_testing_histogram_comparison
from os.path import dirname, abspath

'''Redefine the network and training variable'''
BATCH_SIZE = 200
CHECKPOINTS_DIR += 'sanity_check/'
NUM_EPOCHS = 120

# ------------------------------------------------------------------------------
# -------------------------------Training Pipeline------------------------------
# ------------------------------------------------------------------------------
def training_pipeline(Dataset, Network, LossFcn, Training, checkpoints_prefix, positional_feature_enabled=False):
    dataset_sizes = {}
    # ---------------------------Loading training dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading training dataset...')
    train_dataset = Dataset(DATA_DIR, is_train=True, positional_feature_enabled=positional_feature_enabled)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'train': len(train_dataset)})

    # --------------------------Loading validation dataset--------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading validation dataset...')
    val_dataset = Dataset(DATA_DIR, is_val=True, positional_feature_enabled=positional_feature_enabled)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'val': len(val_dataset)})

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    model = Network(enableBbox=positional_feature_enabled) # Train Image Branch

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
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # --------------------------------Training--------------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Training with dataset size --> ', dataset_sizes)
    data_loaders = {'train': train_loader, 'val': val_loader}
    model_best_fit = Training(data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS, checkpoints_prefix=checkpoints_prefix, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Done... Best Fit Model Saved')
    print('----'*20)

    return model_best_fit

# ------------------------------------------------------------------------------
# -------------------------------Testing Pipeline-------------------------------
# ------------------------------------------------------------------------------
def testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, positional_feature_enabled=False):
    # ---------------------------Loading testing dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading testing dataset...')
    test_dataset = Dataset(DATA_DIR, is_test=True, positional_feature_enabled=positional_feature_enabled)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    model = Network(enableBbox=positional_feature_enabled) # Train Image Branch


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
    for batch_idx, inputs in enumerate(test_loader):
        inputs = tuple(input.to(device) for input in inputs)
        outputs = model(*inputs)
        _, is_correct = LossFcn(*outputs, batch_average_loss=True)
        iFloorPlan = test_dataset.triplets[batch_idx][0].split('/')[5]
        if iFloorPlan in testing_statistics:
            testing_statistics[iFloorPlan]['total'] += 1
            testing_statistics[iFloorPlan]['corrects'] += is_correct.item()
        else:
            testing_statistics.update({iFloorPlan:dict(total=1, corrects=is_correct.item())})

        bar.next()

    bar.finish()
    print('----'*20)
    np.save('testing_statistics.npy', testing_statistics)

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", help="add position feature", action="store_true")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # Train corresponding networks
    Dataset = SanityDataset
    Network = SanityChecker
    LossFcn = TripletLoss()
    TraningFcn = Training
    if args.position:
        checkpoints_prefix = CHECKPOINTS_DIR + 'sg_pos_'
        train_file_names = ['training_statistics_sg.npy', 'training_statistics_sg_pos.npy']
        test_file_names = ['testing_statistics_sg.npy', 'testing_statistics_sg_pos.npy']
    else:
        checkpoints_prefix = CHECKPOINTS_DIR + 'sg_'
        train_file_names = ['training_statistics_sg.npy']
        test_file_names = ['testing_statistics_sg.npy']

    # testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, positional_feature_enabled=args.position)
    # show_testing_histogram_comparison(test_file_names, branch=['SG', 'SG+Pose'], axis_off_set=True)

    # model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn, checkpoints_prefix, positional_feature_enabled=args.position)
    # torch.save(model_best_fit.state_dict(), checkpoints_prefix + 'best_fit.pkl')
    # plot_training_statistics(train_file_names, branch=['SG', 'SG+Pose'])
