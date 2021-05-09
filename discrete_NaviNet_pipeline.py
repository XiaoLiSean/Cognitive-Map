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
from Network.navigation_network.losses import Cross_Entropy_Loss
from Network.navigation_network.trainer import Training, plot_training_statistics
from os.path import dirname, abspath

# ------------------------------------------------------------------------------
# -------------------------------Testing Pipeline-------------------------------
# ------------------------------------------------------------------------------
def testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=False):
    # ---------------------------Loading testing dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading testing dataset...')
    test_dataset = Dataset(DATA_DIR, is_test=True, load_only_image_data=is_only_image_branch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    model = Network(only_image_branch=is_only_image_branch)
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
# ------------------------------------------------------------------------------
def show_testing_histogram(test_file_name):
    fig, ax1 = plt.subplots()
    testing_statistics = np.load(test_file_name, allow_pickle=True).item()
    tags = []
    total = []
    corrects = []
    for key in testing_statistics:
        tags.append(key)
        total.append(testing_statistics[key]['total'])
        corrects.append(testing_statistics[key]['corrects'])

    plt.bar(np.arange(len(tags)), total)
    plt.bar(np.arange(len(tags)), corrects)
    ax1.set_xticks(np.arange(len(tags)))
    ax1.set_xticklabels(tags, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(tags)), np.true_divide(np.array(corrects), np.array(total)), 'r--')
    ax2.set_yticks(np.arange(11)/10)
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(11)/10])
    plt.title('Overall Success Rate {:.2%}'.format(np.true_divide(np.sum(np.array(corrects)), np.sum(np.array(total)))))

    fig.tight_layout()
    plt.show()
# ------------------------------------------------------------------------------
def show_testing_histogram_comparison(test_file_names, branch=['ResNet', 'Navi-Net'], axis_off_set=False):
    fig, ax1 = plt.subplots(figsize=(6,5))
    img_statistics = np.load(test_file_names[0], allow_pickle=True).item()
    all_statistics = np.load(test_file_names[1], allow_pickle=True).item()
    tags = []
    total = []
    img_corrects = []
    all_corrects = []
    for key in img_statistics:
        tags.append(key)
        total.append(img_statistics[key]['total'])
        img_corrects.append(img_statistics[key]['corrects'])
        all_corrects.append(all_statistics[key]['corrects'])

    plt.bar(np.arange(len(tags)), total, label='Total Cases')
    plt.bar(np.arange(len(tags)) - int(axis_off_set)*0.25, all_corrects, width=0.8 - 0.4*int(axis_off_set), label=branch[1] + ' Success')
    plt.bar(np.arange(len(tags)) + int(axis_off_set)*0.25, img_corrects, width=0.8 - 0.4*int(axis_off_set), label=branch[0] + ' Success')
    ax1.set_xticks(np.arange(len(tags)))
    ax1.set_xticklabels(tags, rotation=90)
    ax1.legend(labels=['Total Cases', branch[1] + ' Success', branch[0] + ' Success'], bbox_to_anchor=(0.40, 0.55))

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(tags)), np.true_divide(np.array(img_corrects), np.array(total)), 'b--')
    ax2.plot(np.arange(len(tags)), np.true_divide(np.array(all_corrects), np.array(total)), 'r--')
    img_label = branch[0] + ' Success Rate: {:.2%} overall'.format(np.true_divide(np.sum(np.array(img_corrects)), np.sum(np.array(total))))
    all_label = branch[1] + ' Success Rate: {:.2%} overall'.format(np.true_divide(np.sum(np.array(all_corrects)), np.sum(np.array(total))))
    ax2.legend(labels=[img_label, all_label], bbox_to_anchor=(0.70, 1.2))
    ax2.set_yticks(np.arange(11)/10)
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(11)/10])
    ax1.grid(True)

    fig.tight_layout()
    plt.show()


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
        Network = NavigationNet
        LossFcn = Cross_Entropy_Loss()
        if args.image and not args.all:
            checkpoints_prefix = CHECKPOINTS_DIR + 'image_'
        elif args.all and not args.image:
            checkpoints_prefix = CHECKPOINTS_DIR
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        TraningFcn = Training
        model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn, checkpoints_prefix, is_only_image_branch=args.image)
        torch.save(model_best_fit.state_dict(), checkpoints_prefix + 'best_fit.pkl')
    # --------------------------------------------------------------------------
    # Test corresponding networks
    if args.test:
        Dataset = NaviDataset
        Network = NavigationNet
        LossFcn = Cross_Entropy_Loss()
        if args.image and not args.all:
            train_file_names = [CHECKPOINTS_DIR+'image_training_history/training_statistics.npy']
            test_file_names = CHECKPOINTS_DIR+'image_testing_statistics.npy'
            checkpoints_prefix = CHECKPOINTS_DIR + 'image_'
        elif args.all and not args.image:
            train_file_names = [CHECKPOINTS_DIR+'image_training_history/training_statistics.npy', CHECKPOINTS_DIR+'training_history/training_statistics.npy']
            test_file_names = [CHECKPOINTS_DIR+'image_testing_statistics.npy', CHECKPOINTS_DIR+'testing_statistics.npy']
            checkpoints_prefix = CHECKPOINTS_DIR
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        plot_training_statistics(train_file_names)
        #testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=args.image)

        if args.image and not args.all:
            show_testing_histogram(test_file_names)
        elif args.all and not args.image:
            show_testing_histogram_comparison(test_file_names)
