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
from Network.retrieval_network.datasets import TripletDataset
from Network.retrieval_network.networks import RetrievalTriplet, TripletNetImage
from Network.retrieval_network.losses import TripletLoss
from Network.retrieval_network.trainer import Training, plot_training_statistics
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
    if is_only_image_branch:
        model = Network(enableRoIBridge=False) # Train Image Branch
    else:
        model = Network(self_pretrained_image=True) # freeze image branch and train SG
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
    model_best_fit = Training(data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS, checkpoints_prefix=checkpoints_prefix)

    # ------------------------------------------------------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Done... Best Fit Model Saved')
    print('----'*20)

    return model_best_fit

# ------------------------------------------------------------------------------
# -------------------------------Testing Pipeline-------------------------------
# ------------------------------------------------------------------------------
def store_fail_case(checkpoints_prefix, triplet_name):
    fig, axs = plt.subplots(1,3)
    plt.title(triplet_name[0].split('/')[5])
    for i in range(3):
        img = Image.open(triplet_name[i]+'.png')
        axs[i].imshow(img)
        axs[i].set_title(triplet_name[i].split('/')[6])
        axs[i].axis('off')

    file_name = triplet_name[0].split('/')[5] + triplet_name[0].split('/')[6] + triplet_name[1].split('/')[6] + triplet_name[2].split('/')[6]
    plt.savefig(checkpoints_prefix + 'failCase' + '/' + file_name + '.jpg')
    plt.close()

def show_testing_histogram(testing_statistics):
    fig, ax1 = plt.subplots()
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
    ax2.set_yticklabels(np.arange(11)/10)
    plt.title('Overall Success Rate {:.2%}'.format(np.true_divide(np.sum(np.array(corrects)), np.sum(np.array(total)))))

    fig.tight_layout()
    plt.show()

def testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=False):
    # ---------------------------Loading testing dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading testing dataset...')
    test_dataset = Dataset(DATA_DIR, is_test=True, load_only_image_data=is_only_image_branch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    if is_only_image_branch:
        model = Network(enableRoIBridge=False) # Train Image Branch
    else:
        model = Network(self_pretrained_image=True) # freeze image branch and train SG

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

        if is_correct.item() == 0:
            triplet_name = test_dataset.triplets[batch_idx]
            store_fail_case(checkpoints_prefix, triplet_name)
        bar.next()

    bar.finish()
    print('----'*20)
    show_testing_histogram(testing_statistics)



# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train network", action="store_true")
    parser.add_argument("--test", help="test network", action="store_true")
    parser.add_argument("--image", help="network image branch", action="store_true")
    parser.add_argument("--all", help="entire network", action="store_true")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # Train corresponding networks
    if args.train:
        if args.image and not args.all:
            Dataset = TripletDataset
            Network = TripletNetImage
            LossFcn = TripletLoss()
            checkpoints_prefix = CHECKPOINTS_DIR + 'image_'
        elif args.all and not args.image:
            Dataset = TripletDataset
            Network = RetrievalTriplet
            LossFcn = TripletLoss()
            checkpoints_prefix = None
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        TraningFcn = Training
        model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn, checkpoints_prefix, is_only_image_branch=args.image)
        torch.save(model_best_fit.state_dict(), checkpoints_prefix + 'best_fit.pkl')

    # --------------------------------------------------------------------------
    # Testing corresponding networks
    if args.test:
        if args.image and not args.all:
            Dataset = TripletDataset
            Network = TripletNetImage
            LossFcn = TripletLoss()
            checkpoints_prefix = CHECKPOINTS_DIR + 'image_'
        elif args.all and not args.image:
            Dataset = TripletDataset
            Network = RetrievalTriplet
            LossFcn = TripletLoss()
            checkpoints_prefix = None
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=args.image)
