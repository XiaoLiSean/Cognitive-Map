'''
Retrieval Network Testing in Discrete World, Written by Xiao
For robot localization in a dynamic environment.
'''
# Import params and similarity from lib module
import torch
import argparse, os, copy, pickle, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from progress.bar import Bar
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from termcolor import colored
from lib.scene_graph_generation import Scene_Graph
from Network.retrieval_network.params import *
from Network.retrieval_network.datasets import TripletDataset, PairDataset
from Network.retrieval_network.networks import RetrievalTriplet, TripletNetImage
from Network.retrieval_network.losses import TripletLoss, CosineSimilarity
from Network.retrieval_network.trainer import Training, plot_training_statistics
from os.path import dirname, abspath

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
    if is_only_image_branch:
        model = Network(pretrainedXXXNet=True, XXXNetName=benchmark) # Train Image Branch
    else:
        model = Network(self_pretrained_image=False, pretrainedXXXNet=True)
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
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Decay LR by a factor of 0.1 every 7 epochs
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
# -------------------------------Testing Pipeline-------------------------------
# ------------------------------------------------------------------------------
# all_fail_cases = np.load('./Network/retrieval_network/checkpoints/fail and success cases/all_fail_cases.npy', allow_pickle=True).item()
# ------------------------------------------------------------------------------
def testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=False, benchmark=None):
    # ---------------------------Loading testing dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading testing dataset...')
    test_dataset = Dataset(DATA_DIR, is_test=True, load_only_image_data=is_only_image_branch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    if is_only_image_branch:
        model = Network(pretrainedXXXNet=True, XXXNetName=benchmark) # Train Image Branch
    else:
        model = Network(self_pretrained_image=False, pretrainedXXXNet=True)

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
        iFloorPlan = test_dataset.triplets[batch_idx][0].split('/')[4]
        if iFloorPlan in testing_statistics:
            testing_statistics[iFloorPlan]['total'] += 1
            testing_statistics[iFloorPlan]['corrects'] += is_correct.item()
        else:
            testing_statistics.update({iFloorPlan:dict(total=1, corrects=is_correct.item())})

        bar.next()

    bar.finish()
    print('----'*20)
    np.save(checkpoints_prefix+'testing_statistics.npy', testing_statistics)
    # np.save('all_fail_cases.npy', all_fail_cases)
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
def show_testing_histogram_comparison(parent_dir=CHECKPOINTS_DIR,filename='testing_statistics.npy'):
    fig, ax1 = plt.subplots(figsize=(6,5))
    ax2 = ax1.twinx()
    i = 0
    num = len(os.listdir(parent_dir))
    labels = []
    for xxxnet in os.listdir(parent_dir):
        if not os.path.exists(parent_dir+xxxnet+'/'+filename):
            continue
        testing_statistics = np.load(parent_dir+xxxnet+'/'+filename, allow_pickle=True).item()
        tags = []
        total = []
        corrects = []
        for key in testing_statistics:
            tags.append(key)
            total.append(testing_statistics[key]['total'])
            corrects.append(testing_statistics[key]['corrects'])
        ax1.bar(np.arange(len(tags))*(num+1)-(num-i), corrects, width=1)
        ax2.plot(np.arange(len(tags))*(num+1)-(num-i), np.true_divide(np.array(corrects), np.array(total)))
        i += 1
        labels.append(xxxnet + ': {:.2%} Success'.format(np.true_divide(np.sum(np.array(corrects)), np.sum(np.array(total)))))
        print('{}: {}/{}'.format(xxxnet, np.sum(np.array(corrects)), np.sum(np.array(total))))

    # ax1.legend(labels=labels, bbox_to_anchor=(0.40, 0.55))
    ax1.legend(labels=labels, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=num)
    ax1.set_xticks(np.arange(len(tags))*(num+1)-num/2.0)
    ax1.set_xticklabels(tags, rotation=90)

    # ax2.legend(labels=[img_label, all_label], bbox_to_anchor=(0.70, 1.2))
    ax2.set_yticks(np.arange(11)/10)
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(11)/10])
    ax1.grid(True)
    fig.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
def store_success_case(checkpoints_prefix, triplet_name):
    fig, axs = plt.subplots(1,3)
    plt.title(triplet_name[0].split('/')[5])
    for i in range(3):
        img = Image.open(triplet_name[i]+'.png')
        axs[i].imshow(img)
        axs[i].set_title(triplet_name[i].split('/')[6])
        axs[i].axis('off')

    file_name = triplet_name[0].split('/')[5] + triplet_name[0].split('/')[6] + triplet_name[1].split('/')[6] + triplet_name[2].split('/')[6]
    for key in all_fail_cases:
        if triplet_name in all_fail_cases[key]:
            file_name = key + '+' + file_name
    plt.savefig(checkpoints_prefix + 'unique_success_cases' + '/' + file_name + '.jpg')
    plt.close()

# ------------------------------------------------------------------------------
# ---------------------------Thresholding and Heatmap---------------------------
# ------------------------------------------------------------------------------
def thresholding(Dataset, Network, checkpoints_prefix, is_only_image_branch=False, benchmark=None):
    # ---------------------------Loading testing dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading thresholding/val dataset...')
    dataset = Dataset(DATA_DIR, is_val=True, load_only_image_data=is_only_image_branch)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    if is_only_image_branch:
        model = Network(pretrainedXXXNet=True, XXXNetName=benchmark) # Train Image Branch
    else:
        model = Network(self_pretrained_image=False, pretrainedXXXNet=True)

    model.load_state_dict(torch.load(checkpoints_prefix + 'best_fit.pkl'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model thresholding on: ", device)
    print("Cuda is_available: ", torch.cuda.is_available())
    print('----'*20)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    # ------------------------------Thresholding Main------------------------------------
    heatmap_size = (36, 36)
    staticstics = dict(n=np.zeros(heatmap_size), sum=np.zeros(heatmap_size), sq_sum=np.zeros(heatmap_size))

    bar = Bar('Processing', max=len(loader))
    for batch_idx, inputs in enumerate(loader):
        A = tuple(input.to(device) for idx, input in enumerate(inputs) if idx%2 == 0)
        B = tuple(input.to(device) for idx, input in enumerate(inputs) if idx%2 == 1)
        Vec_A = model.get_embedding(*A)
        Vec_B = model.get_embedding(*B)
        score = CosineSimilarity(Vec_A, Vec_B).item()
        x_A = dataset.pairs[batch_idx][0].split('_')[1]
        z_A = dataset.pairs[batch_idx][0].split('_')[2]
        x_B = dataset.pairs[batch_idx][1].split('_')[1]
        z_B = dataset.pairs[batch_idx][1].split('_')[2]

        d_x = abs(int(x_A)-int(x_B))
        d_z = abs(int(z_A)-int(z_B))
        staticstics['n'][d_x, d_z] += 1
        staticstics['sum'][d_x, d_z] += score
        staticstics['sq_sum'][d_x, d_z] += score**2

        bar.next()

    bar.finish()
    print('----'*20)
    np.save(checkpoints_prefix + 'thresholding_staticstics.npy', staticstics)
    plot_heatmap(staticstics, save_dir=checkpoints_prefix)

# ------------------------------------------------------------------------------
# Visualize Test Staticstics
def plot_heatmap(staticstics, save_dir=None):
    # map_len = 2*staticstics['n'].shape[0] - 1
    # mid_idx = staticstics['n'].shape[0] - 1
    span_len = 21
    map_len = span_len*2 - 1
    mid_idx = span_len - 1

    # Processing the staticstics data to make a symetrical heatmap
    new_staticstics = dict(n=np.zeros((map_len, map_len)), sum=np.zeros((map_len, map_len)), sq_sum=np.zeros((map_len, map_len)))
    for key in staticstics:
        new_staticstics[key][mid_idx:, mid_idx:] = staticstics[key][0:mid_idx+1, 0:mid_idx+1]
        new_staticstics[key][0:mid_idx+1, mid_idx:] = np.flip(staticstics[key][0:mid_idx+1, 0:mid_idx+1], 0)
        new_staticstics[key][mid_idx:, 0:mid_idx+1] = np.flip(staticstics[key][0:mid_idx+1, 0:mid_idx+1], 1)
        new_staticstics[key][0:mid_idx+1, 0:mid_idx+1] = np.flip(staticstics[key][0:mid_idx+1, 0:mid_idx+1], (0,1))

    staticstics = copy.deepcopy(new_staticstics)

    # Calculate staticstics
    num = copy.deepcopy(staticstics['n'])
    num[staticstics['sum'] == 0] = 1.0
    mean = np.true_divide(staticstics['sum'], num)
    std = np.true_divide(staticstics['sq_sum'], num) - np.true_divide(np.power(staticstics['sum'], 2), np.power(num, 2))
    sigma = np.power(std, 0.5)

    # Plot heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,8))
    img_mean = ax1.imshow(mean, cmap=cm.coolwarm)
    img_mean.set_clim(0,1)
    fig.colorbar(img_mean, ax=ax1, shrink=0.6)
    img_sigma = ax2.imshow(sigma, cmap=cm.coolwarm)
    fig.colorbar(img_sigma, ax=ax2, shrink=0.6)

    print(mean[int((map_len-1)*0.5):,int((map_len-1)*0.5):])
    print(sigma[int((map_len-1)*0.5):,int((map_len-1)*0.5):])
    print(staticstics['n'][int((map_len-1)*0.5):, int((map_len-1)*0.5):], )

    # configure heatmap
    for ax, data in zip([ax1, ax2], [mean, sigma]):
        tick_step = 4
        ax.set_xticks(np.arange(0, map_len+1, tick_step))
        ax.set_yticks(np.arange(0, map_len+1, tick_step))
        ax.set_xticklabels(np.arange(0, map_len+1, tick_step)-mid_idx)
        ax.set_yticklabels(np.arange(0, map_len+1, tick_step)-mid_idx)
        ax.set_xlabel('Deviation from anchor along x-axis [in unit of $d$]')
        ax.set_ylabel('Deviation from anchor along y-axis [in unit of $d$]')
    # fig.suptitle('Heatmap for view angle difference of {} degree'.format(angles[k]))

    plt.savefig(save_dir+"heatmap.svg")
    plt.show()

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
    # show_testing_histogram_comparison(parent_dir=CHECKPOINTS_DIR,filename='testing_statistics.npy')
    # --------------------------------------------------------------------------
    # Train corresponding networks
    if args.train:
        Dataset = TripletDataset
        LossFcn = TripletLoss()
        if args.benchmark and not args.rnet:
            Network = TripletNetImage
            checkpoints_prefix = CHECKPOINTS_DIR + args.name + '/'
        elif args.rnet and not args.benchmark:
            Network = RetrievalTriplet
            checkpoints_prefix = CHECKPOINTS_DIR + 'rnet/'
            args.name = 'rnet'
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (rnet/benchmark)')

        TraningFcn = Training
        model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn, checkpoints_prefix, is_only_image_branch=args.benchmark, benchmark=args.name)
        torch.save(model_best_fit.state_dict(), checkpoints_prefix + 'best_fit.pkl')
        plot_training_statistics(parent_dir=CHECKPOINTS_DIR,filename='training_statistics.npy')

    # --------------------------------------------------------------------------
    # Testing corresponding networks
    if args.test:
        Dataset = TripletDataset
        LossFcn = TripletLoss()
        if args.benchmark and not args.rnet:
            Network = TripletNetImage
            checkpoints_prefix = CHECKPOINTS_DIR + args.name + '/'
        elif args.rnet and not args.benchmark:
            Network = RetrievalTriplet
            checkpoints_prefix = CHECKPOINTS_DIR + 'rnet/'
            args.name = 'rnet'
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        testing_pipeline(Dataset, Network, LossFcn, checkpoints_prefix, is_only_image_branch=args.benchmark, benchmark=args.name)
        show_testing_histogram(checkpoints_prefix+'testing_statistics.npy')
        show_testing_histogram_comparison(parent_dir=CHECKPOINTS_DIR,filename='testing_statistics.npy')

    # --------------------------------------------------------------------------
    # Plot Heatmap and determine threshold for localization
    if args.heatmap:
        Dataset = PairDataset
        if args.benchmark and not args.rnet:
            Network = TripletNetImage
            checkpoints_prefix = CHECKPOINTS_DIR + args.name + '/'
        elif args.rnet and not args.benchmark:
            Network = RetrievalTriplet
            checkpoints_prefix = CHECKPOINTS_DIR + 'rnet/'
            args.name = 'rnet'
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        #thresholding(Dataset, Network, checkpoints_prefix, is_only_image_branch=args.benchmark,  benchmark=args.name)
        staticstics = np.load(checkpoints_prefix + 'thresholding_staticstics.npy', allow_pickle=True).item()
        plot_heatmap(staticstics, save_dir=checkpoints_prefix)
