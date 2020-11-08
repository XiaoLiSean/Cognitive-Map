# Import params and similarity from lib module
import torch
import argparse, os, copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from termcolor import colored
from lib.robot_env import Agent_Sim
from lib.object_dynamics import shuffle_scene_layout
from lib.params import SCENE_TYPES, SCENE_NUM_PER_TYPE, NODES
from Network.retrieval_network.params import *
from Network.retrieval_network.datasets import TripletImagesDataset, update_triplet_info
from Network.retrieval_network.networks import TripletNetImage, SiameseNetImage
from Network.retrieval_network.losses import TripletLoss
from Network.retrieval_network.trainer import Training
from os.path import dirname, abspath

# ------------------------------------------------------------------------------
# -------------------------------Training Pipeline------------------------------
# ------------------------------------------------------------------------------
def training_pipeline(Dataset, Network, LossFcn, Training):
    dataset_sizes = {}
    # ---------------------------Loading training dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading training dataset...')
    train_dataset = Dataset(DATA_DIR, IMAGE_SIZE, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'train': len(train_dataset)})

    # --------------------------Loading validation dataset--------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading validation dataset...')
    val_dataset = Dataset(DATA_DIR, IMAGE_SIZE, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'val': len(val_dataset)})

    # ------------------------------Initialize model--------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Initialize model...')
    model = Network()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model training on: ", device)
    print("Cuda is_available: ", torch.cuda.is_available())
    model.to(device)

    # Uncomment to see the summary of the model structure
    # summary(model, [(3, IMAGE_SIZE, IMAGE_SIZE), (3, IMAGE_SIZE, IMAGE_SIZE)])

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
    model_best_fit = Training(device, data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS, checkpoints_prefix=CHECKPOINTS_PREFIX)

    # ------------------------------------------------------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Done... Best Fit Model Saved')
    print('----'*20)

    return model_best_fit

# ------------------------------------------------------------------------------
# -------------------------------Testing Pipeline-------------------------------
# ------------------------------------------------------------------------------
def get_topo_map_features(robot, data_dir, map_round=0):
    features = []
    feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    subnode_angles = [0.0, 90.0, 180.0, 270.0]
    for node in NODES[robot._scene_name]:
        node_feature = []
        for subnode in subnode_angles:
            file_name = 'round' + str(map_round) + '_' + str(node[0]) + '_' + str(node[1]) + '_' + str(subnode) + '_' + 'end.png'
            node_feature.append(feature_transforms(Image.open(data_dir + '/' + file_name)).unsqueeze(dim=0))

        features.append(copy.deepcopy(node_feature))
    return features

# ------------------------------------------------------------------------------
def wrapToPi(angle):
    while angle < -180:
        angle += 180
    while angle > 180:
        angle -= 180
    return angle

# pose_z = [p['x'], p['z'], subnode]
# info = [is_node, node_i, subnode_i]
def localization_eval(model, z, topo_features, pose_z, info, success_and_trial, heatmap_max, heatmap_min, robot, device):
    feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    z_img = feature_transforms(z).unsqueeze(dim=0)
    z_img = z_img.to(device)
    if info[0]:
        gt_img = topo_features[info[1]][info[2]]
        gt_img = gt_img.to(device)
        benchmark = COS(model.get_embedding(gt_img), model.get_embedding(z_img)).item()
        success_and_trial['trials'] += 1
        is_success = True

    for node_i, node_feature in enumerate(topo_features):
        node = NODES[robot._scene_name][node_i]
        d_xy = int((np.abs(pose_z[0]-node[0]) + np.abs(pose_z[1]-node[1])) / robot._grid_size)

        for subnode_i, subnode_feature in enumerate(node_feature):
            d_theta_idx = int(wrapToPi((info[2] - subnode_i)*90) / 90) + 2

            subnode_feature = subnode_feature.to(device)
            score = COS(model.get_embedding(subnode_feature), model.get_embedding(z_img)).item()

            if heatmap_max[d_xy][d_theta_idx] < score:
                heatmap_max[d_xy][d_theta_idx] = score
                if d_theta_idx == 0:
                    heatmap_max[d_xy][4] = score
                elif d_theta_idx == 4:
                    heatmap_max[d_xy][0] = score

            if heatmap_min[d_xy][d_theta_idx] == -1:
                heatmap_min[d_xy][d_theta_idx] = score
                if d_theta_idx == 0:
                    heatmap_min[d_xy][4] = score
                elif d_theta_idx == 4:
                    heatmap_min[d_xy][0] = score

            if heatmap_min[d_xy][d_theta_idx] > score:
                heatmap_min[d_xy][d_theta_idx] = score
                if d_theta_idx == 0:
                    heatmap_min[d_xy][4] = score
                elif d_theta_idx == 4:
                    heatmap_min[d_xy][0] = score

            if info[0] and node_i != info[1] and subnode_i != info[2]:
                if benchmark <= score:
                    is_success = False

    if info[0]:
        if is_success:
            success_and_trial['success'] += 1

def plot_heatmap(heatmap_max, heatmap_min):
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    heatmap_max[heatmap_max < 0.0] = 0.0
    heatmap_min[heatmap_min < 0.0] = 0.0
    for row in range(heatmap_max.shape[0]-1,-1,-1):
        if np.sum(heatmap_max[row,:]) == 0.0 and np.sum(heatmap_min[row,:]) == 0.0:
            heatmap_max = np.delete(heatmap_max, row, 0)
            heatmap_min = np.delete(heatmap_min, row, 0)
        else:
            break

    ax1.imshow(np.transpose(heatmap_max))
    ax2.imshow(np.transpose(heatmap_min))
    ax1.set_xticks(np.arange(heatmap_max.shape[0]))
    ax1.set_yticks(np.arange(heatmap_max.shape[1]))
    ax1.set_xticklabels(np.arange(heatmap_max.shape[0]))
    ax1.set_yticklabels([-180, -90, 0, 90, 180])
    ax2.set_xticks(np.arange(heatmap_max.shape[0]))
    ax2.set_yticks(np.arange(heatmap_max.shape[1]))
    ax2.set_xticklabels(np.arange(heatmap_max.shape[0]))
    ax2.set_yticklabels([-180, -90, 0, 90, 180])
    for i in range(heatmap_max.shape[0]):
        for j in range(heatmap_max.shape[1]):
            text = ax1.text(i, j, '{:.2f}'.format(int(heatmap_max[i, j]*100)/100.0), ha="center", va="center", color="k")
            text = ax2.text(i, j, '{:.2f}'.format(int(heatmap_min[i, j]*100)/100.0), ha="center", va="center", color="w")

    plt.show()

# ------------------------------------------------------------------------------
def testing_pipeline(Network, checkpoint, dynamcis_rounds=DYNAMICS_ROUNDS):
    # --------------------------------------------------------------------------
    # Initialize network
    # ------------------------------------------------------------------
    model = Network()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model testing on: ", device)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # ------------------------------------------------------------------
    # Initialize robot
    # ------------------------------------------------------------------
    robot = Agent_Sim()
    test_path = DATA_DIR + '/test'
    # --------------------------------------------------------------------------
    # Iterate through test scene
    # ------------------------------------------------------------------
    success_and_trials = [dict(trials=0, success=0) for i in range(len(SCENE_TYPES) * SCENE_NUM_PER_TYPE)]
    heatmap_max = -1.0*np.ones((60, 5))
    heatmap_min = -1.0*np.ones((60, 5))
    metric_idx = 0
    for scene_type in SCENE_TYPES:
        for scene_num in range(int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1, SCENE_NUM_PER_TYPE + 1):
            robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
            file_path = test_path + '/' + robot._scene_name
            # ------------------------------------------------------------------
            if not os.path.isdir(file_path):
                print(colored('Testing Info: ','blue') + 'No Data for Scene' + robot._scene_name)
                continue
            # ------------------------------------------------------------------
            # Prepare current scene info: i.e. nodes and features (image + SG)
            # --------------------------------------------------------------
            map = robot.get_reachable_coordinate()
            subnodes = [dict(x=0.0, y=0.0, z=0.0), dict(x=0.0, y=90.0, z=0.0), dict(x=0.0, y=180.0, z=0.0), dict(x=0.0, y=270.0, z=0.0)]
            observations = []
            # Image store for every points and angles in map
            for p in map:
                is_node, node_i = robot.is_node([p['x'], p['z']], threshold=1e-8)
                for subnode_i, subnode in enumerate(subnodes):
                    robot._controller.step(action='TeleportFull', x=p['x'], y=p['y'], z=p['z'], rotation=subnode)
                    img_z = robot.get_current_fram()
                    observations.append(([p['x'], p['z']], [is_node, node_i, subnode_i], img_z))
            print('----'*20 + '\n' + colored('Testing Info: ','blue') + 'Testing in scene' + robot._scene_name)
            # ------------------------------------------------------------------
            # --------------------Start testing in new scene--------------------
            # ------------------------------------------------------------------
            for round in range(dynamcis_rounds):
                # --------------------------------------------------------------
                # get diffrerent info with dynamcis in topological map
                if dynamcis_rounds > 1:
                    topo_features = copy.deepcopy(get_topo_map_features(robot, file_path, map_round=round))
                else:
                    topo_features = copy.deepcopy(get_topo_map_features(robot, file_path, map_round=0))
                # --------------------------------------------------------------
                # Iterate testing through map
                for pose_z, info, img_z in observations:
                    localization_eval(model, img_z, topo_features, pose_z, info, success_and_trials[metric_idx], heatmap_max, heatmap_min, robot, device)

            scene_trials = success_and_trials[metric_idx]['trials']
            scene_success = success_and_trials[metric_idx]['success']
            metric_idx += 1

            print(colored('Testing Info: ','blue') + '{}/{} success rate in scene {}'.format(scene_success, scene_trials, robot._scene_name))
            print('----'*20 + '\n')

            np.save("heatmap_max.npy", heatmap_max)
            np.save("heatmap_min.npy", heatmap_min)
            plot_heatmap(heatmap_max, heatmap_min)

    np.save("success_and_trials.npy", success_and_trials)
    plot_heatmap(heatmap_max, heatmap_min)

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_triplet_info", help="update triplets npy list for both image and SG brach for train and validation",
                        action="store_true")
    parser.add_argument("--train", help="train network image branch", action="store_true")
    parser.add_argument("--test", help="test network image branch", action="store_true")
    args = parser.parse_args()

    # heatmap_max = np.load('heatmap_max.npy')
    # heatmap_min = np.load('heatmap_min.npy')
    # plot_heatmap(heatmap_max, heatmap_min)

    # --------------------------------------------------------------------------
    # Use to load triplet infomation which is used to collect ground truth
    # when you have new incoming dataset/data
    if args.update_triplet_info:
        update_triplet_info(DATA_DIR, PN_THRESHOLD, TRIPLET_MAX_FRACTION_TO_IMAGES, TRIPLET_MAX_NUM_PER_ANCHOR)

    # --------------------------------------------------------------------------
    # Train corresponding networks
    if args.train:
        Dataset = TripletImagesDataset
        Network = TripletNetImage
        LossFcn = TripletLoss(constant_margin=False)
        TraningFcn = Training
        model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn)
        torch.save(model_best_fit.state_dict(), CHECKPOINTS_PREFIX + 'model_best_fit.pkl')

    # --------------------------------------------------------------------------
    # Testing corresponding networks
    if args.test:
        Network = SiameseNetImage
        Checkpoint = CHECKPOINTS_PREFIX + 'image_siamese_dynamics_best_fit.pkl'
        testing_pipeline(Network, Checkpoint)
