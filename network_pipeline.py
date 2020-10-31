# Import params and similarity from lib module
import torch
import argparse, os, copy
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from termcolor import colored
from lib.robot_patrol import Agent_Sim
from lib.object_dynamics import shuffle_scene_layout
from lib.params import SCENE_TYPES, SCENE_NUM_PER_TYPE
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
def get_nodes_features(robot, data_dir):
    features = []
    feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    round = np.random.randint(DYNAMICS_ROUNDS)
    subnode_angles = [0.0, 90.0, 180.0, 270.0]
    for node in NODES[robot._scene_name]:
        node_feature = []
        for subnode in subnode_angles:
            file_name = 'round' + str(round) + '_' + str(node[0]) + '_' + str(node[1]) + '_' + str(subnode) + '_' + 'end.png'
            node_feature.append(feature_transforms(Image.open(data_dir + '/' + file_name)).unsqueeze(dim=0))

        features.append(node_feature)
    return features

def localization(model, z, features, gt):
    feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    is_localized = True
    gt_img = features[gt[0]][gt[1]]
    z_img = feature_transforms(z).unsqueeze(dim=0)
    benchmark = COS(model.get_embedding(gt_img), model.get_embedding(z_img)).item()

    for node_i, node_feature in enumerate(features):
        for subnode_i, subnode_feature in enumerate(node_feature):
            if node_i == gt[0] and subnode_i == gt[1]:
                continue
            score = COS(model.get_embedding(subnode_feature), model.get_embedding(z_img)).item()
            if score >= benchmark:
                is_localized = False
                break
        if not is_localized:
            break

    return is_localized

# ------------------------------------------------------------------------------
def testing_pipeline(Network, checkpoint):
    # --------------------------------------------------------------------------
    # Initialize network
    model = Network()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # ------------------------------------------------------------------
    # Initialize robot
    robot = Agent_Sim()
    test_path = DATA_DIR + '/test'
    # --------------------------------------------------------------------------
    # Iterate through test scene
    for scene_type in SCENE_TYPES:
        for scene_num in range(int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1, SCENE_NUM_PER_TYPE + 1):
            robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
            file_path = test_path + '/' + robot._scene_name
            # ------------------------------------------------------------------
            if not os.path.isdir(file_path):
                print(colored('Testing Info: ','blue') + 'No Data for Scene' + robot._scene_name)
                continue
            # ------------------------------------------------------------------
            # Prepare Topo-map info: i.e. nodes and features (image + SG)
            print('----'*20 + '\n' + colored('Testing Info: ','blue') + 'Testing in scene' + robot._scene_name)
            features = get_nodes_features(robot, file_path)
            subnodes = [dict(x=0.0, y=0.0, z=0.0), dict(x=0.0, y=90.0, z=0.0), dict(x=0.0, y=180.0, z=0.0), dict(x=0.0, y=270.0, z=0.0)]
            # --------------------Start testing in new scene--------------------
            # Initialize test metrics
            scene_trials = 0
            scene_success = 0
            # ------------------------------------------------------------------
            for round in range(DYNAMICS_ROUNDS):
                # --------------------------------------------------------------
                # change object layout
                if round != 0:
                    shuffle_scene_layout(robot._controller)
                    robot.update_event()
                # --------------------------------------------------------------
                # Get and prepare map info
                map = robot.get_reachable_coordinate()
                universal_y = robot.get_agent_position()['y']
                # --------------------------------------------------------------
                # Iterate testing through subnodes
                for node_i, node in enumerate(NODES[robot._scene_name]):
                    for subnode_i, subnode in enumerate(subnodes):
                        # ------------------------------------------------------
                        # Get testing through subnodes grid
                        points = robot.get_near_grids({'x': node[0], 'y': universal_y, 'z': node[1]}, step=LOCALIZATION_GRID_TOL)
                        grids = copy.deepcopy(points)
                        for grid in grids:
                            if grid not in map:
                                points.remove(grid)
                        if len(points) <= 0:
                            continue
                        # ------------------------------------------------------
                        # Iterate testing through subnodes grid
                        for p in points:
                            robot._controller.step(action='TeleportFull', x=p['x'], y=p['y'], z=p['z'], rotation=subnode)
                            z = robot.get_current_fram()
                            is_localized = localization(model, z, features, (node_i, subnode_i))
                            # Update testing data and metrics
                            scene_trials += 1
                            scene_success += int(is_localized)

            print(colored('Testing Info: ','blue') + '{}/{} ({}) success rate in scene {}'.format(scene_success, scene_trials, scene_success/scene_trials , robot._scene_name))
            print('----'*20 + '\n')

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
        torch.save(model_best_fit.state_dict(), CHECKPOINTS_PREFIX + '_best_fit.pkl')

    # --------------------------------------------------------------------------
    # Testing corresponding networks
    if args.test:
        Network = SiameseNetImage
        Checkpoint = CHECKPOINTS_PREFIX + '_best_fit.pkl'
        testing_pipeline(Network, Checkpoint)
