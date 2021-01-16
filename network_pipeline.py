# Import params and similarity from lib module
import torch
import argparse, os, copy, pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from progress.bar import Bar
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from termcolor import colored
from lib.robot_env import Agent_Sim
from lib.object_dynamics import shuffle_scene_layout
from lib.params import SCENE_TYPES, SCENE_NUM_PER_TYPE, NODES
from lib.scene_graph_generation import Scene_Graph
from Network.retrieval_network.params import *
from Network.retrieval_network.datasets import TripletImagesDataset, TripletDataset, SiameseDataset, update_triplet_info, get_adj_matrix
from Network.retrieval_network.networks import TripletNetImage, SiameseNetImage, TripletNetSG, RetrievalTriplet, RetrievalSiamese
from Network.retrieval_network.losses import TripletLoss, MSELoss
from Network.retrieval_network.trainer import Training
from os.path import dirname, abspath

# ------------------------------------------------------------------------------
# -------------------------------Training Pipeline------------------------------
# ------------------------------------------------------------------------------
def training_pipeline(Dataset, Network, LossFcn, Training, checkpoints_prefix):
    dataset_sizes = {}
    # ---------------------------Loading training dataset---------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading training dataset...')
    train_dataset = Dataset(DATA_DIR, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataset_sizes.update({'train': len(train_dataset)})

    # --------------------------Loading validation dataset--------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Loading validation dataset...')
    val_dataset = Dataset(DATA_DIR, is_train=False)
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
    # Use best accuracy to determine the best fit if and only if using TripletLoss
    best_acc_criteria = (LossFcn.__class__.__name__ == 'TripletLoss')

    # --------------------------------Training--------------------------------------
    print('----'*20 + '\n' + colored('Network Info: ','blue') + 'Training with dataset size --> ', dataset_sizes)
    data_loaders = {'train': train_loader, 'val': val_loader}
    model_best_fit = Training(device, data_loaders, dataset_sizes, model, loss_fcn, optimizer, lr_scheduler, best_acc_criteria, num_epochs=NUM_EPOCHS, checkpoints_prefix=checkpoints_prefix)

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
            path_prefix = data_dir + '/' + 'round' + str(map_round) + '_' + str(node[0]) + '_' + str(node[1]) + '_' + str(subnode) + '_end'
            img = feature_transforms(Image.open(path_prefix + '.png')).unsqueeze(dim=0)
            sg = np.load(path_prefix + '.npy', allow_pickle='TRUE').item()
            sg = (get_adj_matrix(sg['on']), get_adj_matrix(sg['in']), get_adj_matrix(sg['proximity']))
            node_feature.append(copy.deepcopy(dict(img=img, sg=sg, img_original=Image.open(path_prefix + '.png'))))

        features.append(copy.deepcopy(node_feature))
    return features

# ------------------------------------------------------------------------------
def wrapToPi(angle):
    while angle < -180:
        angle += 180
    while angle > 180:
        angle -= 180
    return angle

# ------------------------------------------------------------------------------
# This function is used to save datapoint into failure-cases folder where the
# localization evaluation is considered not a success at current pose_z
def save_datapoint(scene_name, pose_z, gtSubnodeFeatures, mostSimilarSubnodeFeatures, currentFeatures, similarity):
    file_name = '{}_at_({:.2},{:.2})_with_similarity_{:.2}'.format(scene_name, pose_z[0], pose_z[1], similarity)

    fig, axs = plt.subplots(2, 3, figsize=(17,10))
    plt.axis('off')

    axs[0,2].imshow(currentFeatures['img'])
    axs[0,2].set_title('Current observation')
    axs[0,2].axis('off')
    sg_data = currentFeatures['sg']
    SG = Scene_Graph(R_on=sg_data['on'], R_in=sg_data['in'], R_proximity=sg_data['proximity'])
    SG.visualize_SG(axis=axs[1,2])
    SG.reset()

    data = [gtSubnodeFeatures, mostSimilarSubnodeFeatures]
    legend = ['GroundTrue subnode', 'most Similar Subnode']
    for i in range(2):
        axs[0,i].imshow(data[i]['img_original'])
        axs[0,i].set_title(legend[i])
        axs[0,i].axis('off')
        sg_data = [tensor.numpy().transpose() for tensor in data[i]['sg']]
        SG = Scene_Graph(R_on=sg_data[0], R_in=sg_data[1], R_proximity=sg_data[2])
        SG.visualize_SG(axis=axs[1,i])
        SG.reset()
        axs[1,i].axis('off')

    plt.savefig('./Network/retrieval_network/failure-Cases/' + file_name + '.png')
    plt.close(fig)

# ------------------------------------------------------------------------------
# pose_z = [p['x'], p['z'], subnode]
# info = [is_node, node_i, subnode_i]
def localization_eval(model, robot, device, success_and_trial, staticstics, topo_features, z, pose_z, info):
    feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    COS = torch.nn.CosineSimilarity(dim=0, eps=1e-10)

    z_img = feature_transforms(z['img']).unsqueeze(dim=0)
    z_img = z_img.to(device)
    if 'sg' in z:
        z_sg = (get_adj_matrix(z['sg']['on']), get_adj_matrix(z['sg']['in']), get_adj_matrix(z['sg']['proximity']))
        z_sg = tuple(sg.unsqueeze(dim=0).to(device) for sg in z_sg)

    if info[0]:
        gt_img = topo_features[info[1]][info[2]]['img']
        gt_img = gt_img.to(device)
        if 'sg' in z:
            gt_sg = tuple(sg.unsqueeze(dim=0).to(device) for sg in topo_features[info[1]][info[2]]['sg'])
            benchmark = COS(model.get_embedding(gt_img, *gt_sg, eval=True), model.get_embedding(z_img, *gt_sg, eval=True)).item()
        else:
            benchmark = COS(model.get_embedding(gt_img), model.get_embedding(z_img)).item()
        success_and_trial['trials'] += 1
        is_success = True

    idx_offset = int(staticstics[0]['n'].shape[0] / 2.0)

    for node_i, node_feature in enumerate(topo_features):
        node = NODES[robot._scene_name][node_i]
        d_x = int((pose_z[0]-node[0]) / robot._grid_size) + idx_offset
        d_z = int((pose_z[1]-node[1]) / robot._grid_size) + idx_offset
        for subnode_i, subnode_feature in enumerate(node_feature):
            d_theta = int(wrapToPi((info[2] - subnode_i)*90) / 90)
            if d_theta == 0:
                d_theta_idx = 0
            elif d_theta == -1:
                d_theta_idx = 1
            elif d_theta == 1:
                d_theta_idx = 2
            else:
                d_theta_idx = 3

            subnode_img = subnode_feature['img'].to(device)
            if 'sg' in z:
                subnode_sg = tuple(sg.unsqueeze(dim=0).to(device) for sg in subnode_feature['sg'])
                score = COS(model.get_embedding(subnode_img, *subnode_sg, eval=True), model.get_embedding(z_img, *z_sg, eval=True)).item()
            else:
                score = COS(model.get_embedding(subnode_img), model.get_embedding(z_img)).item()

            staticstics[d_theta_idx]['n'][d_x, d_z] += 1
            staticstics[d_theta_idx]['sum'][d_x, d_z] += score
            staticstics[d_theta_idx]['sq_sum'][d_x, d_z] += score**2

            if info[0] and node_i != info[1] and subnode_i != info[2]:
                if benchmark <= score:
                    is_success = False
                    failureCaseSimilarity = score
                    failureCaseFeatures = copy.deepcopy(subnode_feature)


    if info[0]:
        if is_success:
            success_and_trial['success'] += 1
        else:
            # memo down failure cases
            save_datapoint(robot._scene_name, pose_z, topo_features[info[1]][info[2]], failureCaseFeatures, z, failureCaseSimilarity)

# ------------------------------------------------------------------------------
# Visualize Test Staticstics
def plot_heatmap(staticstics):
    map_len = staticstics[0]['n'].shape[0]
    angles = [0.0, -90.0, 90.0, 180]
    for k in range(4):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        num = copy.deepcopy(staticstics[k]['n'])
        num[staticstics[k]['sum'] == 0] = 1.0
        mean = np.true_divide(staticstics[k]['sum'], num)
        img_mean = ax1.imshow(mean)
        img_mean.set_clim(0,1)
        fig.colorbar(img_mean, ax=ax1, shrink=0.6)

        std = np.true_divide(staticstics[k]['sq_sum'], num) - np.true_divide(np.power(staticstics[k]['sum'], 2), np.power(num, 2))
        sigma = np.power(std, 0.5)
        sigma_max = np.amax(sigma)
        sigma_min = np.amin(sigma)
        sigma = (sigma - sigma_min) / (sigma_max - sigma_min)
        img_sigma = ax2.imshow(sigma)
        img_sigma.set_clim(0,1)
        fig.colorbar(img_sigma, ax=ax2, shrink=0.6)

        for ax, data in zip([ax1, ax2], [mean, sigma]):
            tick_step = 5
            ax.set_xticks(np.arange(0, map_len+1, tick_step))
            ax.set_yticks(np.arange(0, map_len+1, tick_step))
            ax.set_xticklabels(np.arange(0, map_len+1, tick_step) - int(map_len / 2.0))
            ax.set_yticklabels(np.arange(0, map_len+1, tick_step) - int(map_len / 2.0))
            # for i in range(map_len):
            #     for j in range(map_len):
            #         if mean[i,j] >= 0.5:
            #             text = ax.text(i, j, '{:.2f}'.format(int(data[i,j]*100)/100.0), ha="center", va="center", color="k")

        fig.suptitle('Heatmap for view angle difference of {} degree'.format(angles[k]))

    plt.show()

def plot_success(success_and_trials):
    fig, ax1 = plt.subplots()
    tags = []
    success = [data['success'] for data in success_and_trials]
    trials = [data['trials'] for data in success_and_trials]
    for scene_type in SCENE_TYPES:
        for scene_num in range(int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1, SCENE_NUM_PER_TYPE + 1):
            tags.append(scene_type+str(scene_num))

    plt.bar(np.arange(len(success_and_trials)), trials)
    plt.bar(np.arange(len(success_and_trials)), success)
    ax1.set_xticks(np.arange(len(success_and_trials)))
    ax1.set_xticklabels(tags, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(success_and_trials)), np.true_divide(np.array(success), np.array(trials)), 'r--')
    ax2.set_yticks(np.arange(11)/10)
    ax2.set_yticklabels(np.arange(11)/10)
    plt.title('Overall Success Rate {}'.format(np.true_divide(np.sum(np.array(success)), np.sum(np.array(trials)))))

    fig.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# Testing Main
# ------------------------------------------------------------------------------
def testing_pipeline(Network, checkpoint, image_branch=False, dynamcis_rounds=2):
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
    robot = Agent_Sim(applyActionNoise=True)
    test_path = DATA_DIR + '/test'

    # Initialize testing staticstics
    success_and_trials = [dict(trials=0, success=0) for i in range(len(SCENE_TYPES) * SCENE_NUM_PER_TYPE)]
    heatmap_size = (73, 73)
    template = dict(n=np.zeros(heatmap_size), sum=np.zeros(heatmap_size), sq_sum=np.zeros(heatmap_size))
    staticstics = [copy.deepcopy(template) for i in range(4)]
    metric_idx = 0
    # --------------------------------------------------------------------------
    # Iterate through test scene
    # ------------------------------------------------------------------
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
                is_node, node_i = robot.is_node([p['x'], p['z']], threshold=1e-3)
                for subnode_i, subnode in enumerate(subnodes):
                    # Add Gaussian Noise to pose
                    rotation = copy.deepcopy(subnode)
                    rotation['y'] += np.random.randn()*2
                    p['x'] += np.random.randn()*robot._grid_size*0.25
                    p['z'] += np.random.randn()*robot._grid_size*0.25
                    robot._controller.step(action='TeleportFull', x=p['x'], y=p['y'], z=p['z'], rotation=rotation)
                    if image_branch:
                        img_z = robot.get_current_fram()
                        z = copy.deepcopy(dict(img=img_z))
                    else:
                        img_z, sg_z = robot.get_current_data()
                        z = copy.deepcopy(dict(img=img_z, sg=sg_z))

                    observations.append(copy.deepcopy(([p['x'], p['z']], [is_node, node_i, subnode_i], z)))

            # ------------------------------------------------------------------
            # --------------------Start testing in new scene--------------------
            # ------------------------------------------------------------------
            print('----'*20 + '\n' + colored('Testing Info: ','blue') + 'Testing in scene' + robot._scene_name)
            bar = Bar('Processing', max=dynamcis_rounds*len(observations))
            for round in range(dynamcis_rounds):
                # --------------------------------------------------------------
                # get diffrerent info with dynamcis in topological map
                if dynamcis_rounds > 1:
                    topo_features = copy.deepcopy(get_topo_map_features(robot, file_path, map_round=round))
                else:
                    topo_features = copy.deepcopy(get_topo_map_features(robot, file_path, map_round=0))

                # --------------------------------------------------------------
                # Iterate testing through map
                for pose_z, info, z in observations:
                    localization_eval(model, robot, device, success_and_trials[metric_idx], staticstics, topo_features, z, pose_z, info)
                    bar.next()
            bar.finish()
            # ------------------------------------------------------------------
            scene_trials = success_and_trials[metric_idx]['trials']
            scene_success = success_and_trials[metric_idx]['success']
            metric_idx += 1

            print(colored('Testing Info: ','blue') + '{}/{} success rate in scene {}'.format(scene_success, scene_trials, robot._scene_name))
            print('----'*20 + '\n')

            with open('staticstics.pickle', 'wb') as handle:
                pickle.dump(staticstics, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('success_and_trials.pickle', 'wb') as handle:
                pickle.dump(success_and_trials, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_heatmap(staticstics)
    plot_success(success_and_trials)

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # with open('success_and_trials.pickle', 'rb') as f:
    #     success_and_trials = pickle.load(f)
    #     success_and_trials = success_and_trials[0:20]
    #     plot_success(success_and_trials)
    # exit(0)
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_triplet_info", help="update triplets npy list for both image and SG brach for train and validation",
                        action="store_true")
    parser.add_argument("--train", help="train network", action="store_true")
    parser.add_argument("--test", help="test network", action="store_true")
    parser.add_argument("--image", help="network image branch", action="store_true")
    parser.add_argument("--all", help="entire network", action="store_true")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    # --------------------------------------------------------------------------
    # Use to load triplet infomation which is used to collect ground truth
    # when you have new incoming dataset/data
    if args.update_triplet_info:
        update_triplet_info(DATA_DIR, PN_THRESHOLD, TRIPLET_MAX_FRACTION_TO_IMAGES, TRIPLET_MAX_NUM_PER_ANCHOR)

    # --------------------------------------------------------------------------
    # Train corresponding networks
    if args.train:
        if args.image and not args.all:
            Dataset = TripletImagesDataset
            Network = TripletNetImage
            LossFcn = TripletLoss(constant_margin=False)
            checkpoints_prefix = CHECKPOINTS_DIR + 'image_'
        elif args.all and not args.image:
            Dataset = SiameseDataset
            Network = RetrievalSiamese
            LossFcn = MSELoss()
            checkpoints_prefix = CHECKPOINTS_DIR
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        TraningFcn = Training
        model_best_fit = training_pipeline(Dataset, Network, LossFcn, TraningFcn, checkpoints_prefix)
        torch.save(model_best_fit.state_dict(), checkpoints_prefix + 'best_fit.pkl')

    # --------------------------------------------------------------------------
    # Testing corresponding networks
    if args.test:
        if args.image and not args.all:
            Network = SiameseNetImage
            checkpoints_prefix = CHECKPOINTS_DIR + 'image_'
        elif args.all and not args.image:
            Network = RetrievalSiamese
            checkpoints_prefix = CHECKPOINTS_DIR
        else:
            print('----'*20 + '\n' + colored('Network Error: ','red') + 'Please specify a branch (image/all)')

        Checkpoint = checkpoints_prefix + 'best_fit_MSELoss.pkl'
        testing_pipeline(Network, Checkpoint, image_branch=args.image)
