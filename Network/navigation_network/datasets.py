'''
Retrieval Network,
For robot localization in a dynamic environment.
'''
import numpy as np
import os, sys, random, scipy
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from Network.retrieval_network.datasets import image_pre_transform, get_adj_matrix
from Network.retrieval_network.params import IMAGE_SIZE
from Network.navigation_network.params import DATA_DIR, TRAJECTORY_FILE_NAME
from PIL import Image
from os.path import dirname, abspath
from termcolor import colored
from copy import deepcopy
from lib.scene_graph_generation import Scene_Graph

# ------------------------------------------------------------------------------
# Pair dataset loader
# ------------------------------------------------------------------------------
class NaviDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir=DATA_DIR, trajectory_file_name=TRAJECTORY_FILE_NAME, image_size=IMAGE_SIZE, is_train=False, is_val=False, is_test=False, load_only_image_data=False):
        super(NaviDataset, self).__init__()
        self.data_dir = data_dir
        self.trajectory_file_name = trajectory_file_name
        if not is_train and not is_val and not is_test:
            print('Must specify a network state: \in [train val test]')
        self.network_state = [is_train, is_val, is_test]
        self.image_size = image_size
        self.transforms = image_pre_transform
        self.trajectories, self.actions = self.load_trajectories()
        self.load_only_image_data = load_only_image_data

    def load_trajectories(self):
        trajectories = []
        actions = []
        if self.network_state[0]:
            path = self.data_dir + '/' + 'train'
        elif self.network_state[1]:
            path = self.data_dir + '/' + 'val'
        elif self.network_state[2]:
            path = self.data_dir + '/' + 'test'

        # Iterate through floorplans
        for FloorPlan in os.listdir(path):
            incoming_trajs = np.load(path + '/' + FloorPlan + '/' + self.trajectory_file_name, allow_pickle=True) # load npy list of pairs
            for traj in incoming_trajs:
                curr = path + '/' + FloorPlan + '/' + traj[0]
                goal = path + '/' + FloorPlan + '/' + traj[1]
                action = traj[2]
                trajectories.append(deepcopy((curr, goal)))
                actions.append(deepcopy(np.asarray(action, dtype=np.float32)))

        return trajectories, actions

    def show_data_points(self, imgs, adj_on, adj_in, adj_proximity, fractional_bboxs, paths, action):
        fig, axs = plt.subplots(2, 2, figsize=(17,10))
        fig.suptitle('action {}'.format(action), fontsize=16)

        plt.axis('off')

        SGs = [0,0]

        for i in range(2):
            file_names = paths[i].split('/')
            axs[0,i].set_title(file_names[5])
            SGs[i] = Scene_Graph(R_on=np.transpose(adj_on[i]), R_in=np.transpose(adj_in[i]), R_proximity=np.transpose(adj_proximity[i]), fractional_bboxs=fractional_bboxs[i])
            SGs[i].visualize_one_in_triplet(imgs[i], axs[0,i], axs[1,i])

        plt.show()


    def __getitem__(self, index):
        # Path to triplet data_points
        paths = self.trajectories[index]

        # Load Triplet Images data
        pair_imgs = (self.transforms(Image.open(paths[0]+'.png')),
                     self.transforms(Image.open(paths[1]+'.png')))

        if self.load_only_image_data:
            return pair_imgs, self.actions[index]
        else:
            # Load Triplet SGs data
            A_SG = np.load(paths[0] + '.npy', allow_pickle='TRUE').item()
            B_SG = np.load(paths[1] + '.npy', allow_pickle='TRUE').item()

            adj_on = (get_adj_matrix(A_SG['on']), get_adj_matrix(B_SG['on']))
            adj_in = (get_adj_matrix(A_SG['in']), get_adj_matrix(B_SG['in']))
            adj_proximity = (get_adj_matrix(A_SG['proximity']), get_adj_matrix(B_SG['proximity']))

            fractional_bboxs = (np.asarray(A_SG['fractional_bboxs'], dtype=np.float32),
                                np.asarray(B_SG['fractional_bboxs'], dtype=np.float32))

            obj_occurence_vecs = (np.asarray(A_SG['vec'].todense(), dtype=np.float32),
                                  np.asarray(B_SG['vec'].todense(), dtype=np.float32))

            # self.show_data_points((Image.open(paths[0]+'.png'), Image.open(paths[1]+'.png')), adj_on, adj_in, adj_proximity, fractional_bboxs, paths, self.actions[index])

            return pair_imgs + adj_on + adj_in + adj_proximity + fractional_bboxs + obj_occurence_vecs, self.actions[index]

    def __len__(self):
        return len(self.trajectories)
