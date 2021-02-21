import numpy as np
import os, sys, random, scipy
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import find as find_sparse_idx
from torchvision import datasets, transforms
from Network.retrieval_network.params import IMAGE_SIZE, DATA_DIR, TRIPLET_FILE_NAME
from PIL import Image
from os.path import dirname, abspath
from termcolor import colored
from copy import deepcopy
from lib.params import OBJ_TYPE_NUM
from lib.similarity import view_similarity
from lib.scene_graph_generation import Scene_Graph

# ------------------------------------------------------------------------------
image_pre_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ------------------------------------------------------------------------------
# Third party function from git@github.com:tkipf/pygcn.git
def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # error msg: Integers to negative integer powers are not allowed.
    # r_inv = np.power(rowsum, -1).flatten()
    with np.errstate(divide='ignore'):
        r_inv = np.divide(1, rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

# Third party function from git@github.com:tkipf/pygcn.git
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not scipy.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
# ------------------------------------------------------------------------------
def get_adj_matrix(SG):
    adj = SG.transpose()
    adj = adj.tocoo()
    # presents of i'th object should be encode as adj[i,i] = 1
    # thus adding a identity matrix is inproper (as in following line)
    # adj = row_normalize(adj + scipy.sparse.eye(adj.shape[0]))
    # The correct adj is modified in scene_graph_generation--> scene_graph class
    adj = row_normalize(adj)

    return sparse_mx_to_torch_sparse_tensor(adj).to_dense()

# ------------------------------------------------------------------------------
# Triplet dataset loader
# ------------------------------------------------------------------------------
class TripletDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, data_dir=DATA_DIR, triplet_file_name=TRIPLET_FILE_NAME, image_size=IMAGE_SIZE, is_train=False, is_val=False, is_test=False):
        super(TripletDataset, self).__init__()
        self.data_dir = data_dir
        self.triplet_file_name = triplet_file_name
        if not is_train and not is_val and not is_test:
            print('Must specify a network state: \in [train val test]')
        self.network_state = [is_train, is_val, is_test]
        self.image_size = image_size
        self.transforms = image_pre_transform
        self.triplets = self.load_triplets()

    def load_triplets(self):
        triplets_data = []
        if self.network_state[0]:
            path = self.data_dir + '/' + 'train'
        elif self.network_state[1]:
            path = self.data_dir + '/' + 'val'
        elif self.network_state[2]:
            path = self.data_dir + '/' + 'test'

        # Iterate through floorplans
        for FloorPlan in os.listdir(path):
            incoming_triplets = np.load(path + '/' + FloorPlan + '/' + self.triplet_file_name) # load npy list of triplets
            for triplet in incoming_triplets:
                anchor = path + '/' + FloorPlan + '/' + triplet[0]
                positive = path + '/' + FloorPlan + '/' + triplet[1]
                negative = path + '/' + FloorPlan + '/' + triplet[2]
                triplets_data.append(deepcopy((anchor, positive, negative)))

        return triplets_data

    def show_data_points(self, imgs, adj_on, adj_in, adj_proximity, fractional_bboxs, paths):
        fig, axs = plt.subplots(2, 3, figsize=(17,10))
        plt.axis('off')

        SGs = [0,0,0]

        for i in range(3):
            file_names = paths[i].split('/')
            axs[0,i].set_title(file_names[6])
            SGs[i] = Scene_Graph(R_on=np.transpose(adj_on[i]), R_in=np.transpose(adj_in[i]), R_proximity=np.transpose(adj_proximity[i]), fractional_bboxs=fractional_bboxs[i])
            SGs[i].visualize_one_in_triplet(imgs[i], axs[0,i], axs[1,i])

        plt.show()


    def __getitem__(self, index):
        # Path to triplet data_points
        paths = self.triplets[index]

        # Load Triplet Images data
        triplet_imgs = (self.transforms(Image.open(paths[0]+'.png')),
                        self.transforms(Image.open(paths[1]+'.png')),
                        self.transforms(Image.open(paths[2]+'.png')))

        # Load Triplet SGs data
        anchor_SG = np.load(paths[0] + '.npy', allow_pickle='TRUE').item()
        positive_SG = np.load(paths[1] + '.npy', allow_pickle='TRUE').item()
        negative_SG = np.load(paths[2] + '.npy', allow_pickle='TRUE').item()

        adj_on = (get_adj_matrix(anchor_SG['on']), get_adj_matrix(positive_SG['on']), get_adj_matrix(negative_SG['on']))
        adj_in = (get_adj_matrix(anchor_SG['in']), get_adj_matrix(positive_SG['in']), get_adj_matrix(negative_SG['in']))
        adj_proximity = (get_adj_matrix(anchor_SG['proximity']), get_adj_matrix(positive_SG['proximity']), get_adj_matrix(negative_SG['proximity']))

        fractional_bboxs = (np.asarray(anchor_SG['fractional_bboxs'], dtype=np.float32),
                            np.asarray(positive_SG['fractional_bboxs'], dtype=np.float32),
                            np.asarray(negative_SG['fractional_bboxs'], dtype=np.float32))

        obj_occurence_vecs = (np.asarray(anchor_SG['vec'].todense(), dtype=np.float32),
                              np.asarray(positive_SG['vec'].todense(), dtype=np.float32),
                              np.asarray(negative_SG['vec'].todense(), dtype=np.float32))

        # self.show_data_points((Image.open(paths[0]+'.png'), Image.open(paths[1]+'.png'), Image.open(paths[2]+'.png')), adj_on, adj_in, adj_proximity, fractional_bboxs, paths)

        return triplet_imgs + adj_on + adj_in + adj_proximity + fractional_bboxs + obj_occurence_vecs

    def __len__(self):
        return len(self.triplets)
