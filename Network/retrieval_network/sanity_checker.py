'''
Retrieval Network, Written by Xiao
For robot localization in a dynamic environment.
This file is used to check the effeciveness of the scene graph branch solely in localization
'''
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models, ops
from lib.params import OBJ_TYPE_NUM
from Network.retrieval_network.params import BBOX_EMDEDDING_VEC_LENGTH, WORD_EMDEDDING_VEC_LENGTH, OBJ_FEATURE_VEC_LENGTH
from Network.retrieval_network.params import GCN_TIER, DROPOUT_RATE, IMAGE_SIZE, DATA_DIR, TRIPLET_FILE_NAME
from Network.retrieval_network.networks import TripletNetSG, get_glove_matrix, positional_encoding, denseTensor_to_SparseTensor
from Network.retrieval_network.datasets import get_adj_matrix
from copy import deepcopy
import math, os
import numpy as np

class SanityChecker(torch.nn.Module):
    def __init__(self, GCN_dropout_rate=DROPOUT_RATE, GCN_layers=GCN_TIER, GCN_bias=True, self_pretrained_image=True, enableBbox=False):
        super(SanityChecker, self).__init__()
        self.SG_branch = TripletNetSG(dropout_rate=GCN_dropout_rate, layer_structure=GCN_layers, bias=GCN_bias)
        self.enableBbox = enableBbox
        self.RoIBridge = RoIBridge(enableBbox=enableBbox)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    '''Prepare batch feature matrices for GCN as Batch Inputs '''
    # --------------------------------------------------------------------------
    def get_embedding(self, R_on, R_in, R_prox, batch_fractional_bboxs=None, batch_obj_vecs=None):

        GlV_features = self.RoIBridge.get_GlV_features(R_on.shape[0])
        raw_features = GlV_features.detach().clone()

        if self.enableBbox:

            batch_obj_vecs = torch.squeeze(batch_obj_vecs, dim=2)
            PoE_features = self.RoIBridge.get_PoE_features(batch_fractional_bboxs, batch_obj_vecs)

            # Detach to double ensure no gradient propagates back to ResNet and RoI except for the feature_fcn
            raw_features = torch.cat((PoE_features, GlV_features), dim=1).detach().clone()

        batch_feature_matrices = self.RoIBridge.feature_fcn(raw_features)
        batch_feature_matrices = torch.stack(torch.split(batch_feature_matrices, OBJ_TYPE_NUM, dim=0), dim=0)
        sg_embedding = self.SG_branch.get_embedding(R_on, R_in, R_prox, batch_feature_matrices)

        return sg_embedding

    # --------------------------------------------------------------------------
    def forward(self, A_on, P_on, N_on, A_in, P_in, N_in, A_prox, P_prox, N_prox,
                A_fractional_bboxs=None, P_fractional_bboxs=None, N_fractional_bboxs=None,
                A_vecs=None, P_vecs=None, N_vecs=None):

        if self.enableBbox:
            A_vec = self.get_embedding(A_on, A_in, A_prox, A_fractional_bboxs, A_vecs)
            P_vec = self.get_embedding(P_on, P_in, P_prox, P_fractional_bboxs, P_vecs)
            N_vec = self.get_embedding(N_on, N_in, N_prox, N_fractional_bboxs, N_vecs)
        else:
            A_vec = self.get_embedding(A_on, A_in, A_prox)
            P_vec = self.get_embedding(P_on, P_in, P_prox)
            N_vec = self.get_embedding(N_on, N_in, N_prox)
        return A_vec, N_vec, P_vec

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class RoIBridge(torch.nn.Module):
    def __init__(self, enableBbox=False):
        super(RoIBridge, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Used to embed (roi bounding box + roi visual feature + glove word) as one object feature vector
        # Note: this layer will automatically initialize the weights and bias using uniform_(-stdv, stdv)
        self.feature_fcn = torch.nn.Sequential(
                                    torch.nn.Linear(BBOX_EMDEDDING_VEC_LENGTH*int(enableBbox)+WORD_EMDEDDING_VEC_LENGTH, OBJ_FEATURE_VEC_LENGTH, bias=True),
                                    torch.nn.ReLU(inplace=True)
                                              )
        # size of OBJ_TYPE_NUM * WORD_EMDEDDING_VEC_LENGTH = (256,300)
        self.word_embedding_matrix = get_glove_matrix()
        # size of (IMAGE_SIZE+1) * (BBOX_EMDEDDING_VEC_LENGTH/4) = (225,64) and PE(i) = self.position_embedding_matrix[i,:]
        self.position_embedding_matrix = positional_encoding()
    # --------------------------------------------------------------------------
    '''
    Functions need to avoid for-loop (this one and above one)
    which will cause intense communication between GPU and CPu
    '''
    def get_PoE_features(self, batch_fractional_bboxs, batch_obj_vecs):
        local_batch_size = batch_fractional_bboxs.shape[0]
        PoE_features = torch.zeros(local_batch_size*OBJ_TYPE_NUM, BBOX_EMDEDDING_VEC_LENGTH)
        for i in range(local_batch_size):
            for j in range(OBJ_TYPE_NUM):
                idx = j + OBJ_TYPE_NUM*i
                if batch_obj_vecs[i][j] == 1:
                    bbox_embedding = [self.position_embedding_matrix[int(torch.clamp(frac*IMAGE_SIZE, 0, IMAGE_SIZE)),:] for frac in batch_fractional_bboxs[i][j]]
                    PoE_features[idx,:] = torch.flatten(torch.FloatTensor(bbox_embedding))
                else:
                    pass
        PoE_features = PoE_features.to(self.device)
        return denseTensor_to_SparseTensor(PoE_features)

    def get_GlV_features(self, local_batch_size):
        GlV_features = self.word_embedding_matrix.repeat(local_batch_size, 1)
        GlV_features = denseTensor_to_SparseTensor(torch.Tensor(GlV_features)).to(self.device)
        return GlV_features

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class SanityDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, data_dir=DATA_DIR, triplet_file_name=TRIPLET_FILE_NAME, is_train=False, is_val=False, is_test=False, positional_feature_enabled=False):
        super(SanityDataset, self).__init__()
        self.data_dir = data_dir
        self.triplet_file_name = triplet_file_name
        if not is_train and not is_val and not is_test:
            print('Must specify a network state: \in [train val test]')
        self.network_state = [is_train, is_val, is_test]
        self.triplets = self.load_triplets()
        self.positional_feature_enabled = positional_feature_enabled

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

    def __getitem__(self, index):
        # Path to triplet data_points
        paths = self.triplets[index]

        # Load Triplet SGs data
        anchor_SG = np.load(paths[0] + '.npy', allow_pickle='TRUE').item()
        positive_SG = np.load(paths[1] + '.npy', allow_pickle='TRUE').item()
        negative_SG = np.load(paths[2] + '.npy', allow_pickle='TRUE').item()

        adj_on = (get_adj_matrix(anchor_SG['on']), get_adj_matrix(positive_SG['on']), get_adj_matrix(negative_SG['on']))
        adj_in = (get_adj_matrix(anchor_SG['in']), get_adj_matrix(positive_SG['in']), get_adj_matrix(negative_SG['in']))
        adj_proximity = (get_adj_matrix(anchor_SG['proximity']), get_adj_matrix(positive_SG['proximity']), get_adj_matrix(negative_SG['proximity']))

        if not self.positional_feature_enabled:
            return adj_on + adj_in + adj_proximity
        else:


            fractional_bboxs = (np.asarray(anchor_SG['fractional_bboxs'], dtype=np.float32),
                                np.asarray(positive_SG['fractional_bboxs'], dtype=np.float32),
                                np.asarray(negative_SG['fractional_bboxs'], dtype=np.float32))

            obj_occurence_vecs = (np.asarray(anchor_SG['vec'].todense(), dtype=np.float32),
                                  np.asarray(positive_SG['vec'].todense(), dtype=np.float32),
                                  np.asarray(negative_SG['vec'].todense(), dtype=np.float32))

            # self.show_data_points((Image.open(paths[0]+'.png'), Image.open(paths[1]+'.png'), Image.open(paths[2]+'.png')), adj_on, adj_in, adj_proximity, fractional_bboxs, paths)

            return adj_on + adj_in + adj_proximity + fractional_bboxs + obj_occurence_vecs

    def __len__(self):
        return len(self.triplets)
