import numpy as np
import os, sys, random, scipy
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import find as find_sparse_idx
from torchvision import datasets, transforms
from Network.retrieval_network.params import IMAGE_SIZE
from PIL import Image
from os.path import dirname, abspath
from termcolor import colored
from copy import deepcopy
from lib.params import OBJ_TYPE_NUM
from lib.similarity import view_similarity
from lib.scene_graph_generation import Scene_Graph

# ------------------------------------------------------------------------------
# Get pose dict from file name
def  get_pose_from_name(name):
    info = name.split('_')
    pose = {'x': float(info[1]),
            'z': float(info[2]),
            'theta': float(info[3])}
    return pose

# ------------------------------------------------------------------------------
# This function is used to update triplets npy list in train and val folder
# filename = [FloorPlanX_x_z_theta_i_.png]
def update_triplet_info(DATA_DIR, PN_THRESHOLD, TRIPLET_MAX_FRACTION_TO_POINTS, TRIPLET_MAX_NUM_PER_ANCHOR):

    labels = ['train', 'val']

    for label in labels:
        for FloorPlan in os.listdir(DATA_DIR + '/' + label):
            data_points = []
            total_triplet_num = 0
            for filename in os.listdir(DATA_DIR + '/' + label + '/' + FloorPlan):
                if filename.endswith(".png"):
                    data_points.append(filename[0:-4])

            print(colored('Process A-P list: ','blue') + DATA_DIR + '/' + label + '/' + FloorPlan + '/')
            random.shuffle(data_points)
            total_triplet_max_num = len(data_points)*TRIPLET_MAX_FRACTION_TO_POINTS

            # dictionary map from one anchor to its positive data_points in same FloorPlan
            anchor_to_positives = {}
            anchor_to_negatives = {}

            for idx, anchor in enumerate(data_points):
                anchor_to_positives.update({anchor: []})
                anchor_to_negatives.update({anchor: []})

                pose_anchor = get_pose_from_name(anchor)
                # for one anchor image add positive pairs which are after the anchor in the list
                for i in range(idx+1, len(data_points)):
                    # Find positive pairs
                    pose_i = get_pose_from_name(data_points[i])
                    similarity = view_similarity(pose_anchor, pose_i, visualization_on=False)
                    # Append positive and negative data_points
                    if similarity >= PN_THRESHOLD['p'] and len(anchor_to_positives[anchor]) < TRIPLET_MAX_NUM_PER_ANCHOR:
                        anchor_to_positives[anchor].append((data_points[i], similarity))
                    elif similarity < PN_THRESHOLD['n'] and len(anchor_to_negatives[anchor]) < TRIPLET_MAX_NUM_PER_ANCHOR:
                        anchor_to_negatives[anchor].append((data_points[i], similarity))
                    # break loop for current anchor if TRIPLET_MAX_NUM_PER_ANCHOR is reached
                    new_added_triplets = min([len(anchor_to_positives[anchor]), len(anchor_to_negatives[anchor])])
                    if new_added_triplets >= TRIPLET_MAX_NUM_PER_ANCHOR:
                        break
                # Remove anchor image if it has no positive pairs
                if len(anchor_to_positives[anchor]) == 0 or len(anchor_to_negatives[anchor]) == 0:
                    anchor_to_positives.pop(anchor)
                    continue
                # update total triplet numbers
                total_triplet_num += min([len(anchor_to_positives[anchor]), len(anchor_to_negatives[anchor])])

                if total_triplet_num >= total_triplet_max_num:
                    break

            np.save(DATA_DIR + '/' + label + '/' + FloorPlan + '/' + 'anchor_to_positives.npy', anchor_to_positives) # Save dict as .npy
            np.save(DATA_DIR + '/' + label + '/' + FloorPlan + '/' + 'anchor_to_negatives.npy', anchor_to_negatives) # Save dict as .npy
            np.save(DATA_DIR + '/' + label + '/' + FloorPlan + '/' + 'name_list.npy', data_points) # Save list as .npy
            print(colored('Done A-P list: ','blue') + str(total_triplet_num) + 'pairs, ' + str(len(anchor_to_positives)) + 'anchors')


# ------------------------------------------------------------------------------
class TripletImagesDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, DATA_DIR, image_size=IMAGE_SIZE, is_train=True):
        super(TripletImagesDataset, self).__init__()
        self.data_dir = DATA_DIR
        self.image_size = image_size
        self.is_train = is_train
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.triplets_img, self.triplets_alphas = self.get_triplets()

    def get_triplets(self):
        triplets_img = []
        triplets_alphas = []
        if self.is_train:
            path = self.data_dir + '/' + 'train'
        else:
            path = self.data_dir + '/' + 'val'

        # Iterate through floorplans
        for FloorPlan in os.listdir(path):
            anchor_to_positives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_positives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-positives
            anchor_to_negatives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_negatives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-negatives
            for anchor in anchor_to_positives:
                anchor_img = path + '/' + FloorPlan + '/' + anchor + '.png'
                positives = anchor_to_positives[anchor]
                negatives = anchor_to_negatives[anchor]
                for i in range(min([len(positives), len(negatives)])):
                    positive_img = path + '/' + FloorPlan + '/' + positives[i][0] + '.png'
                    negative_img = path + '/' + FloorPlan + '/' + negatives[i][0] + '.png'
                    # append path to desired image triplets
                    triplets_img.append((deepcopy(anchor_img), deepcopy(positive_img), deepcopy(negative_img)))
                    triplets_alphas.append((deepcopy(positives[i][1]), deepcopy(negatives[i][1])))

        return triplets_img, triplets_alphas

    def __getitem__(self, index):
        # Path to triplet data_points
        paths = self.triplets_img[index]
        triplet = (self.transforms(Image.open(paths[0])), self.transforms(Image.open(paths[1])), self.transforms(Image.open(paths[2])))
        return triplet, self.triplets_alphas[index]

    def __len__(self):
        return len(self.triplets_img)

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
# Siamese dataset loader
# ------------------------------------------------------------------------------
class SiameseDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) all positive and negative samples are chosed as (A-N) and (A-P) siamese pairs
    """

    def __init__(self, DATA_DIR, image_size=IMAGE_SIZE, is_train=True):
        super(SiameseDataset, self).__init__()
        self.data_dir = DATA_DIR
        self.is_train = is_train
        self.image_size = image_size
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.siameses, self.alphas = self.get_siameses()

    def get_siameses(self):
        siameses_data = []
        alphas = []
        if self.is_train:
            path = self.data_dir + '/' + 'train'
        else:
            path = self.data_dir + '/' + 'val'

        # Iterate through floorplans
        for FloorPlan in os.listdir(path):
            anchor_to_positives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_positives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-positives
            anchor_to_negatives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_negatives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-negatives
            for anchor in anchor_to_positives:
                anchor_data = path + '/' + FloorPlan + '/' + anchor
                # Skip empty anchor scene graph which is trival to gcn training
                if self.sg_is_empty(anchor_data + '.npy'):
                    continue
                positives = anchor_to_positives[anchor]
                negatives = anchor_to_negatives[anchor]
                for i in range(min([len(positives), len(negatives)])):
                    positive_data = path + '/' + FloorPlan + '/' + positives[i][0]
                    negative_data = path + '/' + FloorPlan + '/' + negatives[i][0]
                    # Skip empty positive scene graph which is trival to gcn training
                    if self.sg_is_empty(positive_data + '.npy'):
                        continue
                    # append path to desired siamese data points
                    siameses_data.append((deepcopy(anchor_data), deepcopy(positive_data)))
                    alphas.append(deepcopy(positives[i][1]))
                    siameses_data.append((deepcopy(anchor_data), deepcopy(negative_data)))
                    alphas.append(deepcopy(negatives[i][1]))

        return siameses_data, alphas

    def sg_is_empty(self, sg_path):
        is_empty = True
        SG = np.load(sg_path, allow_pickle='TRUE').item()
        for A in [SG['on'], SG['in'], SG['proximity']]:
            idx = find_sparse_idx(A)
            if len(idx[0]) != 0:
                is_empty = False

        return is_empty

    def show_data_points(self, R_on, R_in, R_prox):
        SG = Scene_Graph(R_on=R_on, R_in=R_in, R_proximity=R_prox)
        SG.visualize_SG()


    def __getitem__(self, index):
        # Path to siamese data_points
        paths = self.siameses[index]

        # Load siamese Images data
        siamese_imgs = (self.transforms(Image.open(paths[0]+'.png')),
                        self.transforms(Image.open(paths[1]+'.png')))

        # Load siamese SGs data
        anchor_SG = np.load(paths[0] + '.npy', allow_pickle='TRUE').item()
        pn_SG = np.load(paths[1] + '.npy', allow_pickle='TRUE').item()

        adj_on = (get_adj_matrix(anchor_SG['on']), get_adj_matrix(pn_SG['on']))
        adj_in = (get_adj_matrix(anchor_SG['in']), get_adj_matrix(pn_SG['in']))
        adj_proximity = (get_adj_matrix(anchor_SG['proximity']), get_adj_matrix(pn_SG['proximity']))

        return siamese_imgs + adj_on + adj_in + adj_proximity, self.alphas[index]

    def __len__(self):
        return len(self.siameses)

# ------------------------------------------------------------------------------
# Triplet dataset loader
# ------------------------------------------------------------------------------
class TripletDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, DATA_DIR, image_size=IMAGE_SIZE, is_train=True):
        super(TripletDataset, self).__init__()
        self.data_dir = DATA_DIR
        self.is_train = is_train
        self.image_size = image_size
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.triplets, self.triplets_alphas = self.get_triplets()

    def get_triplets(self):
        triplets_data = []
        triplets_alphas = []
        if self.is_train:
            path = self.data_dir + '/' + 'train'
        else:
            path = self.data_dir + '/' + 'val'

        # Iterate through floorplans
        for FloorPlan in os.listdir(path):
            anchor_to_positives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_positives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-positives
            anchor_to_negatives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_negatives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-negatives
            for anchor in anchor_to_positives:
                anchor_data = path + '/' + FloorPlan + '/' + anchor
                # Skip empty anchor scene graph which is trival to gcn training
                if self.sg_is_empty(anchor_data + '.npy'):
                    continue
                positives = anchor_to_positives[anchor]
                negatives = anchor_to_negatives[anchor]
                for i in range(min([len(positives), len(negatives)])):
                    positive_data = path + '/' + FloorPlan + '/' + positives[i][0]
                    negative_data = path + '/' + FloorPlan + '/' + negatives[i][0]
                    # Skip empty positive scene graph which is trival to gcn training
                    if self.sg_is_empty(positive_data + '.npy'):
                        continue
                    # append path to desired triplets
                    triplets_data.append((deepcopy(anchor_data), deepcopy(positive_data), deepcopy(negative_data)))
                    triplets_alphas.append((deepcopy(positives[i][1]), deepcopy(negatives[i][1])))

        return triplets_data, triplets_alphas

    def sg_is_empty(self, sg_path):
        is_empty = True
        SG = np.load(sg_path, allow_pickle='TRUE').item()
        for A in [SG['on'], SG['in'], SG['proximity']]:
            idx = find_sparse_idx(A)
            if len(idx[0]) != 0:
                is_empty = False

        return is_empty

    def show_data_points(self, R_on, R_in, R_prox):
        SG = Scene_Graph(R_on=R_on, R_in=R_in, R_proximity=R_prox)
        SG.visualize_SG()


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

        return triplet_imgs + adj_on + adj_in + adj_proximity, self.triplets_alphas[index]

    def __len__(self):
        return len(self.triplets)
