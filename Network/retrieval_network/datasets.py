import numpy as np
import os
import sys
import torch
from torchvision import datasets, transforms
from PIL import Image
from os.path import dirname, abspath
from copy import deepcopy

# import from root/lib
root_folder = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_folder)

from lib.similarity import *

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
def update_triplet_info(DATA_DIR, PN_THRESHOLD):

    labels = ['train', 'val']

    for label in labels:
        images = []
        for filename in os.listdir(DATA_DIR + '/' + label):
            if filename.endswith(".png"):
                images.append(filename)

        # dictionary map from one anchor to its positive images in same FloorPlan
        anchor_to_positives = {}
        for idx, anchor in enumerate(images):
            positive_num = 0
            anchor_to_positives.update({anchor: []})
            pose_anchor = get_pose_from_name(anchor)
            # for one anchor image add positive pairs which are after the anchor in the list
            for i in range(idx+1, len(images)):
                pose_i = get_pose_from_name(images[i])
                similarity = view_similarity(pose_anchor, pose_i)
                if similarity > PN_THRESHOLD:
                    positive_num += 1
                    anchor_to_positives[anchor].append(images[i])
            # Remove anchor image if it has no positive pairs
            if positive_num == 0:
                anchor_to_positives.pop(anchor)

        np.save(DATA_DIR + '/' + label + '/' + 'anchor_to_positives.npy', anchor_to_positives) # Save dict as .npy
        np.save(DATA_DIR + '/' + label + '/' 'name_list.npy', images) # Save list as .npy

# ------------------------------------------------------------------------------
class TripletImagesDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, DATA_DIR, IMAGE_SIZE, NEGATIVE_RAND_NUM, is_train=True):
        super(TripletImagesDataset, self).__init__()
        self.data_dir = DATA_DIR
        self.image_size = IMAGE_SIZE
        self.is_train = is_train
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.triplets_img, self.triplets_pose = self.get_triplets(NEGATIVE_RAND_NUM)

    def get_triplets(self, NEGATIVE_RAND_NUM):
        triplets_img = []
        triplets_name = []
        if self.is_train:
            path = self.data_dir + '/' + 'train'
        else:
            path = self.data_dir + '/' + 'val'

        anchor_to_positives = np.load(path + '/' + 'anchor_to_positives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-positives
        name_list = np.load(path + '/' + 'name_list.npy')
        for anchor in anchor_to_positives:
            anchor_img = self.transforms(Image.open(path + '/' + anchor))
            for positive in anchor_to_positives[anchor]:
                positive_img = self.transforms(Image.open(path + '/' + positive))
                for i in range(NEGATIVE_RAND_NUM):
                    while True:
                        idx = np.random.randint(0, len(name_list))
                        if name_list[idx] == anchor or name_list[idx] in anchor_to_positives[anchor]:
                            continue
                        else:
                            negative = name_list[idx]
                            break
                    negative_img = self.transforms(Image.open(path + '/' + negative))
                    # Append triplet data tuple
                    triplets_img.append((deepcopy(anchor_img), deepcopy(positive_img), deepcopy(negative_img)))
                    triplets_name.append([anchor, positive, negative])

        return triplets_img, triplets_name

    def __getitem__(self, index):
        return self.triplets_img[index], self.triplets_pose[index]

    def __len__(self):
        return len(self.triplets_img)
