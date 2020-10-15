import numpy as np
import os, sys, random
import torch
from torchvision import datasets, transforms
from PIL import Image
from os.path import dirname, abspath
from termcolor import colored
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
def update_triplet_info(DATA_DIR, PN_THRESHOLD, TRIPLET_MAX_FRACTION_TO_IMAGES, TRIPLET_MAX_NUM_PER_ANCHOR):

    labels = ['train', 'val', 'test']

    for label in labels:
        for FloorPlan in os.listdir(DATA_DIR + '/' + label):
            images = []
            total_triplet_num = 0
            for filename in os.listdir(DATA_DIR + '/' + label + '/' + FloorPlan):
                if filename.endswith(".png"):
                    images.append(filename)

            print(colored('Process A-P list: ','blue') + DATA_DIR + '/' + label + '/' + FloorPlan + '/')
            random.shuffle(images)
            total_triplet_max_num = len(images)*TRIPLET_MAX_FRACTION_TO_IMAGES

            # dictionary map from one anchor to its positive images in same FloorPlan
            anchor_to_positives = {}
            anchor_to_negatives = {}

            for idx, anchor in enumerate(images):
                anchor_to_positives.update({anchor: []})
                anchor_to_negatives.update({anchor: []})

                pose_anchor = get_pose_from_name(anchor)
                # for one anchor image add positive pairs which are after the anchor in the list
                for i in range(idx+1, len(images)):
                    # Find positive pairs
                    pose_i = get_pose_from_name(images[i])
                    similarity = view_similarity(pose_anchor, pose_i, visualization_on=False)
                    # Append positive and negative images
                    if similarity > PN_THRESHOLD['p'] and len(anchor_to_positives[anchor]) < TRIPLET_MAX_NUM_PER_ANCHOR:
                        anchor_to_positives[anchor].append(images[i])
                    elif similarity < PN_THRESHOLD['n'] and len(anchor_to_negatives[anchor]) < TRIPLET_MAX_NUM_PER_ANCHOR:
                        anchor_to_negatives[anchor].append(images[i])
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
            np.save(DATA_DIR + '/' + label + '/' + FloorPlan + '/' + 'name_list.npy', images) # Save list as .npy
            print(colored('Done A-P list: ','blue') + str(total_triplet_num) + 'pairs')


# ------------------------------------------------------------------------------
class TripletImagesDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, DATA_DIR, IMAGE_SIZE, is_train=True):
        super(TripletImagesDataset, self).__init__()
        self.data_dir = DATA_DIR
        self.image_size = IMAGE_SIZE
        self.is_train = is_train
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.triplets_img, self.triplets_pose = self.get_triplets()

    def get_triplets(self):
        triplets_img = []
        triplets_name = []
        if self.is_train:
            path = self.data_dir + '/' + 'train'
        else:
            path = self.data_dir + '/' + 'val'

        # Iterate through floorplans
        for FloorPlan in os.listdir(path):
            anchor_to_positives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_positives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-positives
            anchor_to_negatives = np.load(path + '/' + FloorPlan + '/' + 'anchor_to_negatives.npy', allow_pickle='TRUE').item() # load npy dict of anchor-negatives
            name_list = np.load(path + '/' + FloorPlan + '/' + 'name_list.npy')
            for anchor in anchor_to_positives:
                anchor_img = self.transforms(Image.open(path + '/' + FloorPlan + '/' + anchor))
                positives = anchor_to_positives[anchor]
                negatives = anchor_to_negatives[anchor]
                for i in range(min([len(positives), len(negatives)])):
                    positive_img = self.transforms(Image.open(path + '/' + FloorPlan + '/' + positives[i]))
                    negative_img = self.transforms(Image.open(path + '/' + FloorPlan + '/' + negatives[i]))
                    triplets_img.append((deepcopy(anchor_img), deepcopy(positive_img), deepcopy(negative_img)))
                    triplets_name.append([anchor, positives[i], negatives[i]])
            break

        return triplets_img, triplets_name

    def __getitem__(self, index):
        return self.triplets_img[index], self.triplets_pose[index]

    def __len__(self):
        return len(self.triplets_img)
