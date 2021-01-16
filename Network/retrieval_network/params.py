import sys
import torch
from lib.params import *

# ------------------------------------------------------------------------------
# Encoded vector [batch_size, IMAGE_ENCODING_VEC_LENGTH,1,1]
# cosine in dim = 1 which is idx of IMAGE_ENCODING_VEC_LENGTH

# COS = torch.nn.CosineSimilarity(dim=1, eps=1e-10)

# ------------------------------------------------------------------------------
IMAGE_SIZE = 224 # Input image size into siamese image branch
PN_THRESHOLD = {'p': 0.6, 'n': 0.5} # triplet, anchor to positive and negative threshold
TRIPLET_MAX_FRACTION_TO_IMAGES = 0.2
TRIPLET_MAX_NUM_PER_ANCHOR = 40
NEGATIVE_RAND_NUM = 1 # randomly chooses NEGTIVE_RAND_NUM of negative for each anchor-positive pair
NUM_WORKERS = 4 # dataloader workers
IMAGE_ENCODING_VEC_LENGTH = 2048 # encoding vector length of the image
SG_ENCODING_VEC_LENGTH = 256 # encoding vector length of the sg
SCENE_ENCODING_VEC_LENGTH = 1024 # encoding vector length of current scene (image + sg)
# ------------------------------------------------------------------------------
# Traning parameters/setting
BATCH_SIZE = 39 # common settings for networks {image=16, SG=120} 26
NUM_EPOCHS = 60 # common settings for networks {image=50, SG=1000}
# --------------------------------------------
# Training hyper-parameter
LEARNING_RATE = 0.001 # common settings for networks {image=0.001, SG=0.01}
MOMENTUM = 0.9 # common settings for networks {image=0.9, SG=5e-4}
# --------------------------------------------
# Decay LR by a factor of GAMMA every STEP_SIZE epochs
STEP_SIZE = 7 # common settings for networks {image=7, SG=10}
GAMMA = 0.1 # common settings for networks {image=0.1, SG=0.01}
# ------------------------------------------------------------------------------
DYNAMICS_ROUNDS = 10
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
LOCALIZATION_GRID_TOL = 1 # Tolerance for correct localization with in one gridstep distance
DATA_DIR = './Network/retrieval_network/datasets' # Training and validation data directory
CHECKPOINTS_DIR = './Network/retrieval_network/checkpoints/'
