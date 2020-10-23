import sys
import torch
from os.path import dirname, abspath
root_folder = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_folder)
from lib.params import *

# Encoded vector [batch_size, IMAGE_ENCODING_VEC_LENGTH,1,1]
# cosine in dim = 1 which is idx of IMAGE_ENCODING_VEC_LENGTH
COS = torch.nn.CosineSimilarity(dim=1, eps=1e-10)

IMAGE_SIZE = 224 # Input image size into siamese image branch
DATA_DIR = 'image_data' # Training and validation data directory
PN_THRESHOLD = {'p': 0.6, 'n': 0.5} # triplet, anchor to positive and negative threshold
TRIPLET_MAX_FRACTION_TO_IMAGES = 0.5
TRIPLET_MAX_NUM_PER_ANCHOR = 5
NEGATIVE_RAND_NUM = 1 # randomly chooses NEGTIVE_RAND_NUM of negative for each anchor-positive pair
NUM_WORKERS = 4 # dataloader workers
IMAGE_ENCODING_VEC_LENGTH = 2048 # encoding vector length of the image

# Traning parameters/setting
BATCH_SIZE = 16 # larger than one
NUM_EPOCHS = 50

# Training hyper-parameter
LEARNING_RATE = 0.001
MOMENTUM = 0.9
# Decay LR by a factor of 0.1 every 7 epochs
STEP_SIZE = 7
GAMMA = 0.1