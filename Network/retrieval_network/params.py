import sys
import torch

# ------------------------------------------------------------------------------
IMAGE_SIZE = 224 # Input image size into siamese image branch
NUM_WORKERS = 0 # dataloader workers set to zero cuz triples shared files
'''Single object embedding'''
BBOX_EMDEDDING_VEC_LENGTH = 64*4 # (u1,v1)+(u2.v2) boundng box embeding
ROI_EMDEDDING_VEC_LENGTH = 2048 # ROI align feature embedding
WORD_EMDEDDING_VEC_LENGTH = 300 # Glove word embedding
OBJ_FEATURE_VEC_LENGTH = 512 # Single object embedding as input to SG branch
'''Embedding output of final layer of SG or/and Img Branch'''
GCN_TIER = [OBJ_FEATURE_VEC_LENGTH, 256, 64, 8]
IMAGE_ENCODING_VEC_LENGTH = 2048 # encoding vector length of the image
SG_ENCODING_VEC_LENGTH = 2048 # encoding vector length of the sg
SCENE_ENCODING_VEC_LENGTH = 1024 # encoding vector length of current scene (image + sg)
# ------------------------------------------------------------------------------
# Traning parameters/setting
BATCH_SIZE = 10 # common settings for networks {image=16, SG=120} 26
NUM_EPOCHS = 60 # common settings for networks {image=50, SG=1000}
# --------------------------------------------
# Training hyper-parameter
LEARNING_RATE = 0.001 # common settings for networks {image=0.001, SG=0.01}
MOMENTUM = 0.9 # common settings for networks {image=0.9, SG=5e-4}
ALPHA_MARGIN = 0.15 # margin used in triplet loss
# --------------------------------------------
# Decay LR by a factor of GAMMA every STEP_SIZE epochs
STEP_SIZE = 7 # common settings for networks {image=7, SG=10}
GAMMA = 0.1 # common settings for networks {image=0.1, SG=0.01}
# ------------------------------------------------------------------------------
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# ------------------------------------------------------------------------------
DATA_DIR = './Network/retrieval_network/datasets-debug' # Training and validation data directory
TRIPLET_FILE_NAME = 'triplets_APN_name_magnitude_5.npy'
CHECKPOINTS_DIR = './Network/retrieval_network/checkpoints/'
# ------------------------------------------------------------------------------
