'''
Retrieval Network, Written by Xiao
For robot localization in a dynamic environment.
'''
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
'''
NUM_EPOCHS, STEP_SIZE, GAMMA, lr0 satisfies
As lr_final_epoch = lr0*power(GAMMA, ceil(NUM_EPOCHS / STEP_SIZE)-1)
if we want lr_final_epoch <= 1e-4
this constraint will regulate the corresponding value of other params
'''
# Traning parameters/setting
BATCH_SIZE = dict(rnet=14, resnet50=26, vgg16=19, resnext50_32x4d=20, googlenet=400) # common settings for networks {image=26, SG=14}
NUM_EPOCHS = 60 # common settings for networks {image=50, SG=1000}
# --------------------------------------------
'''
Meanwhile, too large ALPHA_MARGIN will cause overfitting as it's functional
as a hard constraint on the similarity diff as long as the loss being positive
'''
# Training hyper-parameter
LEARNING_RATE = 0.01 # common settings for networks: adam {image=0.001, SG=0.001} ; SGD optimizer{image=0.01, SG=0.01}
MOMENTUM = 0.9 # common settings for networks {image=0.9}
ALPHA_MARGIN = 0.10 # margin used in triplet loss
DROPOUT_RATE = 0.2 # rate of dropout a certain neuron {20% = 0.2}
# --------------------------------------------
# Decay LR by a factor of GAMMA (LR*=GAMMA) every STEP_SIZE epochs
STEP_SIZE = 10 # common settings for networks {image=10, SG=10}
GAMMA = 0.7 # common settings for networks {image=0.1, SG=0.01}
# ------------------------------------------------------------------------------
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# ------------------------------------------------------------------------------
DATA_DIR = './Network/datasets' # Training and validation data directory
TRIPLET_FILE_NAME = 'triplets_APN_name_magnitude_0.2.npy' # 9.5k triples in total
PAIR_FILE_NAME = 'pairs_name_fraction_0.005.npy' # 93354 pairs
CHECKPOINTS_DIR = './Network/retrieval_network/checkpoints/'
# ------------------------------------------------------------------------------
