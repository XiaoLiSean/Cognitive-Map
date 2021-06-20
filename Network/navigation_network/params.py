'''
Navigation Network
For robot localization in a dynamic environment.
'''
import numpy as np
from lib.params import ADJACENT_NODES_SHIFT_GRID

ACTION_ENCODING = dict(left=np.array([1,0,0]), right=np.array([0,1,0]), forward=np.array([0,0,1]))
ACTION_CLASSNUM = len(ACTION_ENCODING) # dimension of action space [left, right, forward]
HORIZONTAL_MOVE_MAX = ADJACENT_NODES_SHIFT_GRID # maximum number of grid steps in horizontal movement
FORWARD_MOVE_MAX = 4 # maximum number of grid steps in forward movement
NUM_WORKERS = 0 # dataloader workers set to zero cuz triples shared files
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
# Training hyper-parameter
LEARNING_RATE = 0.01 # common settings for networks {image=0.01, SG=0.01}
MOMENTUM = 0.9 # common settings for networks {image=0.9}
# --------------------------------------------
# Decay LR by a factor of GAMMA (LR*=GAMMA) every STEP_SIZE epochs
STEP_SIZE = 7 # common settings for networks {image=10, SG=10}
GAMMA = 0.5 # common settings for networks {image=0.1, SG=0.01}
# ------------------------------------------------------------------------------
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# ------------------------------------------------------------------------------
DATA_DIR = './Network/datasets' # Training and validation data directory
TRAJECTORY_FILE_NAME = 'trajectories_fraction_0.001.npy' #  19043 pairs
CHECKPOINTS_DIR = './Network/navigation_network/checkpoints/'
# ------------------------------------------------------------------------------
