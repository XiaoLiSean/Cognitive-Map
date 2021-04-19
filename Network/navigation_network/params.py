'''
Navigation Network, Written by Xiao
For robot localization in a dynamic environment.
'''

ACTION_CLASSNUM = 3 # dimension of action space [left, right, forward]
# ------------------------------------------------------------------------------
'''
NUM_EPOCHS, STEP_SIZE, GAMMA, lr0 satisfies
As lr_final_epoch = lr0*power(GAMMA, ceil(NUM_EPOCHS / STEP_SIZE)-1)
if we want lr_final_epoch <= 1e-4
this constraint will regulate the corresponding value of other params
'''
# Traning parameters/setting
BATCH_SIZE = 14 # common settings for networks {image=26, SG=14}
NUM_EPOCHS = 60 # common settings for networks {image=50, SG=1000}
# --------------------------------------------
'''
Meanwhile, too large ALPHA_MARGIN will cause overfitting as it's functional
as a hard constraint on the similarity diff as long as the loss being positive
'''
# Training hyper-parameter
LEARNING_RATE = 0.01 # common settings for networks {image=0.01, SG=0.01}
MOMENTUM = 0.9 # common settings for networks {image=0.9}
# --------------------------------------------
# Decay LR by a factor of GAMMA (LR*=GAMMA) every STEP_SIZE epochs
STEP_SIZE = 10 # common settings for networks {image=10, SG=10}
GAMMA = 0.7 # common settings for networks {image=0.1, SG=0.01}
# ------------------------------------------------------------------------------
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# ------------------------------------------------------------------------------
DATA_DIR = './Network/navigation_network/datasets' # Training and validation data directory
PAIR_FILE_NAME = 'pairs_name_fraction_0.005.npy' # 93354 pairs
CHECKPOINTS_DIR = './Network/navigation_network/checkpoints/'
# ------------------------------------------------------------------------------
