import sys
from os.path import dirname, abspath
root_folder = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_folder)
from lib.params import *

BATCH_SIZE = 1
IMAGE_SIZE = 224 # Input image size into siamese image branch
DATA_DIR = 'image_data' # Training and validation data directory
PN_THRESHOLD = 0.5 # triplet, positive to nective threshold
NEGATIVE_RAND_NUM = 10 # randomly chooses NEGTIVE_RAND_NUM of negative for each anchor-positive pair
NUM_WORKERS = 4 # dataloader workers
