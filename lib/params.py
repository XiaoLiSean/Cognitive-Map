# Module for setup params and global variables
from os.path import dirname, abspath
import numpy as np

SIM_WINDOW_HEIGHT = 700
SIM_WINDOW_WIDTH = 900
BAN_TYPE_LIST = ['Floor']   # Ignore non-informative objectType e.g. 'Floor'
INFO_FILE_PATH = dirname(dirname(abspath(__file__))) + '/AI2THOR_info' # File path for info of iTHOR Env.
obj_2_idx_dic = np.load(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', allow_pickle='TRUE').item()
idx_2_obj_list = np.load(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy')
OBJ_TYPE_NUM = len(idx_2_obj_list) # Maximum numbers of objectType in iTHOR Env.
PROXIMITY_THRESHOLD = 3 # distance ratio threshold for proximity determination
VISBILITY_DISTANCE = 1.5 # default 1.5 meter, object within the radius of a cylinder centered about the y-axis of the agent is visible
FIELD_OF_VIEW = 120 # default 90 degree, 120 degree is binocular FoV
SIMILARITY_GRID_ORDER = 2 # Approx Grid Size of 10^SIMILARITY_GRID_ORDER for similarity score between views
