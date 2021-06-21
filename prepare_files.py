from lib.update_objectType import *
from lib.object_dynamics import shuffle_scene_layout
from lib.similarity import *
from lib.params import *
from PIL import Image
from lib.scene_graph_generation import *
from ai2thor.controller import Controller
import ai2thor
import torchvision, torch, copy
import time

if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Prepare files store AI2THOR infomation
    # --------------------------------------------------------------------------
    update_object_type() # obj_2_idx_dic.npy, idx_2_obj_list.npy
    # --------------------------------------------------------------------------
    # Store Glove Vector
    # --------------------------------------------------------------------------
    load_glove() # glove_embedding.npy
    thor_2_glove() # THOR_2_GLOVE.npy
    thor_2_vec() # THOR_2_VEC.npy
