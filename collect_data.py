import argparse
import numpy as np
import random
from copy import deepcopy
from scipy.sparse import lil_matrix
from termcolor import colored
from itertools import combinations
from lib.robot_env import Agent_Sim
from lib.simulation_agent import Robot
from lib.params import VISBILITY_DISTANCE, SCENE_TYPES, SCENE_NUM_PER_TYPE
from Network.retrieval_network.params import TRAIN_FRACTION, VAL_FRACTION, DATA_DIR

# Generate a tuple list of non empty index [(i,j), (j,k) ...]
def get_index_as_tuple_list(array_map):
    tuple_list = []
    # Filter out array entries with only one data
    idx_lists = array_map.nonzero()
    for i in range(len(idx_lists[0])):
        if array_map[idx_lists[0][i],idx_lists[1][i]] > 1:
            tuple_list.append((idx_lists[0][i],idx_lists[1][i]))
    return tuple_list

def manhattan_distance(x, y):
    return np.abs(x[0]-y[0]) + np.abs(x[1]-y[1])

#  from tuple_list
def get_first_tier_AP(idx_list, array_map, key):
    first_tier_AP = []
    for anchor in idx_list:
        num_A = array_map[anchor[0], anchor[1]]
        list_A = [*range(1, num_A + 1)]
        for pair in list(combinations(list_A, 2)):
            first_tier_AP.append(((anchor[0], anchor[1], pair[0], key), (anchor[0], anchor[1], pair[1], key)))
    return first_tier_AP


# randomly generate point from unit ball in L1 given radius
def get_points_from_rect(r):
    rect_tuple_list = []

    if r == 0:
        return [(0,0)]

    if r > 1:
        for i in range(1, r):
            rect_tuple_list.append((i, r-i))
            rect_tuple_list.append((-i, r-i))
            rect_tuple_list.append((i, -r+i))
            rect_tuple_list.append((-i, -r+i))

    rect_tuple_list.append((0, r))
    rect_tuple_list.append((0, -r))
    rect_tuple_list.append((r, 0))
    rect_tuple_list.append((-r, 0))

    return rect_tuple_list

# get negative data directly from array_map give manhattan_distance as constraint
def add_negative_data(ith_tier_AP, array_map, key, distanceToAnchor=1, fixed_Len=False):
    ith_tier_APN = []
    for AP in ith_tier_AP:
        i = AP[0][0]
        j = AP[0][1]
        this_AP_have_negative = False
        for d_ij in get_points_from_rect(distanceToAnchor):
            i_negative = i + d_ij[0]
            j_negative = j + d_ij[1]
            if i_negative >= array_map.shape[0] or j_negative >= array_map.shape[1] or i_negative < 0 or j_negative < 0:
                continue
            if array_map[i_negative, j_negative] != 0:
                for negative_idx in range(1, array_map[i_negative, j_negative]+1):
                    negative = (i_negative, j_negative, negative_idx, key)
                    triplet = (*AP, negative)
                    ith_tier_APN.append(deepcopy(triplet))
                    this_AP_have_negative = True
                    if fixed_Len:
                        break
            if fixed_Len and this_AP_have_negative:
                break

    return ith_tier_APN

def add_negative_data_from_other_angle(fifrth_tier_AP, current_key, array_maps):
    final_APN = []
    for AP in fifrth_tier_AP:
        for key in array_maps:
            if key == current_key:
                continue
            while True:
                negative_i = random.randint(0, array_maps[key].shape[0]-1)
                negative_j = random.randint(0, array_maps[key].shape[1]-1)
                if array_maps[key][negative_i, negative_j] >= 1:
                    break
            negative = (negative_i, negative_j, random.randint(1, array_maps[key][negative_i, negative_j]), key)
            triplet = (*AP, negative)
            final_APN.append(deepcopy(triplet))
            break
    return final_APN

def get_AP_from_previous_tier_AN(prev_tier_APN):
    next_tier_AP = []
    for APN in prev_tier_APN:
        next_tier_AP.append((APN[0], APN[2]))

    return next_tier_AP

def cutoff_dataset(ith_tier_APN, dataset_size, fraction):
    cutoff_idx = min([len(ith_tier_APN), round(dataset_size*fraction)])
    random.shuffle(ith_tier_APN)
    ith_tier_APN = ith_tier_APN[0:cutoff_idx]
    return ith_tier_APN

def generate_triplets(file_path):
    triplets_APN_idx = []
    triplets_APN_name = []
    array_maps = np.load(file_path + '/' + 'array_maps.npy', allow_pickle='TRUE').item()
    for key in array_maps:
        # get array-like map for a specific robot heading [0.0, 90.0, 180.0, 270.0]
        array_map = array_maps[key]
        idx_list = get_index_as_tuple_list(array_map)
        #-----------------------------------------------------------------------
        '''Get 1'st tier data of (A,P,N) (35% of the entire dataset)'''
        # (A,P,N) A = (i,j,idx) where idx \in array_map[i,j]
        #-----------------------------------------------------------------------
        first_tier_AP = get_first_tier_AP(idx_list, array_map, key)
        first_tier_APN = add_negative_data(first_tier_AP, array_map, key, distanceToAnchor=1)
        dataset_size = round(len(first_tier_APN) / 0.50)
        #-----------------------------------------------------------------------
        '''Get 2'nd tier data of (A,P,N) (20% of the entire dataset)'''
        #-----------------------------------------------------------------------
        second_tier_AP = get_AP_from_previous_tier_AN(first_tier_APN)
        second_tier_APN = add_negative_data(second_tier_AP, array_map, key, distanceToAnchor=2)
        second_tier_APN = cutoff_dataset(second_tier_APN, dataset_size, 0.20)
        #-----------------------------------------------------------------------
        '''Get 3'rd tier data of (A,P,N) (20% of the entire dataset)'''
        #-----------------------------------------------------------------------
        third_tier_AP = get_AP_from_previous_tier_AN(second_tier_APN)
        third_tier_APN = add_negative_data(third_tier_AP, array_map, key, distanceToAnchor=3)
        third_tier_APN = cutoff_dataset(third_tier_APN, dataset_size, 0.15)
        #-----------------------------------------------------------------------
        '''Get 4'th tier data of (A,P,N) (20% of the entire dataset)'''
        #-----------------------------------------------------------------------
        fourth_tier_AP = get_AP_from_previous_tier_AN(third_tier_APN)
        fourth_tier_APN = add_negative_data(fourth_tier_AP, array_map, key, distanceToAnchor=4)
        fourth_tier_APN = cutoff_dataset(fourth_tier_APN, dataset_size, 0.10)
        #-----------------------------------------------------------------------
        '''Get 5'th tier data of (A,P,N) (5% of the entire dataset)'''
        #-----------------------------------------------------------------------
        cutoff_idx = min([len(fourth_tier_APN), round(dataset_size*0.05)])
        last_APN_tmp = fourth_tier_APN[0:cutoff_idx]
        fifth_tier_AP = get_AP_from_previous_tier_AN(last_APN_tmp)
        fifth_tier_APN = add_negative_data_from_other_angle(fifth_tier_AP, key, array_maps)
        #-----------------------------------------------------------------------
        original_len = len(triplets_APN_idx)
        triplets_APN_idx.extend(first_tier_APN)
        triplets_APN_idx.extend(second_tier_APN)
        triplets_APN_idx.extend(third_tier_APN)
        triplets_APN_idx.extend(fourth_tier_APN)
        triplets_APN_idx.extend(fifth_tier_APN)
        print('Finished Generate {} Triplets for {} degree: '.format(len(triplets_APN_idx)-original_len, key) +
              '({0:.0%}, '.format(len(first_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%}, '.format(len(second_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%}, '.format(len(third_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%}, '.format(len(fourth_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%})'.format(len(fifth_tier_APN)/(len(triplets_APN_idx)-original_len)))
    #---------------------------------------------------------------------------
    for APN in triplets_APN_idx:
        anchor = str(APN[0][3]) + '_' + str(APN[0][0]) + '_' + str(APN[0][1]) + '_' + str(APN[0][2])
        positive = str(APN[1][3]) + '_' + str(APN[1][0]) + '_' + str(APN[1][1]) + '_' + str(APN[1][2])
        negative = str(APN[2][3]) + '_' + str(APN[2][0]) + '_' + str(APN[2][1]) + '_' + str(APN[2][2])
        triplets_APN_name.append(deepcopy((anchor, positive, negative)))

    np.save(file_path + '/' + 'triplets_APN_name.npy', triplets_APN_name)

    return len(triplets_APN_name)


# ------------------------------------------------------------------------------
# collect train, validation and test data for network image branch
def data_collection(is_train=True, is_test=True):
    # Initialize robot
    robot = Robot()
    # Iterate through all the scenes to collect data and separate them into train, val and test sets
    if is_train:
        for scene_type in SCENE_TYPES:
            for scene_num in range(1, int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1):
                if scene_num <= int(SCENE_NUM_PER_TYPE*TRAIN_FRACTION):
                    FILE_PATH = DATA_DIR + '/train'
                elif scene_num <= int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)):
                    FILE_PATH = DATA_DIR + '/val'

                robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
                robot.coordnates_patroling(saving_data=True, file_path=FILE_PATH, dynamics_rounds=5, pertubation_round=2)
                datanum = generate_triplets(FILE_PATH + '/' + robot._scene_name)
                print('{}: {} triplets'.format(robot._scene_name, datanum))

    if is_test:
        FILE_PATH = DATA_DIR + '/test'
        for scene_type in SCENE_TYPES:
            for scene_num in range(int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1, SCENE_NUM_PER_TYPE + 1):
                robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
                robot.coordnates_patroling(saving_data=True, file_path=FILE_PATH, dynamics_rounds=5, pertubation_round=2)
                datanum = generate_triplets(FILE_PATH + '/' + robot._scene_name)
                print('{}: {} triplets'.format(robot._scene_name, datanum))

# ------------------------------------------------------------------------------
# manually update topological map info
def iter_test_scene(is_xiao=False, is_yidong=False):
    # Initialize robot
    robot = Agent_Sim()
    test_idx_initial = int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1
    test_idx_end = SCENE_NUM_PER_TYPE + 1
    add_on = [2, 3, 2, 3]
    for idx, scene_type in enumerate(SCENE_TYPES):
        if is_xiao:
            test_idx_end = test_idx_initial + add_on[idx]
        if is_yidong:
            test_idx_initial = test_idx_end - ( 5 - add_on[idx])

        for scene_num in range(test_idx_initial, test_idx_end):
            robot.reset_scene(scene_type=scene_type, scene_num=scene_num, ToggleMapView=True, Show_doorway=False, shore_toggle_map=False)
            robot.show_map(show_nodes=False, show_edges=False)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", help="manually update topological map info", action="store_true")
    parser.add_argument("--train", help="collect train and validation data for network image branch", action="store_true")
    parser.add_argument("--test", help="collect test data for network image branch", action="store_true")
    parser.add_argument("--yidong", help="manually collect topological map node for yidong", action="store_true")
    parser.add_argument("--xiao", help="manually collect topological map node for xiao", action="store_true")
    args = parser.parse_args()

    generate_triplets(DATA_DIR + '/train' + '/FloorPlan1')

    # --------------------------------------------------------------------------
    # Used to collect data
    if args.train or args.test:
        data_collection(is_train=args.train, is_test=args.test)

    # Used to see visualization of each test scene and record the topological node manually
    if args.topo:
        iter_test_scene(is_xiao=args.xiao, is_yidong=args.yidong)
