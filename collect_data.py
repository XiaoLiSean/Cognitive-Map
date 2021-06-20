import argparse
import numpy as np
import random, math, os
from copy import deepcopy
from scipy.sparse import lil_matrix
from termcolor import colored
from itertools import combinations
from lib.robot_env import Agent_Sim
from lib.simulation_agent import Robot
from lib.params import VISBILITY_DISTANCE, SCENE_TYPES, SCENE_NUM_PER_TYPE
from Network.retrieval_network.params import TRAIN_FRACTION, VAL_FRACTION, DATA_DIR
from Network.navigation_network.params import ACTION_ENCODING, HORIZONTAL_MOVE_MAX, FORWARD_MOVE_MAX

# Generate a tuple list of non empty index [(i,j), (j,k) ...]
def get_index_as_tuple_list(array_map):
    tuple_list = []
    # Filter out array entries with only one data
    idx_lists = array_map.nonzero()
    reachable_points_num = len(idx_lists[0])
    for i in range(reachable_points_num):
        if array_map[idx_lists[0][i],idx_lists[1][i]] > 1:
            tuple_list.append((idx_lists[0][i],idx_lists[1][i]))
    return tuple_list, reachable_points_num

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
        next_tier_AP.append(deepcopy((APN[0], APN[2])))

    return next_tier_AP

def cutoff_dataset(ith_tier_APN, dataset_size, fraction):
    cutoff_idx = min([len(ith_tier_APN), round(dataset_size*fraction)])
    random.shuffle(ith_tier_APN)
    ith_tier_APN = deepcopy(ith_tier_APN[0:cutoff_idx])
    return ith_tier_APN

def generate_triplets(file_path, magnitude=None):
    if magnitude == None:
        file_name = 'triplets_APN_name.npy'
    else:
        file_name = 'triplets_APN_name_magnitude_' + str(magnitude) + '.npy'

    # if os.path.exists(file_path + '/' + file_name):
    #     return

    triplets_APN_idx = []
    triplets_APN_name = []
    running_total = np.array([0.0]*5, dtype=object)
    array_maps = np.load(file_path + '/' + 'array_maps.npy', allow_pickle='TRUE').item()
    for key in array_maps:
        # get array-like map for a specific robot heading [0.0, 90.0, 180.0, 270.0]
        array_map = array_maps[key]
        idx_list, reachable_points_num = get_index_as_tuple_list(array_map)
        #-----------------------------------------------------------------------
        '''Get 1'st tier data of (A,P,N) (50% of the entire dataset)'''
        # (A,P,N) A = (i,j,idx) where idx \in array_map[i,j]
        #-----------------------------------------------------------------------
        first_tier_AP = get_first_tier_AP(idx_list, array_map, key)
        first_tier_APN = add_negative_data(first_tier_AP, array_map, key, distanceToAnchor=1)
        if magnitude == None:
            dataset_size = round(len(first_tier_APN) / 0.50)
        else:
            dataset_size = reachable_points_num * magnitude
            first_tier_APN = cutoff_dataset(first_tier_APN, dataset_size, 0.50)
        #-----------------------------------------------------------------------
        '''Get 2'nd tier data of (A,P,N) (20% of the entire dataset)'''
        #-----------------------------------------------------------------------
        second_tier_AP = get_AP_from_previous_tier_AN(first_tier_APN)
        second_tier_APN = add_negative_data(second_tier_AP, array_map, key, distanceToAnchor=2)
        second_tier_APN = cutoff_dataset(second_tier_APN, dataset_size, 0.20)
        #-----------------------------------------------------------------------
        '''Get 3'rd tier data of (A,P,N) (15% of the entire dataset)'''
        #-----------------------------------------------------------------------
        third_tier_AP = get_AP_from_previous_tier_AN(second_tier_APN)
        third_tier_APN = add_negative_data(third_tier_AP, array_map, key, distanceToAnchor=3)
        third_tier_APN = cutoff_dataset(third_tier_APN, dataset_size, 0.15)
        #-----------------------------------------------------------------------
        '''Get 4'th tier data of (A,P,N) (10% of the entire dataset)'''
        #-----------------------------------------------------------------------
        fourth_tier_AP = get_AP_from_previous_tier_AN(third_tier_APN)
        fourth_tier_APN = add_negative_data(fourth_tier_AP, array_map, key, distanceToAnchor=4)
        fourth_tier_APN = cutoff_dataset(fourth_tier_APN, dataset_size, 0.10)
        #-----------------------------------------------------------------------
        '''Get 5'th tier data of (A,P,N) (5% of the entire dataset)'''
        #-----------------------------------------------------------------------
        fifth_tier_AP = get_AP_from_previous_tier_AN(fourth_tier_APN)
        fifth_tier_APN = add_negative_data_from_other_angle(fifth_tier_AP, key, array_maps)
        fifth_tier_APN = cutoff_dataset(fifth_tier_APN, dataset_size, 0.05)
        #-----------------------------------------------------------------------
        original_len = len(triplets_APN_idx)
        triplets_APN_idx.extend(deepcopy(first_tier_APN))
        triplets_APN_idx.extend(deepcopy(second_tier_APN))
        triplets_APN_idx.extend(deepcopy(third_tier_APN))
        triplets_APN_idx.extend(deepcopy(fourth_tier_APN))
        triplets_APN_idx.extend(deepcopy(fifth_tier_APN))
        print('Finished Generate {} Triplets for {} degree: '.format(len(triplets_APN_idx)-original_len, key) +
              '({0:.0%}, '.format(len(first_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%}, '.format(len(second_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%}, '.format(len(third_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%}, '.format(len(fourth_tier_APN)/(len(triplets_APN_idx)-original_len)) +
              '{0:.0%})'.format(len(fifth_tier_APN)/(len(triplets_APN_idx)-original_len)))
        running_total += np.array([len(first_tier_APN), len(second_tier_APN), len(third_tier_APN), len(fourth_tier_APN), len(fifth_tier_APN)], dtype=object)
    #---------------------------------------------------------------------------
    for APN in triplets_APN_idx:
        anchor = str(APN[0][3]) + '_' + str(APN[0][0]) + '_' + str(APN[0][1]) + '_' + str(APN[0][2])
        positive = str(APN[1][3]) + '_' + str(APN[1][0]) + '_' + str(APN[1][1]) + '_' + str(APN[1][2])
        negative = str(APN[2][3]) + '_' + str(APN[2][0]) + '_' + str(APN[2][1]) + '_' + str(APN[2][2])
        triplets_APN_name.append(deepcopy((anchor, positive, negative)))

    np.save(file_path + '/' + file_name, triplets_APN_name)

    return len(triplets_APN_idx), running_total

# ------------------------------------------------------------------------------
# Generate pair data points for heatmap plotting of retrieval network
# ------------------------------------------------------------------------------
def get_all_pairs_in_array_map(array_map):
    all_list = []
    slot_list, _ = get_index_as_tuple_list(array_map)
    for slot in slot_list:
        num = array_map[slot[0], slot[1]]
        items = [*range(1, num + 1)]
        for i in items:
            all_list.append((slot[0], slot[1], i))
    return list(combinations(all_list, 2))

def generate_pairs(file_path, fraction=None):
    if fraction == None:
        file_name = 'pairs_name.npy'
    else:
        file_name = 'pairs_name_fraction_' + str(fraction) + '.npy'

    pairs_name = []
    running_total = 0
    array_maps = np.load(file_path + '/' + 'array_maps.npy', allow_pickle='TRUE').item()
    for key in array_maps:
        array_map = array_maps[key]
        pairs = get_all_pairs_in_array_map(array_map)

        if fraction != None:
            endIdx = int(fraction*len(pairs))
            random.shuffle(pairs)
            pairs = deepcopy(pairs[0:endIdx])

        for pair in pairs:
            A = str(key) + '_' + str(pair[0][0]) + '_' + str(pair[0][1]) + '_' + str(pair[0][2])
            B = str(key) + '_' + str(pair[1][0]) + '_' + str(pair[1][1]) + '_' + str(pair[1][2])
            pairs_name.append(deepcopy((A, B)))

    np.save(file_path + '/' + file_name, pairs_name)

    return len(pairs_name)

# ------------------------------------------------------------------------------
# collect train, validation and test data for networks
# ------------------------------------------------------------------------------
def data_collection(partition_num, partition_list):
    # Initialize robot
    robot = Robot()
    scene_list = []
    for i_partition in partition_list:
        start_idx = math.ceil(SCENE_NUM_PER_TYPE/partition_num)*(int(i_partition) - 1) + 1
        end_idx = min([SCENE_NUM_PER_TYPE, math.ceil(SCENE_NUM_PER_TYPE/partition_num)*int(i_partition)])
        scene_list.extend([*range(start_idx, end_idx + 1)])

    # Iterate through all the scenes to collect data and separate them into train, val and test sets
    for scene_type in SCENE_TYPES:
        for scene_num in scene_list:
            if scene_num <= int(SCENE_NUM_PER_TYPE*TRAIN_FRACTION):
                FILE_PATH = DATA_DIR + '/train'
            elif scene_num <= int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)):
                FILE_PATH = DATA_DIR + '/val'
            else:
                FILE_PATH = DATA_DIR + '/test'

            robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
            robot.coordnates_patroling(saving_data=True, file_path=FILE_PATH, dynamics_rounds=5, pertubation_round=2)
            datanum = generate_triplets(FILE_PATH + '/' + robot._scene_name, magnitude=5)
            if datanum != None:
                print('{}: {} triplets'.format(robot._scene_name, datanum))

# ------------------------------------------------------------------------------
# generate triplets train, validation and test data for network image branch
def regenerate_triplets(magnitude):
    total = np.array([0.0]*5, dtype=object)
    # Iterate through all the scenes to collect data and separate them into train, val and test sets
    for scene_type in SCENE_TYPES:
        for scene_num in range(1, SCENE_NUM_PER_TYPE + 1):
            if scene_num <= int(SCENE_NUM_PER_TYPE*TRAIN_FRACTION):
                FILE_PATH = DATA_DIR + '/train'
            elif scene_num <= int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)):
                FILE_PATH = DATA_DIR + '/val'
            else:
                FILE_PATH = DATA_DIR + '/test'

            if scene_type == 'Kitchen':
            	add_on = 0
            elif scene_type == 'Living room':
            	add_on = 200
            elif scene_type == 'Bedroom':
            	add_on = 300
            elif scene_type == 'Bathroom':
            	add_on = 400
            scene_name = 'FloorPlan' + str(add_on + scene_num)
            print('----'*20)
            datanum, running_total = generate_triplets(FILE_PATH + '/' + scene_name, magnitude=magnitude)
            total += running_total
            print('Total {}: {} triplets, {}='.format(scene_name, datanum, np.sum(total)) +
                  '[{}({:.0%}), '.format(total[0], total[0]/np.sum(total)) +
                  '{}({:.0%}), '.format(total[1], total[1]/np.sum(total)) +
                  '{}({:.0%}), '.format(total[2], total[2]/np.sum(total)) +
                  '{}({:.0%}), '.format(total[3], total[3]/np.sum(total)) +
                  '{}({:.0%})]'.format(total[4], total[4]/np.sum(total)))

# ------------------------------------------------------------------------------
# Used for determine the loclaization similarity threshold
def generate_pairs_in_validation(fraction=None):
    # Iterate through all validation scenes to collect pairs to generate heatmap
    total = 0
    for scene_type in SCENE_TYPES:
        if scene_type == 'Kitchen':
        	add_on = 0
        elif scene_type == 'Living room':
        	add_on = 200
        elif scene_type == 'Bedroom':
        	add_on = 300
        elif scene_type == 'Bathroom':
        	add_on = 400

        for scene_num in range(int(SCENE_NUM_PER_TYPE*TRAIN_FRACTION) + 1, int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1):
            FILE_PATH = DATA_DIR + '/val'
            scene_name = 'FloorPlan' + str(add_on + scene_num)
            print('----'*20)
            datanum = generate_pairs(FILE_PATH + '/' + scene_name, fraction=fraction)
            total += datanum
            print('Total {}: {} pairs in Scene {}'.format(total, datanum, scene_name))

# ------------------------------------------------------------------------------
# Generate trajectory data points for navigation_network
# ------------------------------------------------------------------------------
def get_action(dH, dF):
    if dH == 0:
        action = dF*ACTION_ENCODING['forward']
    elif dH > 0:
        action = dH*ACTION_ENCODING['right'] + dF*ACTION_ENCODING['forward']
    elif dH < 0:
        action = -dH*ACTION_ENCODING['left'] + dF*ACTION_ENCODING['forward']

    action = action/np.sum(action)
    return action

def generate_uniform_trajectory(array_map, angle, fraction):

    goal_counting_arr = np.zeros((2*HORIZONTAL_MOVE_MAX+1, FORWARD_MOVE_MAX+1))
    trajectory_data = [[[] for i in range(FORWARD_MOVE_MAX+1)] for j in range(2*HORIZONTAL_MOVE_MAX+1)]
    theta = float(angle)*np.pi/180.0
    # iterate through all entries in array_map as starting point of trajectories
    for i in range(array_map.shape[0]):
        for j in range(array_map.shape[1]):
            if array_map[i, j] == 0:
                continue
            # chose goals starting at array_map[i,j] and end at array_map[i+di,j+dj]
            # with movement [dH, dF] = [Horizontal movement, Forward movement]
            for dH in range(-HORIZONTAL_MOVE_MAX, HORIZONTAL_MOVE_MAX+1): # Horizontal movement
                for dF in range(0, FORWARD_MOVE_MAX+1): # Forward movement
                    if dH == 0 and dF == 0: # no movement in this case
                        continue
                    di = round(np.cos(theta)*dH + np.sin(theta)*dF)
                    dj = round(-np.sin(theta)*dH + np.cos(theta)*dF)

                    i_goal = i + di
                    j_goal = j + dj
                    if i_goal < 0 or i_goal >= array_map.shape[0] or j_goal < 0 or j_goal >= array_map.shape[1]:
                        continue
                    if array_map[i_goal, j_goal] == 0:
                        continue
                    # horizontal movement first to get larger view overlap at the begining and end of trajectory
                    action = get_action(dH, dF)
                    # augment trajectory by the dynamics rounds
                    for cur_dynamics in range(1, array_map[i, j]+1):
                        for goal_dynamics in range(1, array_map[i_goal, j_goal]+1):
                            goal_counting_arr[dH+HORIZONTAL_MOVE_MAX,dF] += 1
                            trajectory_data[dH+HORIZONTAL_MOVE_MAX][dF].append(deepcopy(((i,j,cur_dynamics),(i_goal,j_goal,goal_dynamics), action)))

    goal_counting_arr[HORIZONTAL_MOVE_MAX,0] = np.max(goal_counting_arr)
    max_len = int(fraction*np.sum(goal_counting_arr))
    # cut off data and store in trajectory_data_augmented
    trajectory_data_augmented = []
    for dH in range(-HORIZONTAL_MOVE_MAX, HORIZONTAL_MOVE_MAX+1): # Horizontal movement
        for dF in range(0, FORWARD_MOVE_MAX+1): # Forward movement
            if dH == 0 and dF == 0: # no movement in this case
                continue
            trajectory_data_augmented.extend(deepcopy(trajectory_data[dH+HORIZONTAL_MOVE_MAX][dF]))

    random.shuffle(trajectory_data_augmented)
    trajectory_data_augmented = deepcopy(trajectory_data_augmented[0:max_len])

    return trajectory_data_augmented

def generate_trajectory(file_path, fraction=1.0):
    if fraction == 1.0:
        file_name = 'trajectories.npy'
    elif fraction < 1.0 and fraction > 0.0:
        file_name = 'trajectories_fraction_' + str(fraction) + '.npy'

    pairs_name = []
    running_total = 0
    array_maps = np.load(file_path + '/' + 'array_maps.npy', allow_pickle='TRUE').item()
    for key in array_maps:
        array_map = array_maps[key]
        pairs = generate_uniform_trajectory(array_map, key, fraction)

        for pair in pairs:
            curr = str(key) + '_' + str(pair[0][0]) + '_' + str(pair[0][1]) + '_' + str(pair[0][2])
            goal = str(key) + '_' + str(pair[1][0]) + '_' + str(pair[1][1]) + '_' + str(pair[1][2])
            action = pair[2]
            pairs_name.append(deepcopy((curr, goal, action)))

    np.save(file_path + '/' + file_name, pairs_name)
    return len(pairs_name)

def regenerate_trajectory(fraction):
    # Iterate through all validation scenes to collect pairs to generate heatmap
    total = dict(train=0, val=0, test=0)
    for scene_type in SCENE_TYPES:
        for scene_num in range(1, SCENE_NUM_PER_TYPE + 1):
            if scene_num <= int(SCENE_NUM_PER_TYPE*TRAIN_FRACTION):
                phase = 'train'
            elif scene_num <= int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)):
                phase = 'val'
            else:
                phase = 'test'
            FILE_PATH = DATA_DIR + '/' + phase

            if scene_type == 'Kitchen':
            	add_on = 0
            elif scene_type == 'Living room':
            	add_on = 200
            elif scene_type == 'Bedroom':
            	add_on = 300
            elif scene_type == 'Bathroom':
            	add_on = 400
            scene_name = 'FloorPlan' + str(add_on + scene_num)
            print('----'*20)
            running_total = generate_trajectory(FILE_PATH + '/' + scene_name, fraction=fraction)
            total[phase] += running_total
            print('{} {} points,\t Total {}'.format(scene_name, running_total, total))

# ------------------------------------------------------------------------------
# manually update topological map info
# ------------------------------------------------------------------------------
def iter_test_scene():
    # Initialize robot
    robot = Agent_Sim()
    test_idx_initial = int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1
    test_idx_end = SCENE_NUM_PER_TYPE + 1
    add_on = [2, 3, 2, 3]
    for idx, scene_type in enumerate(SCENE_TYPES):
        for scene_num in range(test_idx_initial, test_idx_end):
            robot.reset_scene(scene_type=scene_type, scene_num=scene_num, ToggleMapView=True, Show_doorway=False, shore_toggle_map=False)
            robot.show_map(show_nodes=True, show_edges=True)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate_RNet", help="regenerate triplets for RNet", action="store_true")
    parser.add_argument("--regenerate_NaviNet", help="regenerate trajectory for NaviNet", action="store_true")
    parser.add_argument("--gen_pair_in_val", help="regenerate pair data for heatmap plotting of retrieval network", action="store_true")
    parser.add_argument("--collect_partition", nargs="+", default=[])
    parser.add_argument("--topo", help="manually update topological map info", action="store_true")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Used for regenerating triplets
    if args.regenerate_RNet:
        # 0.2: 9499 triplets in total
        # 0.3: 14279 triplets
        # 0.5: 23825
        # 1: 66161 triplets in total
        regenerate_triplets(0.5)
    # Used for regenerating trajectory data for Navigation Network
    elif args.regenerate_NaviNet:
        # 0.0005: 9437
        # 0.001: 19043
        regenerate_trajectory(0.0005)
    # Used for determine the loclaization similarity threshold
    elif args.gen_pair_in_val:
        generate_pairs_in_validation(fraction=0.005)
    # Used to collect data
    elif len(args.collect_partition) > 0:
        partition_num = 30
        data_collection(partition_num, args.collect_partition)
    # Used to see visualization of each test scene and record the topological node manually
    elif args.topo:
        iter_test_scene()
