# Module for objectType info update
from ai2thor.controller import Controller
from termcolor import colored
from lib.params import *
from lib.scene_graph_generation import *
import numpy as np


# Function for construct dictionary of d[objectType] = index
# and list for lst[index] = objectType
# The obsolete index is listed in ai2thor.allenai.org/ithor/documentation/objects/actionable-properties/#table-of-object-actionable-properties

# --------------------------------------------------------------------------
# Prepare FloorPlan name list
def update_floor_plan():
    # For iTHOR
    # Kitchens: FloorPlan1 - FloorPlan30
    # Living rooms: FloorPlan201 - FloorPlan230
    # Bedrooms: FloorPlan301 - FloorPlan330
    # Bathrooms: FloorPLan401 - FloorPlan430
    iTHOR_num = np.hstack([np.arange(1,31), np.arange(201,231), np.arange(301,331), np.arange(401,431)])
    iTHOR = ['FloorPlan'+str(num) for num in iTHOR_num]

    RoboTHOR = []
    # Load Train Scene
    for i in range(1,13):
        for j in range(1,6):
            RoboTHOR.append('FloorPlan_Train'+str(i)+'_'+str(j))
    # Load Validation Scene
    for i in range(1,4):
        for j in range(1,6):
            RoboTHOR.append('FloorPlan_Val'+str(i)+'_'+str(j))
    # Save as npy file
    np.save(INFO_FILE_PATH + '/' + 'iTHOR_FloorPlan.npy', iTHOR) # Save list as .npy
    np.save(INFO_FILE_PATH + '/' + 'RoboTHOR_FloorPlan.npy', RoboTHOR) # Save list as .npy
    return iTHOR, RoboTHOR

# --------------------------------------------------------------------------
# Function used to show objectType numbers
# Modified to augment receptacle objectType numbers i.e. CounterTop1 and CounterTop2
def show_object_type_max():
    iTHOR, RoboTHOR = update_floor_plan()
    controller = Controller()
    count_rec_max = {} # used to count maximum numbers of a certain receptacle appearance
    tmp_rec_max = {} # temporary stores count_rec_max from previous iteration
    for floor_plan in (iTHOR + RoboTHOR):
        controller.reset(floor_plan)
        event = controller.step(action='Pass')
        # Iterate through the objectType and
        # count their maximum numbers of appearance for same receptacle objectType in one scene/Env
        for obj in event.metadata['objects']:
            name = obj['objectType']
            # Ignore non-informative objectType e.g. 'Floor' and non receptacles
            if name in BAN_TYPE_LIST or not obj['receptacle']:
                continue
            if name in tmp_rec_max:
                tmp_rec_max[name] += 1
            else:
                tmp_rec_max.update({name : 1})
        # Update the info in count_rec_max using tmp_rec_max
        for name in tmp_rec_max:
            if name in count_rec_max and tmp_rec_max[name] > count_rec_max[name]:
                count_rec_max[name] = tmp_rec_max[name]
            elif name not in count_rec_max:
                count_rec_max.update({name: tmp_rec_max[name]})
        tmp_rec_max = {}

    print(count_rec_max)

# --------------------------------------------------------------------------
# This dictionary is modified from the output of above function show_object_type_max():
# Output:
# REC_MAX_DIC = {'StoveBurner': 6, 'Drawer': 27, 'CounterTop': 4, 'Cabinet': 28,
#                'Microwave': 1, 'Shelf': 15, 'Toaster': 1, 'Pan': 1, 'Plate': 6,
#                'GarbageCan': 1, 'Pot': 2, 'Cup': 3, 'Fridge': 1, 'CoffeeMachine': 1,
#                'Bowl': 1, 'SinkBasin': 2, 'Mug': 1, 'Stool': 2, 'Chair': 8, 'Sink': 2,
#                'SideTable': 8, 'DiningTable': 3, 'Safe': 1, 'Box': 4, 'ArmChair': 4,
#                'CoffeeTable': 4, 'TVStand': 2, 'Sofa': 2, 'Ottoman': 1, 'Desk': 5,
#                'Dresser': 4, 'DogBed': 1, 'Bed': 2, 'LaundryHamper': 1, 'ShelvingUnit': 1,
#                'Footstool': 2, 'BathtubBasin': 1, 'Bathtub': 1, 'TowelHolder': 2,
#                'ToiletPaperHanger': 1, 'HandTowelHolder': 4, 'Toilet': 1}

# --------------------------------------------------------------------------
# This function is used to get possible parent objectType of some receptacles such as 'Shelf'
def get_type_parent(name='Shelf'):
    iTHOR, RoboTHOR = update_floor_plan()
    controller = Controller()
    parent_in_name = []
    parent_on_name = []
    SG = Scene_Graph()
    i = obj_2_idx_dic[name] # idx of objectType e.g. 'Shelf' in global vector
    for floor_plan in (iTHOR + RoboTHOR):
        controller.reset(floor_plan)
        event = controller.step(action='Pass')
        SG.reset()
        # Update SG from all objects
        SG.update_from_data(event.metadata['objects'])
        is_independent = True
        for j in range(OBJ_TYPE_NUM):
            if SG._R_on[i,j]:
                is_independent = False
                if idx_2_obj_list[j] not in parent_on_name:
                    parent_on_name.append(idx_2_obj_list[j])
            if SG._R_in[i,j]:
                is_independent = False
                if idx_2_obj_list[j] not in parent_in_name:
                    parent_in_name.append(idx_2_obj_list[j])
        if is_independent:
            print("Some shelf is independent in ", floor_plan)
    print(parent_in_name, parent_on_name)
# Result ouput:
# 'Shelf' in ['ShelvingUnit', 'DiningTable', 'SideTable', 'CoffeeTable', 'TVStand', 'Desk', 'ShowerGlass', 'Dresser']
# Some Shelf is independent

# --------------------------------------------------------------------------
# Note for deleting some of the receptacles
# 'StoveBurner': 6 is essentially one stove
# 'Plate': 6 and 'Cup': 3 are not so informative and important
# 'TowelHolder': 2 and 'HandTowelHolder': 4 is not so informative and important
# --------------------------------------------------------------------------
# Side Notes: FloorPlan206 have 15 'Shelf' ... shelf belongs to TVStand, ShelvingUnit
#             FloorPlan9 have 28 'Cabinet'
#             FloorPlan_Train12_5 have 8 'SideTable'
REC_MAX_DIC = {'Drawer': 27, 'CounterTop': 4, 'Cabinet': 28, 'Microwave': 1, 'Shelf': 15, 'Toaster': 1, 'Pan': 1, 'GarbageCan': 1, 'Pot': 2,
 'Fridge': 1, 'CoffeeMachine': 1, 'Bowl': 1, 'SinkBasin': 2, 'Mug': 1, 'Stool': 2, 'Chair': 8, 'Sink': 2, 'SideTable': 8,
 'DiningTable': 3, 'Safe': 1, 'Box': 4, 'ArmChair': 4, 'CoffeeTable': 4, 'TVStand': 2, 'Sofa': 2, 'Ottoman': 1, 'Desk': 5, 'Dresser': 4,
 'DogBed': 1, 'Bed': 2, 'LaundryHamper': 1, 'ShelvingUnit': 1, 'Footstool': 2, 'BathtubBasin': 1, 'Bathtub': 1, 'ToiletPaperHanger': 1, 'Toilet': 1}


# --------------------------------------------------------------------------
# Function used to count objectType numbers
# Modified to augment receptacle objectType numbers i.e. CounterTop1 and CounterTop2
def update_object_type():
    iTHOR, RoboTHOR = update_floor_plan()
    obj_2_idx_dic = {}
    idx_2_obj_list = []
    objType_num = 0
    controller = Controller()
    for floor_plan in (iTHOR + RoboTHOR):
        controller.reset(floor_plan)
        event = controller.step(action='Pass')
        # update obj_2_idx_dic.npy and idx_2_obj_list.npy
        for obj in event.metadata['objects']:
            name = obj['objectType']
            if name in BAN_TYPE_LIST:     # Ignore non-informative objectType e.g. 'Floor'
                continue
            if name not in obj_2_idx_dic:
                obj_2_idx_dic.update({name : objType_num})
                idx_2_obj_list.append(name)
                objType_num = objType_num + 1

    np.save(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', obj_2_idx_dic) # Save dictionary as .npy
    np.save(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy', idx_2_obj_list) # Save list as .npy

# --------------------------------------------------------------------------
# This function is used to check objectType info against official info
# in ai2thor.allenai.org/ithor/documentation/objects/actionable-properties/#table-of-object-actionable-properties
def check_object_type():
    pass

# --------------------------------------------------------------------------
# Function used to count object numbers by their name
def update_object():
    iTHOR, RoboTHOR = update_floor_plan()
    obj_list = []   # by 'name' attributes
    controller = Controller()
    for floor_plan in (iTHOR + RoboTHOR):
        controller.reset(floor_plan)
        event = controller.step(action='Pass')
        for obj in event.metadata['objects']:
            name = obj['name']
            if obj['objectType'] in BAN_TYPE_LIST:     # Ignore non-informative objectType e.g. 'Floor'
                continue
            if name not in obj_list:
                obj_list.append(name)
    print(len(obj_list))
