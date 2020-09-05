# Module for objectType info update
from ai2thor.controller import Controller
from termcolor import colored
from params import INFO_FILE_PATH, BAN_TYPE_LIST
import numpy as np


# Function for construct dictionary of d[objectType] = index
# and list for lst[index] = objectType
# The obsolete index is listed in ai2thor.allenai.org/ithor/documentation/objects/actionable-properties/#table-of-object-actionable-properties

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

def update_object_type():
    iTHOR, RoboTHOR = update_floor_plan()
    obj_2_idx_dic = {}
    idx_2_obj_list = []
    objType_num = 0
    controller = Controller()
    for floor_plan in (iTHOR + RoboTHOR):
        controller.reset(floor_plan)
        event = controller.step(action='Pass')
        for obj in event.metadata['objects']:
            name = obj['objectType']
            if name in BAN_TYPE_LIST:     # Ignore non-informative objectType e.g. 'Floor'
                continue
            if name not in obj_2_idx_dic:
                obj_2_idx_dic.update({name : objType_num})
                idx_2_obj_list.append(name)
                objType_num = objType_num + 1

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




if __name__ == '__main__':
    # Reload the excel info into 'obj_2_idx_dic.npy' and 'idx_2_obj_list.npy'
    print(colored('WARNING: ','magenta') + 'The objectType in the Excel file is obsolete')
    print(colored('INFO: ','blue') + "Reloading AI2THOR objectType info into 'obj_2_idx_dic.npy' and 'idx_2_obj_list.npy'")
    update_object_type()
    #update_object()    # 6336 object instances in total
    print(colored('INFO: ','blue') + 'Done')
