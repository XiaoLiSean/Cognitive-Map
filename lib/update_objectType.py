# Module for objectType info update
from ai2thor.controller import Controller
from os.path import dirname, abspath
from termcolor import colored
import numpy as np

INFO_FILE_PATH = dirname(dirname(abspath(__file__))) + '/AI2THOR_info' # File path for info of iTHOR Env.

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
            if name not in obj_2_idx_dic:
                obj_2_idx_dic.update({name : objType_num})
                idx_2_obj_list.append(name)
                objType_num = objType_num + 1

    print(objType_num)
    #for i in range(objType_num):
    #    print(idx_2_obj_list[i], obj_2_idx_dic[idx_2_obj_list[i]])

    np.save(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', obj_2_idx_dic) # Save dictionary as .npy
    np.save(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy', idx_2_obj_list) # Save list as .npy





if __name__ == '__main__':
    # Reload the excel info into 'obj_2_idx_dic.npy' and 'idx_2_obj_list.npy'
    print(colored('WARNING: ','magenta') + 'The objectType in the Excel file is obsolete')
    print(colored('INFO: ','blue') + "Reloading AI2THOR objectType info into 'obj_2_idx_dic.npy' and 'idx_2_obj_list.npy'")
    update_object_type()
    print(colored('INFO: ','blue') + 'Done')
