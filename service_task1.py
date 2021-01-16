import argparse, copy
import multiprocessing, time
from lib.navigation import Navigation
from Map.map_plotter import Plotter
from distutils.util import strtobool

# TODO: Store knowledge graph without visible filter which is ground truth (egg in frage)
# Distance filter for SG in local SUB_NODES_NUM
# Reasoning of subnodes taking goal object as input
# Generalize action

parser = argparse.ArgumentParser()
parser.add_argument("--scene_type", type=int, default=1,  help="Choose scene type for simulation, 1 for Kitchens, 2 for Living rooms, 3 for Bedrooms, 4 for Bathrooms")
parser.add_argument("--scene_num", type=int, default=0,  help="Choose scene num for simulation, from 1 - 30")
parser.add_argument("--save_directory", type=str, default='./data',  help="Data saving directory")
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False,  help="Output debug info if True")
parser.add_argument("--test_data", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting test dataset")
parser.add_argument("--AI2THOR", type=lambda x: bool(strtobool(x)), default=False, help="True for RobotTHOR false for ITHOR")
args = parser.parse_args()

def get_mug(navigation):
    controller = navigation.Robot._AI2THOR_controller._controller
    event = controller.step(action='LookDown', degrees=40)
    navigation.Robot.send_msg_to_client(is_reached=True)
    time.sleep(1)
    for o in event.metadata['objects']:
        if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
            # pick up the mug
            event = controller.step(action='PickupObject',
                                    objectId=o['objectId'],
                                    raise_for_failure=True)
            navigation.Robot.send_msg_to_client(is_reached=True)
            time.sleep(1)
            mug_object_id = o['objectId']
            break
    controller.step('HideObject', objectId=mug_object_id)
    navigation.Robot.send_msg_to_client(is_reached=True)
    time.sleep(1)
    controller.step(action='LookUp', degrees=40)
    navigation.Robot.send_msg_to_client(is_reached=True)
    time.sleep(1)

    return mug_object_id

def put_into_microwave(navigation, mug_object_id):
    controller = navigation.Robot._AI2THOR_controller._controller
    event = controller.step('Pass')
    # the agent should now be looking at the microwave
    for o in event.metadata['objects']:
        if o['visible'] and o['openable'] and o['objectType'] == 'Microwave':
            # open the microwave
            event = controller.step(action='OpenObject',
                                    objectId=o['objectId'],
                                    raise_for_failure=True)
            navigation.Robot.send_msg_to_client(is_reached=True)
            time.sleep(1)
            receptacle_object_id = o['objectId']
            break

    controller.step('UnhideObject', objectId=mug_object_id)
    navigation.Robot.send_msg_to_client(is_reached=True)
    time.sleep(1)
    # put the object in the microwave
    controller.step(action='PutObject', receptacleObjectId=receptacle_object_id, objectId=mug_object_id, raise_for_failure=True)
    navigation.Robot.send_msg_to_client(is_reached=True)
    time.sleep(1)
    # close the microwave
    controller.step(action='CloseObject', objectId=receptacle_object_id, raise_for_failure=True)
    navigation.Robot.send_msg_to_client(is_reached=True)
    time.sleep(1)

def navigation_fcn(server, comfirmed, initialized):
    navigation = Navigation(scene_type=args.scene_type, scene_num=args.scene_num, save_directory=args.save_directory, AI2THOR=args.AI2THOR, server=server, comfirmed=comfirmed)
    navigation.Update_node_generator()
    navigation.Update_topo_map_env()
    navigation.Update_planner_env()
    # Send information to initialize plot map
    scene_info = navigation.Robot._AI2THOR_controller.get_scene_info()
    server.send(scene_info)
	# Navigation task
    while True:
        if initialized.value:
            navigation.Closed_loop_nav(goal_node_index=8, goal_orientation=270)
            mug_object_id = get_mug(navigation)
            navigation.Closed_loop_nav(goal_node_index=1, goal_orientation=90, current_node_index=8, current_orientation=270)
            put_into_microwave(navigation, mug_object_id)
            break

def visualization_fcn(client, comfirmed, initialized):
    scene_info = client.recv()
    visualization_panel = Plotter(*scene_info, client=client, comfirmed=comfirmed)
    initialized.value = 1
    while True:
        visualization_panel.show_map()

if __name__ == '__main__':
	comfirmed = multiprocessing.Value('i')  # Int value: 1 for confirm complete task and other process can go on while 0 otherwise
	comfirmed.value = 0
	initialized = multiprocessing.Value('i')  # Int value
	initialized.value = 0
	server, client = multiprocessing.Pipe()  # server send date and client receive data

	navi_node = multiprocessing.Process(target=navigation_fcn, args=(server, comfirmed, initialized))
	visual_node = multiprocessing.Process(target=visualization_fcn, args=(client, comfirmed, initialized))
	navi_node.start()
	visual_node.start()
	navi_node.join()
	visual_node.join()
