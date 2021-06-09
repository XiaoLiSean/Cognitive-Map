import argparse, copy
import multiprocessing, time
from lib.navigation import Navigation
from Map.map_plotter import Plotter
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument("--scene_type", type=int, default=1,  help="Choose scene type for simulation, 1 for Kitchens, 2 for Living rooms, 3 for Bedrooms, 4 for Bathrooms")
parser.add_argument("--scene_num", type=int, default=30,  help="Choose scene num for simulation, from 1 - 30")
parser.add_argument("--grid_size", type=float, default=0.25,  help="Grid size of AI2THOR simulation")
parser.add_argument("--rotation_step", type=float, default=10,  help="Rotation step of AI2THOR simulation")
parser.add_argument("--sleep_time", type=float, default=0.005,  help="Sleep time between two actions")
parser.add_argument("--save_directory", type=str, default='./data',  help="Data saving directory")
parser.add_argument("--overwrite_data", type=lambda x: bool(strtobool(x)), default=False, help="overwrite the existing data or not")
parser.add_argument("--log_level", type=int, default=2,  help="Level of showing log 1-5 where 5 is most detailed")
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False,  help="Output debug info if True")
parser.add_argument("--test_data", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting test dataset")
parser.add_argument("--special", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting special long range dataset")
parser.add_argument("--AI2THOR", type=lambda x: bool(strtobool(x)), default=False, help="True for RobotTHOR false for ITHOR")
args = parser.parse_args()

def navigation_fcn(server, comfirmed, initialized):
	navigation = Navigation(netName='rnet', scene_type=args.scene_type, scene_num=args.scene_num, save_directory=args.save_directory, AI2THOR=args.AI2THOR, server=server, comfirmed=comfirmed)
	navigation.Update_node_generator()
	navigation.Update_topo_map_env()
	navigation.Update_planner_env()
	# Send information to initialize plot map
	scene_info = navigation.Robot._AI2THOR_controller.get_scene_info()
	server.send(scene_info)
	# Navigation task
	navigation.node_generator.Shuffle_scene()
	navigation.Closed_loop_nav(current_node_index=10, current_orientation=0, goal_node_index=5, goal_orientation=0)
	# navigation.Closed_loop_nav(current_node_index=10, current_orientation=180, goal_node_index=9, goal_orientation=180)
	# navigation.Closed_loop_nav(current_node_index=9, current_orientation=180, goal_node_index=3, goal_orientation=0)
	# navigation.Closed_loop_nav(current_node_index=1, current_orientation=0, goal_node_index=16, goal_orientation=0)
	# navigation.Closed_loop_nav(current_node_index=16, current_orientation=0, goal_node_index=3, goal_orientation=0)
	# navigation.Closed_loop_nav(current_node_index=3, current_orientation=0, goal_node_index=4, goal_orientation=0)
	# navigation.Closed_loop_nav(current_node_index=4, current_orientation=0, goal_node_index=4, goal_orientation=90)


	# navigation.nav_test_simplified()
	# while True:
	# 	if initialized.value:
	# 		navigation.Closed_loop_nav(goal_node_index=3, goal_orientation=270)
	# 		navigation.Closed_loop_nav(goal_node_index=2, goal_orientation=270)
	# 		break

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
