from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import dijkstar as dij
from distutils.util import strtobool
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import time
import copy
import random
import logging
import os
import sys
sys.path.append('./Network')
from Map import *

parser = argparse.ArgumentParser()
parser.add_argument("--scene_type", type=int, default=1,  help="Choose scene type for simulation, 1 for Kitchens, 2 for Living rooms, 3 for Bedrooms, 4 for Bathrooms")
parser.add_argument("--scene_num", type=int, default=0,  help="Choose scene num for simulation, from 1 - 30")
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


def Navigate_by_path(Dumb_Navigetion_object, Topological_map_object, path):

	init_position = Topological_map_object.Get_node_value_by_name(path[0])['position']
	_, orientation = Topological_map_object.Get_node_index_orien(path[0])

	Dumb_Navigetion_object._Agent_action.Teleport_agent(init_position)
	Dumb_Navigetion_object.Rotate_to_degree(orientation)

	rotation_standard = list(Dumb_Navigetion_object._Agent_action.Get_agent_rotation().values())

	success_case = 0

	for node_path in path:

		node_value = Topological_map_object.Get_node_value_by_name(node_path)
		_, orientation = Topological_map_object.Get_node_index_orien(node_path)

		goal_frame = node_value['image']
		goal_pose = {'position': node_value['position'], 'rotation': copy.deepcopy(rotation_standard)}
		goal_pose['rotation'][1] = orientation

		if Dumb_Navigetion_object.Navigate_by_ActionNet(image_goal=goal_frame, goal_pose=goal_pose, max_steps=Dumb_Navigetion_object._Navigation_max_try):
			success_case += 1
			print('reach node ', node_path)


if __name__ == '__main__':
	Dumb_Navigetion = Dumb_Navigetion(args.AI2THOR, args.scene_type, args.scene_num, args.grid_size,
		args.rotation_step, args.sleep_time, args.save_directory, overwrite_data=args.overwrite_data,
		for_test_data=args.test_data, debug=args.debug, more_special=True)

	Dumb_Navigetion.Set_localization_network()

	node_list = [[2.00, -1.50], [2.00, -2.50], [2.00, -3.50], [3.00, -2.50], [3.00, -3.50], [3.00, -4.50], [4.00, -1.50], [4.00, -2.50], [4.00, -4.50],
	[5.00,-4.50], [6.00, -4.50], [6.00, -2.50], [6.00, -3.25], [7.00, -3.25], [7.00, -2.50], [8.00, -3.25], [8.00, -4.25], [8.00, -2.50], [8.00, -1.50], [9.00, -2.50], [9.00, -1.50]]

	# node_generator = Node_generator(controller=Dumb_Navigetion._Agent_action._controller)
	# node_generator.Get_node_from_position(node_list)
	# node_generator.Get_connected_orientaton_by_overlap_scene()
	# node_pair_list = node_generator.Get_neighbor_nodes()
	# subnodes = node_generator.Get_connected_subnodes()
	# exit()

	node_index_list = [79, 39, 1, 77, 33, 75, 127, 103, 101, 123, 146, 257, 166, 179, 226, 199, 200, 224, 254, 250, 279]
	# print('node_index_list: ', len(node_index_list))
	# exit()
	node_pair_list = [[79, 39], [79, 1], [1, 77], [33, 75], [33, 101], [127, 103], [123, 146], [123, 166], [146, 166], [146, 179], [166, 200], [226, 199], [199, 224], [199, 254],
	[199, 250], [199, 279], [200, 224], [200, 254], [200, 279], [224, 254], [224, 250], [224, 279], [254, 250], [250, 279], [1, 33], [179, 226], [179, 199], [179, 200], [179, 250], [226, 200], [226, 224], [226, 250],
	[226, 279], [199, 200], [254, 279], [79, 75], [39, 1], [39, 77], [39, 33], [39, 101], [1, 101], [33, 123], [123, 179],
	[146, 199], [146, 224], [146, 279], [166, 179], [179, 224], [79, 33], [77, 33], [75, 101], [257, 226]]
	subnodes = [[0, 2], [0, 2], [0, 3], [0, 2], [0, 2], [0, 2], [0], [0, 2], [0, 2], [0, 2], [0, 1], [0, 1], [0, 1, 2], [0], [0], [0, 1], [0, 2], [0, 2], [0, 2], [0, 1], [0, 1], [0, 1], [0, 1, 3], [0], [1, 2, 3], [1],
	[1, 2], [1, 2], [1], [1], [1], [1], [1], [1, 2], [1, 3], [2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2], [2], [2], [2], [2, 3], [2], [3], [3], [3], [3]]

	# topo_map = Topological_map(controller=Dumb_Navigetion._Agent_action._controller, node_index_list=node_generator._node_index_list, neighbor_nodes_pair=node_pair_list)
	topo_map = Topological_map(controller=Dumb_Navigetion._Agent_action._controller, node_index_list=node_index_list, neighbor_nodes_pair=node_pair_list)
	# map.Set_Unit_rotate_func(Agent_action.Unit_rotate)
	topo_map.init_data()
	topo_map.Set_Teleport_agent_func(Dumb_Navigetion._Agent_action.Teleport_agent)
	topo_map.Set_Rotate_to_degree_func(Dumb_Navigetion._Agent_action.Rotate_to_degree)
	topo_map.Add_all_node()

	topo_map.Add_all_edges(connected_subnodes=subnodes)

	# for n, nbrs in topo_map._graph.adj.items():
	# 	print('n: ', n)
	# 	print('nbrs: ', nbrs)
	# 	print('weight: ', nbrs[])

	# exit()

	topo_map.Build_dij_graph()

	path = topo_map.Find_dij_path(current_node_index=0, current_orientation=270, goal_node_index=20, goal_orientation=90)

	path = ['node_0_degree_270', 'node_0_degree_180',
	 		'node_1_degree_180', 'node_2_degree_180',
			'node_2_degree_90', 'node_4_degree_90',
			'node_4_degree_180', 'node_5_degree_180',
			'node_5_degree_90', 'node_8_degree_90',
			'node_9_degree_90', 'node_10_degree_90',
			'node_10_degree_0', 'node_12_degree_0',
			'node_12_degree_90', 'node_13_degree_90',
			'node_15_degree_90', 'node_15_degree_0',
			'node_17_degree_0', 'node_18_degree_0',
			'node_18_degree_90', 'node_20_degree_90']

	print('path: ', path)
	# exit()

	Navigate_by_path(Dumb_Navigetion, topo_map, path)
	# print('topo_map._graph: ', topo_map._graph.nodes)

	print('self._step_poses: ', Dumb_Navigetion._step_poses)

	topo_map.show_map(show_nodes=True, show_edges=True)
