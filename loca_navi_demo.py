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
sys.path.append('./experiment')
from Map import *
from lib.params import *

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

class Navigation():
	def __init__(self, scene_type, scene_num, save_directory, AI2THOR):
		self.Dumb_Navigetion = Dumb_Navigetion(scene_type=scene_type, scene_num=scene_num, save_directory=save_directory, AI2THOR=AI2THOR)
		self.node_generator = Node_generator(controller=self.Dumb_Navigetion._Agent_action._controller)
		self.topo_map = Topological_map(controller=self.Dumb_Navigetion._Agent_action._controller, node_index_list=None, neighbor_nodes_pair=None)
		self.planner = Planner()
		self._fail_case_tolerance = 3
		self._valid_action_type = VALID_ACTION_TYPE

	def Update_node_generator(self):
		self.node_generator.Init_node_generator()
		self._node_list = NODES[self.Dumb_Navigetion._Agent_action.Get_scene_name()]
		print('self._node_list: ', self._node_list)
		self.node_generator.Get_node_from_position(self._node_list)
		# self.node_generator.Get_connected_orientaton_by_overlap_scene()
		# self._node_pair_list = self.node_generator.Get_neighbor_nodes()
		# self._subnodes = self.node_generator.Get_connected_subnodes()
		self._node_pair_list = [[71, 54], [71, 87], [71, 81], [71, 12], [54, 87], [54, 81], [54, 12], [40, 29], [40, 70], [40, 61],
		[40, 87], [40, 81], [29, 70], [29, 61], [29, 87], [29, 81], [70, 61], [70, 87], [70, 81], [61, 87], [61, 81], [12, 14],
		[9, 28], [9, 56], [28, 56], [14, 1], [71, 29], [71, 1], [54, 40], [54, 29], [54, 1], [40, 12], [40, 9], [40, 14], [40, 1],
		[29, 12], [29, 1], [87, 81], [87, 74], [87, 63], [12, 1], [9, 1], [74, 63], [71, 40], [71, 70],
		[71, 61], [54, 70], [54, 61], [9, 14], [28, 14], [28, 1], [56, 14], [29, 9], [87, 56], [81, 56], [12, 9], [56, 74], [56, 63]]
		self._subnodes = [[0, 1, 3], [0, 2], [0, 2], [0, 1, 2], [0, 2], [0, 2], [0, 1], [0, 1, 2, 3], [0, 2], [0, 1, 2, 3], [0], [0],
		[0, 1, 2], [0, 1, 2], [0], [0], [0, 1, 2, 3], [0], [0, 1], [0, 3], [0], [0, 1, 2, 3], [0, 2], [0, 2], [0], [0, 2, 3], [1, 2],
		[1], [1, 2, 3], [1, 2], [1], [1], [1, 3], [1], [1], [1], [1], [1, 2, 3],
		[1], [1], [1, 2, 3], [1, 2, 3], [1, 3], [2], [2], [2], [2], [2], [2, 3], [2, 3], [2, 3], [2], [3], [3], [3], [3], [3], [3]]

	def Update_topo_map_env(self):
		self.topo_map.Set_env_from_Dumb_Navigetion(dumb_navigetion=self.Dumb_Navigetion)
		self.topo_map.Update_topo_map(node_index_list=self.node_generator.Get_node_list(), node_pair_list=self._node_pair_list, connected_subnodes=self._subnodes)
		# self.topo_map.Update_topo_map(connected_subnodes=subnodes)

	def Update_planner_env(self):
		self.planner.Set_env_from_topo_map(topo_map=self.topo_map)
		self.planner.Build_dij_graph()

	def Switch_scene(self, scene_type, scene_num):

		print('DOOR_NODE[self.Dumb_Navigetion._Agent_action.Get_scene_name()]: ', DOOR_NODE[self.Dumb_Navigetion._Agent_action.Get_scene_name()])
		door_node_index, door_node_orien=DOOR_NODE[self.Dumb_Navigetion._Agent_action.Get_scene_name()]
		door_node_orien = int(door_node_orien)
		door_node_name = self.topo_map.Get_node_name(node_num=door_node_index, orientation=door_node_orien)
		match, current_node_name = self.Node_localize(starting_node_name=door_node_name)

		if match:
			current_node_index, current_node_orientation = self.topo_map.Get_node_index_orien(node_name=current_node_name)

		self.Closed_loop_nav(current_node_index=current_node_index, current_orientation=current_node_orientation,
							 goal_node_index=door_node_index, goal_orientation=door_node_orien)

		self.Dumb_Navigetion.Reset_scene(scene_type=scene_type, scene_num=scene_num)

		door_node_index_new, door_node_orien_new = DOOR_NODE[self.Dumb_Navigetion._Agent_action.Get_scene_name()]
		door_node_orien_new = int(door_node_orien_new)
		door_node_name_new = self.topo_map.Get_node_name(node_num=door_node_index_new, orientation=door_node_orien_new)
		door_node_position, _, _ = self.topo_map.Get_node_value_by_name(node_name=door_node_name_new)
		self.Dumb_Navigetion._Agent_action.Teleport_agent(door_node_position)
		self.Dumb_Navigetion.Rotate_to_degree(goal_degree=self.Dumb_Navigetion.Wrap_to_degree(degree=(door_node_orien_new + 180)))

		self.Update_node_generator()
		self.Update_topo_map_env()
		self.Update_planner_env()

	def Task_nav(self, task_objects):

		door_node_index_orien = DOOR_NODE[self.Dumb_Navigetion._Agent_action.Get_scene_name()]
		door_node_name = 'node_' + str(door_node_index_orien[0]) + '_degree_' + str(door_node_index_orien[1])
		match, current_node_name = self.Node_localize(starting_node_name=door_node_name)
		goal_node_names = []

		for task_object in task_objects:
			goal_node_names.append(self.topo_map.Get_object_closest_node(task_object))
                
		for i, goal_node_name in enumerate(goal_node_names):

			current_node_index, current_node_orientation = self.topo_map.Get_node_index_orien(current_node_name)
			goal_node_index, goal_node_orientation = self.topo_map.Get_node_index_orien(goal_node_name)
			self.Closed_loop_nav(current_node_index=current_node_index, current_orientation=current_node_orientation,
								 goal_node_index=goal_node_index, goal_orientation=goal_node_orientation)
			current_node_name = copy.deepcopy(goal_node_name)

	def Node_localize(self, starting_node_name):

		match = False
		node_matched_name = None
		search_queue = list(self.topo_map.Get_node_adj_by_name(node_name=starting_node_name).keys())
		searched_node = []

		while len(search_queue) > 0:

			node_pos, node_image, _ = self.topo_map.Get_node_value_by_name(search_queue[0])
			searching_node_index, searching_node_orientation = self.topo_map.Get_node_index_orien(search_queue[0])
			searching_node_pose = {'position': node_pos, 'rotation': {'x': 0, 'y': searching_node_orientation, 'z': 0}}

			if self.Dumb_Navigetion.Navigation_stop(image_goal=node_image, image_current=self.Dumb_Navigetion._Agent_action.Get_frame(), goal_pose=None, hardcode=False):
				match = True
				node_matched_name = copy.deepcopy(search_queue[0])
				search_queue.clear()

			else:
				searched_node.append(search_queue[0])
				searching_node_adj = list(self.topo_map.Get_node_adj_by_name(node_name=search_queue[0]).keys())
				search_queue.remove(search_queue[0])

				for node_name in searching_node_adj:
					if node_name in searched_node or node_name in search_queue:
						pass
					else:
						search_queue.append(node_name)

		return match, node_matched_name

	def Closed_loop_nav(self, current_node_index=0, current_orientation=270, goal_node_index=5, goal_orientation=270):
		path = self.planner.Find_dij_path(current_node_index=current_node_index, current_orientation=current_orientation,
										  goal_node_index=goal_node_index, goal_orientation=goal_orientation)
		print('path: ', path)

		nav_result = self.Navigate_by_path(path=path)

		print('nav_result: ', nav_result)
		# exit()
		while isinstance(nav_result, str):

			match = False
			while not match:
				success, _ = self.Dumb_Navigetion.Random_move_w_weight(all_actions=False, weight=[0.25, 0.25, 0.25, 0.25])
				match, node_matched_name = self.Node_localize(starting_node_name=nav_result)
				print('node_matched_name: ', node_matched_name)

			searching_node_index, searching_node_orientation = self.topo_map.Get_node_index_orien(node_matched_name)

			if node_matched_name in path:
				path = path[path.index(node_matched_name):]
				print('path: ', path)
			else:
				path = self.planner.Find_dij_path(current_node_index=searching_node_index, current_orientation=searching_node_orientation,
							  goal_node_index=goal_node_index, goal_orientation=goal_orientation)
				print('repath: ', path)

			# match_node = False

			# while not match_node:

			# 	success, _ = self.Dumb_Navigetion.Random_move_w_weight(all_actions=False, weight=[0.25, 0.25, 0.25, 0.25])
			# 	search_queue = list(self.topo_map.Get_node_adj_by_name(node_name=nav_result).keys())
			# 	# print('search_queue: ', search_queue)
			# 	searched_node = []

			# 	while len(search_queue) > 0:

			# 		node_pos, node_image, _ = self.topo_map.Get_node_value_by_name(search_queue[0])
			# 		searching_node_index, searching_node_orientation = self.topo_map.Get_node_index_orien(search_queue[0])
			# 		searching_node_pose = {'position': node_pos, 'rotation': {'x': 0, 'y': searching_node_orientation, 'z': 0}}

			# 		if self.Dumb_Navigetion.Navigation_stop(image_goal=node_image, image_current=self.Dumb_Navigetion._Agent_action.Get_frame(), goal_pose=None, hardcode=False):
						
			# 			match_node = True
			# 			print('matched node: ', search_queue[0])
			# 			if search_queue[0] in path:
			# 				path = path[path.index(search_queue[0]):]
			# 			else:
			# 				path = self.planner.Find_dij_path(current_node_index=searching_node_index, current_orientation=searching_node_orientation,
			# 							  goal_node_index=goal_node_index, goal_orientation=goal_orientation)
			# 				print('path: ', path)
			# 			search_queue.clear()

			# 		else:
			# 			searched_node.append(search_queue[0])
			# 			searching_node_adj = list(self.topo_map.Get_node_adj_by_name(node_name=search_queue[0]).keys())
			# 			search_queue.remove(search_queue[0])

			# 			for node_name in searching_node_adj:
			# 				if node_name in searched_node or node_name in search_queue:
			# 					pass
			# 				else:
			# 					search_queue.append(node_name)

			nav_result = self.Navigate_by_path(path=path)

	def Navigate_by_path(self, path):

		init_position = self.topo_map.Get_node_value_dict_by_name(path[0])['position']
		_, orientation = self.topo_map.Get_node_index_orien(path[0])

		self.Dumb_Navigetion._Agent_action.Teleport_agent(init_position)
		self.Dumb_Navigetion.Rotate_to_degree(orientation)

		rotation_standard = list(self.Dumb_Navigetion._Agent_action.Get_agent_rotation().values())

		failed_case = 0

		for node_path_num, node_path in enumerate(path):

			node_value = self.topo_map.Get_node_value_dict_by_name(node_path)
			_, orientation = self.topo_map.Get_node_index_orien(node_path)

			goal_frame = node_value['image']
			goal_pose = {'position': node_value['position'], 'rotation': copy.deepcopy(rotation_standard)}
			goal_pose['rotation'][1] = orientation

			if self.Dumb_Navigetion.Navigate_by_ActionNet(image_goal=goal_frame, goal_pose=goal_pose, max_steps=self.Dumb_Navigetion._Navigation_max_try):
				failed_case = 0
				print('reach node ', node_path)
				time.sleep(1)
			else:
				failed_case += 1
				print('failed case: ', failed_case)
				if failed_case >= self._fail_case_tolerance or node_path == path[-1]:
					return node_path
		return True

# def Navigate_by_path(Dumb_Navigetion_object, Topological_map_object, path):

# 	init_position = Topological_map_object.Get_node_value_dict_by_name(path[0])['position']
# 	_, orientation = Topological_map_object.Get_node_index_orien(path[0])

# 	Dumb_Navigetion_object._Agent_action.Teleport_agent(init_position)
# 	Dumb_Navigetion_object.Rotate_to_degree(orientation)

# 	rotation_standard = list(Dumb_Navigetion_object._Agent_action.Get_agent_rotation().values())

# 	success_case = 0

# 	for node_path in path:

# 		node_value = Topological_map_object.Get_node_value_dict_by_name(node_path)
# 		_, orientation = Topological_map_object.Get_node_index_orien(node_path)

# 		goal_frame = node_value['image']
# 		goal_pose = {'position': node_value['position'], 'rotation': copy.deepcopy(rotation_standard)}
# 		goal_pose['rotation'][1] = orientation

# 		if Dumb_Navigetion_object.Navigate_by_ActionNet(image_goal=goal_frame, goal_pose=goal_pose, max_steps=Dumb_Navigetion_object._Navigation_max_try):
# 			success_case += 1
# 			print('reach node ', node_path)
# 			time.sleep(2)



if __name__ == '__main__':

	# dij_graph = dij.Graph()
	# dij_graph.add_edge(0, 1, 1)
	# dij_graph.add_edge(0, 1, 0.5)
	# dij_graph.add_edge(1, 2, 0.5)
	# print(dij_graph)
	# exit()

	navigation = Navigation(scene_type=args.scene_type, scene_num=args.scene_num, save_directory=args.save_directory, AI2THOR=args.AI2THOR)
	navigation.Update_node_generator()
	navigation.Update_topo_map_env()
	navigation.Update_planner_env()
	navigation.Closed_loop_nav()

	# navigation.Switch_scene(scene_type=2, scene_num=26)
	time.sleep(2)

	exit()




	Dumb_Navigetion = Dumb_Navigetion(args.AI2THOR, args.scene_type, args.scene_num, args.grid_size,
		args.rotation_step, args.sleep_time, args.save_directory, overwrite_data=args.overwrite_data,
		for_test_data=args.test_data, debug=args.debug, more_special=True)

	# Dumb_Navigetion.Set_localization_network()

	node_list = [[2.00, -1.50], [2.00, -2.50], [2.00, -3.50], [3.00, -2.50], [3.00, -3.50], [3.00, -4.50], [4.00, -1.50], [4.00, -2.50], [4.00, -4.50], 
	[5.00,-4.50], [6.00, -4.50], [6.00, -2.50], [6.00, -3.25], [7.00, -3.25], [7.00, -2.50], [8.00, -3.25], [8.00, -4.25], [8.00, -2.50], [8.00, -1.50], [9.00, -2.50], [9.00, -1.50]]

	node_generator = Node_generator(controller=Dumb_Navigetion._Agent_action._controller)
	node_generator.Get_node_from_position(node_list)
	node_generator.Get_connected_orientaton_by_overlap_scene()
	node_pair_list = node_generator.Get_neighbor_nodes()
	subnodes = node_generator.Get_connected_subnodes()
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

	topo_map.Set_env_from_Dumb_Navigetion(dumb_navigetion=Dumb_Navigetion)
	topo_map.Update_topo_map(connected_subnodes=subnodes)

	# for n, nbrs in topo_map._graph.adj.items():
	# 	print('n: ', n)
	# 	print('nbrs: ', nbrs)
	# 	# print('weight: ', nbrs[])
	# print('topo_map._graph.adj.items(): ', topo_map._graph.adj)
	# exit()

	planner = Planner()
	planner.Set_env_from_topo_map(topo_map=topo_map)

	planner.Build_dij_graph()

	path = planner.Find_dij_path(current_node_index=0, current_orientation=270, goal_node_index=20, goal_orientation=270)
	print('path: ', path)
	# exit()

	Navigate_by_path(Dumb_Navigetion, topo_map, path)
	# print('topo_map._graph: ', topo_map._graph.nodes)

	print('self._step_poses: ', Dumb_Navigetion._step_poses)

	topo_map.show_map(show_nodes=True, show_edges=True)