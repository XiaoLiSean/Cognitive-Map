from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import dijkstar as dij
from distutils.util import strtobool
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import time, copy, random
import logging, os, sys
sys.path.append('./Network') # import navi and retrieval_network
sys.path.append('./experiment') # import simulation robot
from Map import * # import topological map
from lib.params import *


class Navigation():
	def __init__(self, scene_type, scene_num, save_directory, AI2THOR, isResNetLocalization=False, server=None, comfirmed=None, netName='rnet'):
		self.Robot = Robot(scene_type=scene_type, scene_num=scene_num, netName=netName, save_directory=save_directory, AI2THOR=AI2THOR, server=server, comfirmed=comfirmed)
		self.node_generator = Node_generator(controller=self.Robot._AI2THOR_controller._controller)
		self.topo_map = Topological_map(controller=self.Robot._AI2THOR_controller._controller, node_index_list=None, neighbor_nodes_pair=None)
		self.planner = Planner()
		self.planner.Set_planning_method(using_subnode=True)
		self._fail_case_tolerance = 1
		self._valid_action_type = VALID_ACTION_TYPE
		self._fail_type = {'translation': 0, 'rotation': 0}
		self._fail_types = {'navigation': 0, 'localization': 0}
		self._action_case_num = 0
		self._action_success_num = 0
		self._nav_test_case_num = 0
		self._nav_success_case_num = 0
		self._node_list = None
		self._impassable_edges = []
		self._impassable_reason = []

	def nav_test(self):
		for scene_type in range(1, 4):
			for scene_num in range(26, 28):
				self.Switch_scene(scene_type=scene_type, scene_num=scene_num)
				# self.Plotter.show_map(show_nodes=False)
				for start in range(len(self._node_list)):
					for goal in range(len(self._node_list)):
						start_orien = random.choice([0, 90, 180, 270])
						goal_orien = random.choice([0, 90, 180, 270])
						self.Closed_loop_nav(current_node_index=start, current_orientation=start_orien, goal_node_index=goal, goal_orientation=goal_orien)
		self.Write_csv()

	def nav_test_single_scene(self):
		for start in range(len(self._node_list)):
			for goal in range(len(self._node_list)):
				start_orien = random.choice([0, 90, 180, 270])
				goal_orien = random.choice([0, 90, 180, 270])
				self.Closed_loop_nav(current_node_index=start, current_orientation=start_orien, goal_node_index=goal, goal_orientation=goal_orien)
		self.Write_csv()

	def nav_test_simplified(self):

		neighbor_nodes = [[] for _ in range(len(self._node_list))]
		node_coor_indexes = []

		for i, node_coor in enumerate(self._node_list):
			node_coordinate = {'x': node_coor[0], 'y': self.topo_map._reachable[0]['y'], 'z': node_coor[1]}
			node_coor_indexes.append(self.topo_map._reachable.index(node_coordinate))

		for i, node_coor_index in enumerate(node_coor_indexes):

			for node_pair in self.topo_map._neighbor_nodes_pair:
				if node_coor_index in node_pair:
					if node_coor_index == node_pair[0] and not node_coor_indexes.index(node_pair[1]) in neighbor_nodes[i]:
						neighbor_nodes[i].append(node_coor_indexes.index(node_pair[1]))
					if node_coor_index == node_pair[1] and not node_coor_indexes.index(node_pair[0]) in neighbor_nodes[i]:
						neighbor_nodes[i].append(node_coor_indexes.index(node_pair[0]))

		self._impassable_edges = []
		tested_neighbor_case = 0
		failed_neighbor_case = 0
		for start_node_i in range(len(self._node_list)):
		# for start_node_i in range(1):

			for goal_node_index in neighbor_nodes[start_node_i]:

				for orientation_test in [0, 90, 180, 270]:

					path = self.planner.Find_dij_path(current_node_index=start_node_i, current_orientation=orientation_test,
										  goal_node_index=goal_node_index, goal_orientation=orientation_test)
					if len(path) > 2:
						continue
					nav_result, _ = self.Closed_loop_nav(current_node_index=start_node_i, current_orientation=orientation_test,
											goal_node_index=goal_node_index, goal_orientation=orientation_test)
					tested_neighbor_case += 1
					if not nav_result is True:
						# self._impassable_edges.append('node_' + str(start_node_i) + '_degree_' + str(orientation_test) + 'node_' + str(goal_node_index) + '_degree_' + str(orientation_test))
						# self._impassable_edges.append(
						# 				  self._build_impassable_edge_name(start_node_index=start_node_i, start_node_orientation=orientation_test,
						# 				  goal_node_index=goal_node_index, goal_node_orientation=orientation_test)
						# 				  )
						failed_neighbor_case += 1

			for orientation_test in [0, 90, 180, 270]:

				for orientation_difference in [90, -90]:

					path = self.planner.Find_dij_path(current_node_index=start_node_i, current_orientation=orientation_test,
											  goal_node_index=start_node_i, goal_orientation=self.planner._wrap_to_360(degree=orientation_test+orientation_difference))

					if len(path) > 2:
						continue

					nav_result, _ = self.Closed_loop_nav(current_node_index=start_node_i, current_orientation=orientation_test,
											goal_node_index=start_node_i, goal_orientation=self.planner._wrap_to_360(degree=orientation_test+orientation_difference))
					tested_neighbor_case += 1
					if not nav_result is True:
						failed_neighbor_case += 1

		case_num = 0
		fail_case_num = 0

		# print('tested_neighbor_case:', tested_neighbor_case)
		# print('failed_neighbor_case: ', failed_neighbor_case)
		# impassable_edges.append('node_0_degree_90node_1_degree_90')

		for start in range(len(self._node_list)):

			for goal in range(len(self._node_list)):

				for start_orien in [0, 90, 180, 270]:

					for goal_orien in [0, 90, 180, 270]:

						path = self.Find_dij_path_wt_impassable(current_node_index=start, current_orientation=start_orien,
												  goal_node_index_=goal, goal_orientation=goal_orien)
						if path is False:
							fail_case_num += 1
						case_num += 1

		# print('fail_case_num: ', fail_case_num)
		# print('case_num: ', case_num)
		# print('self._fail_types: ', self._fail_types)

		loca_neighbor_error = list(filter(lambda x: x == 'localization', self._impassable_reason))
		loca_neighbor_error_num = len(loca_neighbor_error)
		navi_neighbor_error_num = len(self._impassable_reason) - loca_neighbor_error_num

		# print('loca_neighbor_error_num: ', loca_neighbor_error_num)
		# print('navi_neighbor_error_num: ', navi_neighbor_error_num)

		nav_test = open('service_task_test.csv', 'a')
		nav_test_writer = csv.writer(nav_test)
		nav_test_writer.writerow([case_num, fail_case_num, tested_neighbor_case, failed_neighbor_case, navi_neighbor_error_num, loca_neighbor_error_num,
		self._fail_types['navigation'], self._fail_types['localization']])

		return fail_case_num / case_num

	def _build_impassable_edge_name(self, start_node_index=None, start_node_orientation=None,
										  goal_node_index=None, goal_node_orientation=None,
										  start_node_name=None, goal_node_name=None):
		if not start_node_name is None and not goal_node_name is None:
			return start_node_name + '_' + goal_node_name
		else:
			return self.topo_map.Get_node_name(node_num=start_node_index, orientation=start_node_orientation) + '_' +\
				   self.topo_map.Get_node_name(node_num=goal_node_index, orientation=goal_node_orientation)

	def _get_node_index_w_orientation(self, impassable_edge_name):
		impassable_edge_name_splited = impassable_edge_name.split('_')
		return impassable_edge_name_splited[1], impassable_edge_name_splited[3], impassable_edge_name_splited[5], impassable_edge_name_splited[7]

	def Write_csv(self):
		nav_test = open('nav_test.csv','a')
		nav_test_writer = csv.writer(nav_test)
		nav_test_writer.writerow([self._nav_test_case_num, self._nav_success_case_num,
								  self._action_case_num, self._action_success_num, self._fail_type['translation'], self._fail_type['rotation']])


	def Update_node_generator(self):
		self.node_generator.Init_node_generator()
		self._node_list = NODES[self.Robot._AI2THOR_controller.Get_scene_name()]
		# print('self._node_list: ', self._node_list)
		self.node_generator.Get_node_from_position(self._node_list)
		self.node_generator.Get_connected_orientaton_by_geometry()
		# index of subnodes in reachable points
		self._node_pair_list = self.node_generator.Get_neighbor_nodes()
		# corresponding to _node_pair_list
		self._subnodes = self.node_generator.Get_connected_subnodes()

		# print('self._node_pair_list: ', len(self._node_pair_list))

	def Update_topo_map_env(self):
		self.topo_map.Set_env_from_Robot(Robot=self.Robot)
		self.topo_map.Update_topo_map(node_index_list=self.node_generator.Get_node_list(), node_pair_list=self._node_pair_list, connected_subnodes=self._subnodes)
		return self.topo_map

	def Update_planner_env(self):
		self.planner.Set_env_from_topo_map(topo_map=self.topo_map)
		self.planner.Build_dij_graph()

	def Switch_scene(self, scene_type, scene_num, shuffle=True):

		# print('DOOR_NODE[self.Robot._AI2THOR_controller.Get_scene_name()]: ', DOOR_NODE[self.Robot._AI2THOR_controller.Get_scene_name()])
		door_node_index, door_node_orien = DOOR_NODE[self.Robot._AI2THOR_controller.Get_scene_name()]
		door_node_orien = int(door_node_orien)
		door_node_name = self.topo_map.Get_node_name(node_num=door_node_index, orientation=door_node_orien)
		match, current_node_name = self.Node_localize_BF(starting_node_name=door_node_name)

		if match:
			current_node_index, current_node_orientation = self.topo_map.Get_node_index_orien(node_name=current_node_name)

		self.Closed_loop_nav(current_node_index=current_node_index, current_orientation=current_node_orientation,
							 goal_node_index=door_node_index, goal_orientation=door_node_orien)

		self.Robot.Reset_scene(scene_type=scene_type, scene_num=scene_num)

		self.Update_node_generator()
		self.Update_topo_map_env()
		self.Update_planner_env()

		if shuffle:
			self.node_generator.Shuffle_scene()

		door_node_index_new, door_node_orien_new = DOOR_NODE[self.Robot._AI2THOR_controller.Get_scene_name()]
		door_node_orien_new = int(door_node_orien_new)
		door_node_name_new = self.topo_map.Get_node_name(node_num=door_node_index_new, orientation=door_node_orien_new)
		door_node_position, _, _ = self.topo_map.Get_node_value_by_name(node_name=door_node_name_new)
		self.Robot._AI2THOR_controller.Teleport_agent(door_node_position)
		self.Robot._AI2THOR_controller.Rotate_to_degree(goal_degree=self.Robot._AI2THOR_controller.Wrap_to_degree(degree=(door_node_orien_new + 180)))

	def Task_nav(self, task_objects):

		door_node_index_orien = DOOR_NODE[self.Robot._AI2THOR_controller.Get_scene_name()]
		door_node_name = 'node_' + str(door_node_index_orien[0]) + '_degree_' + str(door_node_index_orien[1])
		match, current_node_name = self.Node_localize_BF(starting_node_name=door_node_name)
		goal_node_names = []

		for task_object in task_objects:
			goal_node_names.append(self.topo_map.Get_object_closest_node(task_object))

		for i, goal_node_name in enumerate(goal_node_names):

			current_node_index, current_node_orientation = self.topo_map.Get_node_index_orien(current_node_name)
			goal_node_index, goal_node_orientation = self.topo_map.Get_node_index_orien(goal_node_name)
			self.Closed_loop_nav(current_node_index=current_node_index, current_orientation=current_node_orientation,
								 goal_node_index=goal_node_index, goal_orientation=goal_node_orientation)
			current_node_name = copy.deepcopy(goal_node_name)

	def Node_localize_BF(self, starting_node_name):

		match = False
		node_matched_name = None
		search_queue = list(self.topo_map.Get_node_adj_by_name(node_name=starting_node_name).keys())
		searched_node = []

		while len(search_queue) > 0:

			node_pos, node_image, _ = self.topo_map.Get_node_value_by_name(search_queue[0])
			searching_node_index, searching_node_orientation = self.topo_map.Get_node_index_orien(search_queue[0])
			searching_node_pose = {'position': node_pos, 'rotation': {'x': 0, 'y': searching_node_orientation, 'z': 0}}

			# if self.Robot.Navigation_stop(image_goal=node_image, image_current=self.Robot._AI2THOR_controller.Get_frame(), goal_pose=None, hardcode=False):
			if self.Robot.HardCodeLocalization(goal_pose=searching_node_pose):
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

	def Closed_loop_nav(self, goal_node_index=5, goal_orientation=270, current_node_index=None, current_orientation=None):

		if current_node_index is None and current_orientation is None:
			current_node_index = self.Robot._AI2THOR_controller.Get_agent_current_pos_index()
			if current_node_index in self.topo_map._node_index_list:
				current_node_index = self.topo_map._node_index_list.index(current_node_index)
			else:
				current_node_index = 0
			current_orientation = self.Robot._AI2THOR_controller.Get_agent_current_orientation()
		path = self.planner.Find_dij_path(current_node_index=current_node_index, current_orientation=current_orientation,
										  goal_node_index=goal_node_index, goal_orientation=goal_orientation)
		# print('path: ', path)

		nav_result = self.Navigate_by_path(path=path)

		self._nav_test_case_num += 1
		if nav_result is True:
			self._nav_success_case_num += 1

		return nav_result

		# print('nav_result: ', nav_result)
		# exit()
		while isinstance(nav_result, str):

			match = False
			while not match:
				success, _ = self.Robot._AI2THOR_controller.Random_move_w_weight(all_actions=False, weight=[0.25, 0.25, 0.25, 0.25])
				match, node_matched_name = self.Node_localize_BF(starting_node_name=nav_result)
				# print('node_matched_name: ', node_matched_name)

			searching_node_index, searching_node_orientation = self.topo_map.Get_node_index_orien(node_matched_name)

			if node_matched_name in path:
				path = path[path.index(node_matched_name):]
				# print('path: ', path)
			else:
				path = self.planner.Find_dij_path(current_node_index=searching_node_index, current_orientation=searching_node_orientation,
							  goal_node_index=goal_node_index, goal_orientation=goal_orientation)
				# print('repath: ', path)

			nav_result = self.Navigate_by_path(path=path)

	def set_temp_dij_edge_cost(self, start, goal, orientation, cost, dij_use_subnode=True):
		self.planner._dij_graph.add_edge(start, goal, cost)

	def set_temp_dij_edge_cost_by_name(self, start_node_name, goal_node_name, cost, dij_use_subnode=True):

		if dij_use_subnode:
			start = self.planner.Get_subnode_dij_index(node_name=start_node_name)
			goal = self.planner.Get_subnode_dij_index(node_name=goal_node_name)
		else:
			start = self.planner.Get_node_dij_index(node_name=start_node_name)
			goal = self.planner.Get_node_dij_index(node_name=goal_node_name)

		self.planner._dij_graph.add_edge(start, goal, cost)

	def Find_dij_path_wt_impassable(self, current_node_index, current_orientation, goal_node_index_, goal_orientation):

		max_try = 100
		abort = False
		edges_changed_temp = []
		a_large_cost = 100000

		for i in range(max_try):

			if abort:
				break

			impassable_edge_clear = True
			path_temp = self.planner.Find_dij_path(current_node_index=current_node_index, current_orientation=current_orientation,
							  goal_node_index=goal_node_index_, goal_orientation=goal_orientation)

			# if goal_node_index > 20 or current_node_index > 20:
			# 	print('current_node_index: ', current_node_index)
			# 	print('goal_node_index: ', goal_node_index)

			# print(path_temp)

			if len(path_temp) > 1:
				for path_i in range(len(path_temp) - 1):
					edge_temp = path_temp[path_i] + path_temp[path_i + 1]
					edge_temp = self._build_impassable_edge_name(start_node_name=path_temp[path_i], goal_node_name=path_temp[path_i + 1])
					if edge_temp in self._impassable_edges:

						# print('edge_temp in self._impassable_edges: ', edge_temp)

						impassable_edge_clear = False
						start_node_index, _ = self.topo_map.Get_node_index_orien(node_name=path_temp[path_i])
						goal_node_index, _ = self.topo_map.Get_node_index_orien(node_name=path_temp[path_i + 1])

						if self.planner._subnode_plan:
							start_node_index = self.planner.Get_subnode_dij_index(node_name=path_temp[path_i])
							goal_node_index = self.planner.Get_subnode_dij_index(node_name=path_temp[path_i + 1])
						else:
							start_node_index = self.planner.Get_node_dij_index(node_name=path_temp[path_i])
							goal_node_index = self.planner.Get_node_dij_index(node_name=path_temp[path_i + 1])

						if self.planner._dij_graph[start_node_index][goal_node_index] == a_large_cost:
							abort = True
							self._fail_types[self._impassable_reason[self._impassable_edges.index(edge_temp)]] += 1
							# print('path_temp: ', path_temp)
							# print(self._impassable_reason[self._impassable_edges.index(edge_temp)])
							# print(edge_temp)
							# print('self.planner._subnode_plan: ', self.planner._subnode_plan)
							# print('i: ', i)
							break

						# edges_changed_temp.append([start_node_index, goal_node_index, self.planner._dij_graph[start_node_index][goal_node_index]])
						# self.set_temp_dij_edge_cost(start=start_node_index, goal=goal_node_index, cost=a_large_cost)
						edges_changed_temp.append([path_temp[path_i], path_temp[path_i + 1], self.planner._dij_graph[start_node_index][goal_node_index]])
						self.set_temp_dij_edge_cost_by_name(start_node_name=path_temp[path_i], goal_node_name=path_temp[path_i + 1], cost=a_large_cost, dij_use_subnode=self.planner._subnode_plan)
						break

			if impassable_edge_clear:
				for edges_changed in edges_changed_temp:
					self.set_temp_dij_edge_cost_by_name(start_node_name=edges_changed[0], goal_node_name=edges_changed[1], cost=edges_changed[2], dij_use_subnode=self.planner._subnode_plan)
				# print('path_temp: ', path_temp)
				return path_temp

		for edges_changed in edges_changed_temp:
			self.set_temp_dij_edge_cost_by_name(start_node_name=edges_changed[0], goal_node_name=edges_changed[1], cost=edges_changed[2], dij_use_subnode=self.planner._subnode_plan)
		# print('edges_changed_temp: ', edges_changed_temp)
		# print(self.planner._dij_graph)

		return False

	def Navigate_by_path(self, path):

		init_position = self.topo_map.Get_node_value_dict_by_name(path[0])['position']
		_, orientation = self.topo_map.Get_node_index_orien(path[0])

		self.Robot._AI2THOR_controller.Teleport_agent(init_position, position_localize=True)
		self.Robot._AI2THOR_controller.Rotate_to_degree(orientation)

		rotation_standard = list(self.Robot.Get_robot_orientation().values())

		failed_case = 0
		# fail_types = {'navigation': 0, 'localization': 0}

		for node_path_num, node_path in enumerate(path):

			# print('node_path in Navigate_by_path: ', node_path)

			node_value = self.topo_map.Get_node_value_dict_by_name(node_path)
			_, orientation = self.topo_map.Get_node_index_orien(node_path)

			goal_frame = node_value['image']
			goal_scene_graph = node_value['scene_graph']
			goal_pose = {'position': node_value['position'], 'rotation': copy.deepcopy(rotation_standard)}
			goal_pose['rotation'][1] = orientation
			goal_action_type = 'translation'
			if node_path_num > 0:
				_, orientation_pre = self.topo_map.Get_node_index_orien(path[node_path_num - 1])
				if not orientation_pre == orientation:
					goal_action_type = 'rotation'

			################
			# This part is to hardcode the rotation action
			if goal_action_type == 'rotation':
				rotation_degree = int(orientation - orientation_pre)
				if rotation_degree == 270:
					rotation_degree = -90
				elif rotation_degree == -270:
					rotation_degree = 90
				# print('rotation hard code', rotation_degree)
			else:
				rotation_degree = None
			####################
			# rotation_degree = None

			self._action_case_num += 1

			if node_path_num == 0:
				nav_by_actionnet_result = True
			else:
				nav_by_actionnet_result = self.Robot.Navigate_by_ActionNet(image_goal=goal_frame, goal_pose=goal_pose, goal_scene_graph=goal_scene_graph, max_steps=self.Robot._Navigation_max_try, rotation_degree=rotation_degree)

			if nav_by_actionnet_result is True:
				failed_case = 0

				self._action_success_num += 1
				# print('reach node ', node_path)
			else:
				failed_case += 1
				# self._fail_types[nav_by_actionnet_result[1]] += 1
				# print('self._fail_types: ', self._fail_types)

				# Actually the case that this fails when node_path_num == 0 happens...
				if node_path_num > 0:
					start_node_index, orientation_start = self.topo_map.Get_node_index_orien(node_name=path[node_path_num - 1])
					goal_node_index, orientation_goal = self.topo_map.Get_node_index_orien(node_name=node_path)
					# impassable_edge = 'node_' + str(start_node_index) + '_degree_' + str(orientation_start) + 'node_' + str(goal_node_index) + '_degree_' + str(orientation_goal)
					impassable_edge = self._build_impassable_edge_name(start_node_index=start_node_index, start_node_orientation=orientation_start,
										  goal_node_index=goal_node_index, goal_node_orientation=orientation_goal)
					if not impassable_edge in self._impassable_edges:
						self._impassable_edges.append(impassable_edge)
						self._impassable_reason.append(nav_by_actionnet_result[1])

				# if node_path_num == 0 and len(path) > 1:
				# 	start_node_index, orientation_start = self.topo_map.Get_node_index_orien(node_name=path[node_path_num])
				# 	goal_node_index, orientation_goal = self.topo_map.Get_node_index_orien(node_name=path[node_path_num + 1])
				# 	impassable_edge = self._build_impassable_edge_name(start_node_index=start_node_index, start_node_orientation=orientation_start,
				# 						  goal_node_index=goal_node_index, goal_node_orientation=orientation_goal)
				# 	if not impassable_edge in self._impassable_edges:
				# 		self._impassable_edges.append(impassable_edge)
				# 		self._impassable_reason.append(nav_by_actionnet_result[1])

				self._fail_type[goal_action_type] += 1
				# print('failed case: ', failed_case)
				if failed_case >= self._fail_case_tolerance or node_path == path[-1]:
					return (node_path, self._fail_types)
		return (True, self._fail_types)
