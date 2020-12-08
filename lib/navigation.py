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
sys.path.append('./Network')
sys.path.append('./experiment')
from Map import *
from lib.params import *


class Navigation():
	def __init__(self, scene_type, scene_num, save_directory, AI2THOR, server=None, comfirmed=None):
		self.Robot = Robot(scene_type=scene_type, scene_num=scene_num, save_directory=save_directory, AI2THOR=AI2THOR, server=server, comfirmed=comfirmed)
		self.node_generator = Node_generator(controller=self.Robot._AI2THOR_controller._controller)
		self.topo_map = Topological_map(controller=self.Robot._AI2THOR_controller._controller, node_index_list=None, neighbor_nodes_pair=None)
		self.planner = Planner()
		self._fail_case_tolerance = 3
		self._valid_action_type = VALID_ACTION_TYPE
		self._fail_type = {'translation': 0, 'rotation': 0}
		self._action_case_num = 0
		self._action_success_num = 0
		self._nav_test_case_num = 0
		self._nav_success_case_num = 0
		self._node_list = None

	def nav_test(self):
		for scene_type in range(3, 4):
			for scene_num in range(27, 28):
				self.Switch_scene(scene_type=scene_type, scene_num=scene_num)
				# self.Plotter.show_map(show_nodes=False)
				for start in range(len(self._node_list)):
					for goal in range(len(self._node_list)):
						start_orien = random.choice([0, 90, 180, 270])
						goal_orien = random.choice([0, 90, 180, 270])
						# print('start: ', start)
						# print('goal: ', goal)
						self.Closed_loop_nav(current_node_index=start, current_orientation=start_orien, goal_node_index=goal, goal_orientation=goal_orien)
		self.Write_csv()

	def Write_csv(self):
		nav_test = open('nav_test.csv','a')
		nav_test_writer = csv.writer(nav_test)
		nav_test_writer.writerow([self._nav_test_case_num, self._nav_success_case_num,
								  self._action_case_num, self._action_success_num, self._fail_type['translation'], self._fail_type['rotation']])


	def Update_node_generator(self):
		self.node_generator.Init_node_generator()
		self._node_list = NODES[self.Robot._AI2THOR_controller.Get_scene_name()]
		print('self._node_list: ', self._node_list)
		self.node_generator.Get_node_from_position(self._node_list)
		self.node_generator.Get_connected_orientaton_by_geometry()
		# index of subnodes in reachable points
		self._node_pair_list = self.node_generator.Get_neighbor_nodes()
		# corresponding to _node_pair_list
		self._subnodes = self.node_generator.Get_connected_subnodes()

		print('self._node_pair_list: ', len(self._node_pair_list))

	def Update_topo_map_env(self):
		self.topo_map.Set_env_from_Robot(Robot=self.Robot)
		self.topo_map.Update_topo_map(node_index_list=self.node_generator.Get_node_list(), node_pair_list=self._node_pair_list, connected_subnodes=self._subnodes)
		return self.topo_map

	def Update_planner_env(self):
		self.planner.Set_env_from_topo_map(topo_map=self.topo_map)
		self.planner.Build_dij_graph()

	def Switch_scene(self, scene_type, scene_num):

		print('DOOR_NODE[self.Robot._AI2THOR_controller.Get_scene_name()]: ', DOOR_NODE[self.Robot._AI2THOR_controller.Get_scene_name()])
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

			if self.Robot.Navigation_stop(image_goal=node_image, image_current=self.Robot._AI2THOR_controller.Get_frame(), goal_pose=None, hardcode=False):
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
			current_orientation = self.Robot._AI2THOR_controller.Get_agent_current_orientation()
			print('current_node_index: ', current_node_index)
			print('current_orientation: ', current_orientation)
		path = self.planner.Find_dij_path(current_node_index=current_node_index, current_orientation=current_orientation,
										  goal_node_index=goal_node_index, goal_orientation=goal_orientation)
		print('path: ', path)

		nav_result = self.Navigate_by_path(path=path)

		self._nav_test_case_num += 1
		if nav_result is True:
			self._nav_success_case_num += 1

		return

		print('nav_result: ', nav_result)
		# exit()
		while isinstance(nav_result, str):

			match = False
			while not match:
				success, _ = self.Robot._AI2THOR_controller.Random_move_w_weight(all_actions=False, weight=[0.25, 0.25, 0.25, 0.25])
				match, node_matched_name = self.Node_localize_BF(starting_node_name=nav_result)
				print('node_matched_name: ', node_matched_name)

			searching_node_index, searching_node_orientation = self.topo_map.Get_node_index_orien(node_matched_name)

			if node_matched_name in path:
				path = path[path.index(node_matched_name):]
				print('path: ', path)
			else:
				path = self.planner.Find_dij_path(current_node_index=searching_node_index, current_orientation=searching_node_orientation,
							  goal_node_index=goal_node_index, goal_orientation=goal_orientation)
				print('repath: ', path)

			nav_result = self.Navigate_by_path(path=path)

	def Navigate_by_path(self, path):

		init_position = self.topo_map.Get_node_value_dict_by_name(path[0])['position']
		_, orientation = self.topo_map.Get_node_index_orien(path[0])

		self.Robot._AI2THOR_controller.Teleport_agent(init_position, position_localize=True)
		self.Robot._AI2THOR_controller.Rotate_to_degree(orientation)

		rotation_standard = list(self.Robot.Get_robot_orientation().values())

		failed_case = 0

		for node_path_num, node_path in enumerate(path):

			node_value = self.topo_map.Get_node_value_dict_by_name(node_path)
			_, orientation = self.topo_map.Get_node_index_orien(node_path)

			goal_frame = node_value['image']
			goal_pose = {'position': node_value['position'], 'rotation': copy.deepcopy(rotation_standard)}
			goal_pose['rotation'][1] = orientation
			goal_action_type = 'translation'
			if node_path_num > 0:
				_, orientation_pre = self.topo_map.Get_node_index_orien(path[node_path_num - 1])
				if not orientation_pre == orientation:
					goal_action_type = 'rotation'

			if goal_action_type == 'rotation':
				rotation_degree = int(orientation - orientation_pre)
				if rotation_degree == 270:
					rotation_degree = -90
				elif rotation_degree == -270:
					rotation_degree = 90
				print('rotation hard code', rotation_degree)
			else:
				rotation_degree = None

			self._action_case_num += 1
			if self.Robot.Navigate_by_ActionNet(image_goal=goal_frame, goal_pose=goal_pose, max_steps=self.Robot._Navigation_max_try, rotation_degree=rotation_degree):
				failed_case = 0
				
				self._action_success_num += 1
				print('reach node ', node_path)
				# time.sleep(1)
			else:
				failed_case += 1
				
				self._fail_type[goal_action_type] += 1
				print('failed case: ', failed_case)
				if failed_case >= self._fail_case_tolerance or node_path == path[-1]:
					return node_path
		return True
