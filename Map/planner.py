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

class Planner():
	def __init__(self):
		self._subnode_plan = False
		self._orientations = [0, 90, 180, 270]
		self._graph = None
		self._grid_size = 0.25
		self._dij_graph = dij.Graph()

	def Set_env_from_topo_map(self, topo_map):
		self._grid_size = copy.deepcopy(topo_map._grid_size)
		self._graph = topo_map.Export_graph()
		self._dij_graph = dij.Graph()

	def Set_planning_method(self, using_subnode=False):
		self._subnode_plan = using_subnode

	def Get_subnode_dij_index(self, node_name):
		node_num, orientation = self.Get_node_index_orien(node_name)
		return node_num * len(self._orientations) + self._orientations.index(orientation)

	def Get_node_dij_index(self, node_name):
		node_num, orientation = self.Get_node_index_orien(node_name)
		return node_num

	def Get_subnode_index_from_dij_index(self, dij_index):
		if self._subnode_plan:
			node_num = int(dij_index / len(self._orientations))
			orientation = int(dij_index % len(self._orientations))
			node_name = self.Get_node_name(node_num=node_num, orientation=self._orientations[orientation])
			return node_name
		else:
			logging.error('Trying to get subnode index in node planning mode')
			return -1

	def Get_node_index_from_dij_index(self, dij_index):
		if self._subnode_plan:
			node_num = int(dij_index / len(self._orientations))
			orientation = int(dij_index % len(self._orientations))
			node_name = self.Get_node_name(node_num=node_num, orientation=self._orientations[orientation])
			logging.warnning('Trying to use get node index in subnode planning mode')
			return node_name
		else:
			return dij_index

	def Build_dij_graph(self):
		if self._graph is None:
			logging.error('Need to get graph from topo map')
			return

		if self._subnode_plan:
			self._build_subnode_dij_graph()
		else:
			self._build_node_dij_graph()

	def _build_subnode_dij_graph(self):

		for current_node_name, neighbor_nodes in self._graph.adj.items():

			current_node_dij_index = self.Get_subnode_dij_index(node_name=current_node_name)

			for neighbor_node_name, weight_dict in neighbor_nodes.items():

				neighbor_node_dij_index = self.Get_subnode_dij_index(node_name=neighbor_node_name)
				self._dij_graph.add_edge(current_node_dij_index, neighbor_node_dij_index, weight_dict['weight'])

		return

	def _build_node_dij_graph(self):

		for current_node_name, neighbor_nodes in self._graph.adj.items():

			current_node_dij_index = self.Get_node_dij_index(node_name=current_node_name)

			for neighbor_node_name, weight_dict in neighbor_nodes.items():

				neighbor_node_dij_index = self.Get_node_dij_index(node_name=neighbor_node_name)

				if current_node_dij_index == neighbor_node_dij_index:
					continue
				else:

					current_node_position, _, _ = self.Get_node_value_by_name(current_node_name)
					neighbor_node_position, _, _ = self.Get_node_value_by_name(neighbor_node_name)
					current_node_position = np.array(current_node_position)
					neighbor_node_position = np.array(neighbor_node_position)
					distance = np.sum(np.abs(np.round((current_node_position - neighbor_node_position) / self._grid_size)))

					if distance > 6:
						distance *= 2

					self._dij_graph.add_edge(current_node_dij_index, neighbor_node_dij_index, distance)
					# self._dij_graph.add_edge(neighbor_node_dij_index, current_node_dij_index, distance)

		return

	def Find_dij_path(self, current_node_index, current_orientation, goal_node_index, goal_orientation):
		if self._subnode_plan:
			return self.Find_subnode_dij_path(current_node_index, current_orientation, goal_node_index, goal_orientation)
		else:
			return self.Find_node_dij_path(current_node_index, current_orientation, goal_node_index, goal_orientation)

	def Find_subnode_dij_path(self, current_node_index, current_orientation, goal_node_index, goal_orientation):

		current_node_name = self.Get_node_name(node_num=current_node_index, orientation=current_orientation)
		current_dij_index = self.Get_subnode_dij_index(node_name=current_node_name)

		goal_node_name = self.Get_node_name(node_num=goal_node_index, orientation=goal_orientation)
		goal_dij_index = self.Get_subnode_dij_index(node_name=goal_node_name)

		try:
			result = dij.find_path(self._dij_graph, current_dij_index, goal_dij_index)
			path = result.nodes
		except:
			print('Planning error')
			path = []

		path_nodes_name = []
		for node_dij_index in path:
			path_nodes_name.append(self.Get_subnode_index_from_dij_index(node_dij_index))

		return path_nodes_name

	def Find_node_dij_path(self, current_node_index, current_orientation, goal_node_index, goal_orientation):

		current_node_name = self.Get_node_name(node_num=current_node_index, orientation=current_orientation)
		current_dij_index = self.Get_node_dij_index(node_name=current_node_name)

		goal_node_name = self.Get_node_name(node_num=goal_node_index, orientation=goal_orientation)
		goal_dij_index = self.Get_node_dij_index(node_name=goal_node_name)

		result = dij.find_path(self._dij_graph, current_dij_index, goal_dij_index)
		path = result.nodes
		# print('Find_node_dij_path: ', 'Find_node_dij_path')
		path_subnode = []
		path_subnode.append(current_node_name)
		# print('path_subnode: ', path_subnode)

		# print('path: ', path)

		for path_node in path:
			# print('path_node: ', path_node)
			path_subnode = self._local_subnode_plan(path_subnode=path_subnode, goal_node=path_node)
			# if path_node == 1:
			# 	break

		path_subnode = self._local_subnode_plan(path_subnode=path_subnode, goal_node=goal_node_index, goal_orientation=goal_orientation)

		# print('path_subnode: ', path_subnode)
		# path_nodes_name = []
		# for node_dij_index in path:
		# 	path_nodes_name.append(self.Get_subnode_index_from_dij_index(node_dij_index))

		return path_subnode

	def _local_subnode_plan(self, path_subnode, goal_node, goal_orientation=None):

		path_subnode_temp = copy.deepcopy(path_subnode)

		current_sunode_name = path_subnode[-1]
		current_node_num, orientation = self.Get_node_index_orien(current_sunode_name)
		moved_node_name = None

		if not current_node_num == goal_node:

			moving_orientations = self._find_optimal_orientation(current_node=current_node_num, goal_node=goal_node)
			# print('moving_orientations: ', moving_orientations)
			for moving_orientation in moving_orientations:

				attempting_current_node = self.Get_node_name(node_num=current_node_num, orientation=moving_orientation)
				attempting_goal_node = self.Get_node_name(node_num=goal_node, orientation=moving_orientation)

				if attempting_goal_node in list(self._graph.adj[attempting_current_node].keys()):
					# print('working moving_orientation: ', moving_orientation)
					# print('path_subnode_temp: ', path_subnode_temp)

					# print('attempting_current_node: ', attempting_current_node)
					# print()
					path_subnode_temp.extend(self._rotation_planner(current_node_name=current_sunode_name, goal_orientation=moving_orientation))
					# print('path_subnode_temp: ', path_subnode_temp)
					path_subnode_temp.append(attempting_goal_node)
					# print('path_subnode_temp: ', path_subnode_temp)
					moved_node_name = attempting_goal_node
					break
		else:
			moved_node_name = path_subnode[-1]

		if not goal_orientation is None:
			path_subnode_temp.extend(self._rotation_planner(current_node_name=moved_node_name, goal_orientation=goal_orientation))

		# print('path_subnode_temp: ', path_subnode_temp)
		return path_subnode_temp

	def _find_optimal_orientation(self, current_node, goal_node):

		rand_current_node_subnode_name = self.Get_node_name(node_num=current_node, orientation=0)
		rand_goal_node_subnode_name = self.Get_node_name(node_num=goal_node, orientation=0)

		current_node_position, _, _ = self.Get_node_value_by_name(node_name=rand_current_node_subnode_name)
		goal_node_position, _, _ = self.Get_node_value_by_name(node_name=rand_goal_node_subnode_name)

		error_position = list(map(lambda x, y: x - y, goal_node_position, current_node_position))
		moving_orientation = np.arctan2(error_position[0], error_position[2]) * 180 / np.pi

		moving_orientation = int(np.round(moving_orientation / 90) * 90)

		if moving_orientation < 0:
			moving_orientation += 360

		rand_90_degree = random.choice([90, -90])

		return [moving_orientation, self._wrap_to_360(moving_orientation + 180),
		self._wrap_to_360(moving_orientation + rand_90_degree), self._wrap_to_360(moving_orientation - rand_90_degree)]


	def _rotation_planner(self, current_node_name, goal_orientation):

		rotation_subnode_path = []
		node_num, orientation_current = self.Get_node_index_orien(current_node_name)

		if orientation_current == goal_orientation:
			return rotation_subnode_path

		goal_subnode_name = self.Get_node_name(node_num=node_num, orientation=goal_orientation)

		if goal_subnode_name in list(self._graph.adj[current_node_name].keys()):

			rotation_subnode_path.append(goal_subnode_name)

		else:

			rand_90_rot = random.choice([90, -90])
			intermediate_orientation = self._wrap_to_360(orientation_current + rand_90_rot)
			intermediate_subnode_name = self.Get_node_name(node_num=node_num, orientation=intermediate_orientation)

			if not intermediate_subnode_name in list(self._graph.adj[current_node_name].keys()):

				rand_90_rot = -rand_90_rot
				intermediate_orientation = self._wrap_to_360(orientation_current + rand_90_rot)
				intermediate_subnode_name = self.Get_node_name(node_num=node_num, orientation=intermediate_orientation)

			if not intermediate_subnode_name in list(self._graph.adj[current_node_name].keys()):
				logging.error('Graph not complete when doing rotation planning')
				return None

			rotation_subnode_path.append(intermediate_subnode_name)

			if goal_subnode_name in list(self._graph.adj[intermediate_subnode_name].keys()):

				rotation_subnode_path.append(goal_subnode_name)

			else:

				logging.error('Graph not complete when doing rotation planning')
				return None

		return rotation_subnode_path

	def _wrap_to_360(self, degree):
		while degree >= 360:
			degree -= 360
		while degree < 0:
			degree += 360
		return int(degree)

	def Get_node_value_by_name(self, node_name):
		values = self._graph.nodes[node_name]
		return values['position'], values['image'], values['scene_graph']

	def Get_node_name(self, node_num, orientation):
		return 'node_' + str(node_num) + '_degree_' + str(orientation)

	def Get_node_index_orien(self, node_name):
		name_split = node_name.split('_')
		return int(name_split[1]), int(name_split[3])
