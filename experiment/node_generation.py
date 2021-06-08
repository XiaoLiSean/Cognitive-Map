from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from dijkstar import Graph, find_path
from scipy import spatial
from distutils.util import strtobool
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import time
import copy
import argparse
import random
import logging
import os
import sys
sys.path.append('../')
# from Map import Topological_map
from lib.object_dynamics import *
from lib.params import *
from matplotlib.patches import Ellipse, Circle

n_clusters = 3


class Node_generator():
	def __init__(self, controller=None, node_radius=2.0, fieldOfView=120):
		self._grid_size = 0.25
		self._fieldOfView = FIELD_OF_VIEW
		if controller is None:
			self._controller = Controller(scene='FloorPlan30', gridSize=self._grid_size, fieldOfView=self._fieldOfView, visibilityDistance=VISBILITY_DISTANCE, agentMode='bot')
			# self._controller = Controller(scene='FloorPlan_Train9_3', gridSize=self._grid_size, fieldOfView=self._fieldOfView, visibilityDistance=node_radius, agentMode='bot')
		else:
			self._controller = controller
		self._event = self._controller.step(action='Pass')

		self._general_orientation = self._event.metadata['agent']['rotation']

		self._reachable =None
		self._reachable_list = []
		self._reachable_position = []
		self._objects = None
		self._objects_position = []
		self._obj_nearest_point = {}

		self._points = None
		self._reachable_x = []
		self._reachable_y = []
		self._general_y = None
		self._obj_x = []
		self._obj_y = []
		self._neighbor_x = []
		self._neighbor_y = []

		self._x_max = -10
		self._x_min = 10
		self._y_max = -10
		self._y_min = 10

		self._boundary_grid_size = self._grid_size / 4
		self._map_boundary = []
		self._smaller_grid_map_boundary = []
		self._map_boundary_x = []
		self._map_boundary_y = []
		self._neighbor_nodes = []

		self._neighboring_node_connected_subnode = {}

		self._tree = None
		self._tree_obj = None
		self._boundary_tree = None


		self._cluster_center = None
		self._cluster_center_point = []
		self._n_clusters = random.randint(6, 8)
		self._kmeans = KMeans(n_clusters=self._n_clusters)
		self._object_label = None

		self._node_index_list = []
		self._node_radius = VISBILITY_DISTANCE

		self._graph = None

		self._node_vs_visible_obj = {}
		self._label_vs_node = {}

		self._test_x = []
		self._test_y = []
		self._group_x = [[] for j in range(4)]
		self._group_y = [[] for j in range(4)]
		self._connected_subnodes = None
		self._neighbor_nodes_dis = None
		self._neighbor_nodes_facing = None
		self.Init_node_generator()

	def Init_node_generator(self):
		self.Update_event()
		self._get_reachable()
		self._build_tree()
		self._get_cluster_center()
		self._build_graph()

	def Get_node_from_position(self, positions):

		rand_reachable = self._reachable_list[0]
		self._node_index_list = []
		for position in positions:
			position_temp = [position[0], rand_reachable[1], position[1]]

			self._node_index_list.append(self._reachable_list.index(position_temp))

	def Get_all_objects_shuffle(self):
		for _ in range(5):
			self.Shuffle_scene()
			self._get_object()
			self._build_obj_tree()
			self._get_node_vs_obj()

	def Shuffle_scene(self):
		shuffle_scene_layout(controller=self._controller)

	def Get_node_list(self):
		return self._node_index_list

	def Get_neighbor_nodes(self):
		return self._neighbor_nodes

	def Get_connected_subnodes(self):
		return self._connected_subnodes
# ------------------------------------------------------------------------------
# Modified by Xiao
# ------------------------------------------------------------------------------
	def get_reachable_coordinate(self):
		self._event = self._controller.step(action='GetReachablePositions')
		return self._event.metadata['actionReturn']

	def is_reachable(self, pi, pj):
	    map = self.get_reachable_coordinate()
	    diff = (np.array(pj) - np.array(pi)) / self._grid_size
	    sign = np.sign(diff)
	    diff = np.abs(diff.astype(int))
	    current_pose = dict(x=pi[0], y=map[0]['y'], z=pi[1])
	    count = 0
	    for i in range(1, diff[0]+1):
	        current_pose['x'] += sign[0]*self._grid_size
	        if current_pose in map:
	            count += 1
	    for j in range(1, diff[1]+1):
	        current_pose['z'] += sign[1]*self._grid_size
	        if current_pose in map:
	            count += 1
	    if count == (diff[0] + diff[1]):
	        return True


	    current_pose = dict(x=pi[0], y=map[0]['y'], z=pi[1])
	    count = 0
	    for j in range(1, diff[1]+1):
	        current_pose['z'] += sign[1]*self._grid_size
	        if current_pose in map:
	            count += 1
	    for i in range(1, diff[0]+1):
	        current_pose['x'] += sign[0]*self._grid_size
	        if current_pose in map:
	            count += 1
	    if count == diff[0] + diff[1]:
	        return True

	    return False

	def Get_connected_orientaton_by_geometry(self):
		map = self.get_reachable_coordinate()
		self._connected_subnodes = []
		self._neighbor_nodes = []
		epsilon = 0.01*self._grid_size # used to count for numerical error
		for node_i_idx in range(len(self._node_index_list) - 1):
			node_i = map[self._node_index_list[node_i_idx]]
			node_i = np.array([node_i['x'], node_i['z']])
			for node_j_idx in range(node_i_idx+1, len(self._node_index_list)):
				node_j = map[self._node_index_list[node_j_idx]]
				node_j = np.array([node_j['x'], node_j['z']])
				diff = np.abs(node_j - node_i)

				is_edge = False

				if (diff[0] - FORWARD_GRID * self._grid_size) <= epsilon:
					if (diff[1] - ADJACENT_NODES_SHIFT_GRID * self._grid_size) <= epsilon:
						is_edge = self.is_reachable(node_i, node_j)
						if is_edge:
							self._neighbor_nodes.append([self._node_index_list[node_i_idx], self._node_index_list[node_j_idx]])
							if node_j[0] > node_i[0]:
								self._connected_subnodes.append([1,3]) # i to j in subnode 1 [90 deg], j to i in subnode 3 [270 deg]
							else:
								self._connected_subnodes.append([3,1]) # i to j in subnode 3 [270 deg], j to i in subnode 1 [90 deg]

				if (diff[1] - FORWARD_GRID * self._grid_size) <= epsilon:
					if (diff[0] - ADJACENT_NODES_SHIFT_GRID * self._grid_size) <= epsilon:
						is_edge = self.is_reachable(node_i, node_j)
						if is_edge:
							self._neighbor_nodes.append([self._node_index_list[node_i_idx], self._node_index_list[node_j_idx]])
							if node_j[1] > node_i[1]:
								self._connected_subnodes.append([0,2]) # i to j in subnode 0 [0 deg], j to i in subnode 2 [180 deg]
							else:
								self._connected_subnodes.append([2,0]) # i to j in subnode 2 [180 deg], j to i in subnode 0 [0 deg]

	def Teleport_agent(self, position, rotation=0):
		self.Update_event()
		if isinstance(position, dict):
			position_list = list(position.values())
		else:
			position_list = copy.deepcopy(position)
		self._event = self._controller.step(action='TeleportFull', x=position_list[1], y=self._general_y, z=position_list[0],
			rotation=dict(x=self._general_orientation['x'], y=rotation, z=self._general_orientation['z']))

	def Plot_map(self):
		map_boundary_x = []
		map_boundary_y = []

		# print('self._smaller_grid_map_boundary: ', len(self._smaller_grid_map_boundary))

		for point in self._smaller_grid_map_boundary:
			map_boundary_x.append(point[0])
			map_boundary_y.append(point[1])
		# for point in self._map_boundary:
		# 	map_boundary_x.append(point[0])
		# 	map_boundary_y.append(point[1])

		fig, ax = plt.subplots()

		plt.scatter(self._reachable_x, self._reachable_y, color='#1f77b4')
		plt.scatter(self._obj_x, self._obj_y, color='#ff7f0e')


		for i in range(len(self._node_index_list)):
			# if i == 3:
			# cir1 = Circle(xy = (self._reachable_position[self._node_index_list[i]][0], self._reachable_position[self._node_index_list[i]][1]), radius=self._node_radius, alpha=0.3)
			plt.scatter(self._reachable_position[self._node_index_list[i]][0], self._reachable_position[self._node_index_list[i]][1], color='#00FFFF')
			# ax.add_patch(cir1)
		# print('len(self._node_index_list): ', len(self._node_index_list))



		# for i in range(len(self._neighbor_nodes)):
		# 	x = []
		# 	x.append(self._reachable_position[self._neighbor_nodes[i][0]][0])
		# 	x.append(self._reachable_position[self._neighbor_nodes[i][1]][0])
		# 	y = []
		# 	y.append(self._reachable_position[self._neighbor_nodes[i][0]][1])
		# 	y.append(self._reachable_position[self._neighbor_nodes[i][1]][1])
		# 	plt.plot(x, y, color='#2ca02c')


		# direction = [[0.20, 0], [0, 0.20], [-0.20, 0], [0, -0.20]]
		# for i in range(len(self._connected_subnodes)):
		# 	for j in range(len(self._connected_subnodes[i])):
		# 		# if self._connected_subnodes[i][j]
		# 		x = []
		# 		x.append(self._reachable_position[self._neighbor_nodes[i][0]][0] + direction[self._connected_subnodes[i][j]][0])
		# 		x.append(self._reachable_position[self._neighbor_nodes[i][1]][0] + direction[self._connected_subnodes[i][j]][0])
		# 		y = []
		# 		y.append(self._reachable_position[self._neighbor_nodes[i][0]][1] + direction[self._connected_subnodes[i][j]][1])
		# 		y.append(self._reachable_position[self._neighbor_nodes[i][1]][1] + direction[self._connected_subnodes[i][j]][1])
		# 		plt.plot(x, y, color='#A52A2A')


		# plt.scatter(self._neighbor_x, self._neighbor_y, color='#2ca02c')
		plt.scatter(map_boundary_x, map_boundary_y, color='#2ca02c')
		plt.scatter(self._test_x, self._test_y, color='#9400D3')
		# plt.scatter(self._group_x[0], self._group_y[0], color='#9400D3')
		# plt.scatter(self._group_x[1], self._group_y[1], color='#A52A2A')
		# plt.scatter(self._group_x[2], self._group_y[2], color='#DEB887')
		# plt.scatter(self._group_x[3], self._group_y[3], color='#00FFFF')

		plt.axis('equal')
		plt.show()
		return

	def _build_graph(self):
		self._graph = Graph()
		for pos_index in range(len(self._reachable_position)):
			neighbor_indexes = self._tree.query_ball_point(self._reachable_position[pos_index], r=1.02 * self._grid_size)
			neighbor_indexes.remove(pos_index)
			for neighbor_index in neighbor_indexes:
				self._graph.add_edge(pos_index, neighbor_index, 1)
		return

	def _build_tree(self):
		self._points = list(zip(self._reachable_x, self._reachable_y))
		self._tree = spatial.KDTree(list(zip(self._reachable_x, self._reachable_y)))

		# self._tree_obj = spatial.KDTree(list(zip(self._obj_x, self._obj_y)))
		return

	def _build_obj_tree(self):
		self._tree_obj = spatial.KDTree(list(zip(self._obj_x, self._obj_y)))
		return

	def Update_event(self):
		self._event = self._controller.step(action='Pass')

	def _get_reachable(self):
		self._event = self._controller.step(action='GetReachablePositions')
		self._reachable = self._event.metadata['actionReturn']
		self._reachable_list = []
		self._reachable_position = []
		self._reachable_x = []
		self._reachable_y = []

		self._x_max = -10
		self._x_min = 10
		self._y_max = -10
		self._y_min = 10

		for point in self._reachable:
			self._reachable_list.append(list(point.values()))
			self._reachable_x.append(point['z'])

			if point['z'] > self._x_max:
				self._x_max = point['z']
			if point['z'] < self._x_min:
				self._x_min = point['z']

			if point['x'] > self._y_max:
				self._y_max = point['x']
			if point['x'] < self._y_min:
				self._y_min = point['x']

			self._reachable_y.append(point['x'])
			self._general_y = point['y']
			self._reachable_position.append([point['z'], point['x']])

		# print('self._reachable_list: ', len(self._reachable_list))
		# print('self._reachable_position: ', len(self._reachable_position))

		return

	def _get_object(self):
		self.Update_event()
		self._obj_x = []
		self._obj_y = []
		self._objects = self._event.metadata['objects']
		for obj in self._objects:
			self._obj_x.append(obj['position']['z'])
			self._obj_y.append(obj['position']['x'])
			self._objects_position.append([obj['position']['z'], obj['position']['x']])

		return

	def _get_cluster_center(self):
		self._kmeans.fit(self._reachable_position)
		label_pred = self._kmeans.labels_
		self._cluster_center = self._kmeans.cluster_centers_

		inertia = self._kmeans.inertia_
		for i, center_pos in enumerate(self._cluster_center):
			dis_index = 1
			while True:
				index = self._tree.query_ball_point(center_pos, r=dis_index * self._grid_size)
				if len(index) > 0:
					self._cluster_center_point.append(self._reachable_position[index[0]])
					break
				else:
					dis_index += 1

	def _assign_obj_cluster(self):
		for i, obj_pos in enumerate(self._objects_position):
			dis_index = 1
			while True:
				index = self._tree.query_ball_point(obj_pos, r=dis_index * self._grid_size)
				if len(index) > 0:
					self._obj_nearest_point[i] = self._reachable_position[index[0]]
					self._neighbor_x.append(self._obj_nearest_point[i][0])
					self._neighbor_y.append(self._obj_nearest_point[i][1])
					break
				else:
					dis_index += 1
		self._object_label = self._kmeans.predict(list(self._obj_nearest_point.values()))

	def _connect_by_node(self, current_node, starting, goal):
		node_list_temp = copy.deepcopy(current_node)
		result = find_path(self._graph, starting, goal)
		path = result.nodes
		# print('path: ', path)
		for path_point_index in path:
			add = True
			for node_index in node_list_temp:
				if np.linalg.norm(list(map(lambda x, y: x - y,
					self._reachable_position[path_point_index], self._reachable_position[node_index]))) < 1.5 * self._node_radius:
					add = False
					break
			if path_point_index == path[len(path) - 1]:
				add = True
			if add:
				node_list_temp.append(path_point_index)
		return node_list_temp

	def Build_node_map(self):
		# label_point = {}
		# for i in range(len(self._cluster_center_point)):
		# 	label_point[i] = []
		# for key, value in self._obj_nearest_point.items():
		# 	label_point[self._object_label[key]].append(self._reachable_position.index(value))
		# for i in range(len(self._cluster_center_point)):
		# 	label_point[i] = list(set(label_point[i]))
		# print('label_point:', label_point)
		label_point = self._label_vs_node
		self._node_index_list = []
		for center_index, center in enumerate(self._cluster_center_point):
			if not center_index in list(label_point.keys()):
				continue
			center_connected_point_indexes = label_point[center_index]
			center_point_index = self._reachable_position.index(center)
			# print('center_point_index: ', center_point_index)
			# print('center_connected_point_indexes: ', center_connected_point_indexes)
			for goal_index in center_connected_point_indexes:
				# self._node_index_list = self._connect_by_node(self._node_index_list, center_point_index, goal_index)
				self._node_index_list = self._connect_by_node(self._node_index_list, goal_index, goal_index)


		for center_index in range(len(self._cluster_center_point)):
			for goal_center_index in range(center_index + 1, len(self._cluster_center_point)):
				current_center_point_index = self._reachable_position.index(self._cluster_center_point[center_index])
				goal_center_point_index = self._reachable_position.index(self._cluster_center_point[goal_center_index])
				self._node_index_list = self._connect_by_node(self._node_index_list, current_center_point_index, goal_center_point_index)
		self._node_index_list = list(set(self._node_index_list))
		# print('self._node_index_list: ', self._node_index_list)
		return
