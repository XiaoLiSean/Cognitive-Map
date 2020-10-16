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
from matplotlib.patches import Ellipse, Circle

n_clusters = 3


class Node_generator():
	def __init__(self, node_radius=1.5):
		self._grid_size = 0.25
		self._controller = Controller(scene='FloorPlan1', gridSize=self._grid_size, fieldOfView=120, visibilityDistance=node_radius, agentMode='bot')
		self._event = self._controller.step(action='Pass')

		self._general_orientation = self._event.metadata['agent']['rotation']

		self._reachable =None
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

		self._tree = None
		self._tree_obj = None

		self._cluster_center = None
		self._cluster_center_point = []
		self._n_clusters = random.randint(6, 8)
		self._kmeans = KMeans(n_clusters=self._n_clusters)
		self._object_label = None

		self._node_index_list = []
		self._node_radius = node_radius

		self._graph = None

		self._node_vs_visible_obj = {}
		self._label_vs_node = {}

		self.Update_event()
		self._get_object()
		self._get_reachable()
		self._build_tree()
		self._get_cluster_center()
		self._build_graph()
		self._assign_obj_cluster()
		self._get_node_vs_obj()
		self.Build_node_map()


	def _get_node_vs_obj(self):
		pts_object_cover = {}

		for pos_index, pos in enumerate(self._reachable_position):
			index = self._tree_obj.query_ball_point(pos, r=self._node_radius)
			if len(index) > 0:
				pts_object_cover[pos_index] = index

		for pos_index, object_cover in pts_object_cover.items():
			self.Teleport_agent(position=self._reachable_position[pos_index])
			self._node_vs_visible_obj[pos_index] = []
			for _ in range(4):
				self._objects = self._event.metadata['objects']
				visible_obj_temp = [self._objects.index(obj) for obj in self._objects if obj['visible']]
				self._node_vs_visible_obj[pos_index].extend(visible_obj_temp)
				self._event = self._controller.step(action='RotateRight', degrees=90)
			self._node_vs_visible_obj[pos_index] = list(set(self._node_vs_visible_obj[pos_index]))

		pos_vs_vis_obj_num = {}
		vis_obj_vs_node = {}
		best_node_obj_pair = {}

		for node_index, objs in self._node_vs_visible_obj.items():
			pos_vs_vis_obj_num[node_index] = len(objs)
			for obj_index in objs:
				if obj_index in list(vis_obj_vs_node.keys()):
					vis_obj_vs_node[obj_index].append(node_index)
				else:
					vis_obj_vs_node[obj_index] = [node_index]

		for obj, nodes in vis_obj_vs_node.items():
			best_node_index = None
			best_node_vis_obj = -1
			for node_index in nodes:
				if pos_vs_vis_obj_num[node_index] > best_node_vis_obj:
					best_node_index = node_index
					best_node_vis_obj = pos_vs_vis_obj_num[node_index]
			best_node_obj_pair[obj] = best_node_index

		node_list = list(set(list(best_node_obj_pair.values())))
		node_position = []
		for node_index in node_list:
			node_position.append(self._reachable_position[node_index])
		node_label = self._kmeans.predict(node_position)
		for i in range(len(node_list)):
			if not node_label[i] in list(self._label_vs_node.keys()):
				self._label_vs_node[node_label[i]] = [node_list[i]]
			else:
				self._label_vs_node[node_label[i]].append(node_list[i])
		return

	def Teleport_agent(self, position, rotation=0):
		self.Update_event()
		if isinstance(position, dict):
			position_list = list(position.values())
		else:
			position_list = copy.deepcopy(position)
		self._event = self._controller.step(action='TeleportFull', x=position_list[1], y=self._general_y, z=position_list[0],
			rotation=dict(x=self._general_orientation['x'], y=rotation, z=self._general_orientation['z']))

	def Plot_map(self):
		fig, ax = plt.subplots()
		for i in range(len(self._node_index_list)):
			cir1 = Circle(xy = (self._reachable_position[self._node_index_list[i]][0], self._reachable_position[self._node_index_list[i]][1]), radius=self._node_radius, alpha=0.3)
			ax.add_patch(cir1)
		print('len(self._node_index_list): ', len(self._node_index_list))
		plt.scatter(self._reachable_x, self._reachable_y, color='#1f77b4')
		plt.scatter(self._obj_x, self._obj_y, color='#ff7f0e')
		plt.scatter(self._neighbor_x, self._neighbor_y, color='#2ca02c')
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
		points_y = []
		points_x = []
		for point in self._reachable:
			points_x.append(point['z'])
			points_y.append(point['x'])

		self._points = list(zip(self._reachable_x, self._reachable_y))
		self._tree = spatial.KDTree(list(zip(self._reachable_x, self._reachable_y)))

		self._tree_obj = spatial.KDTree(list(zip(self._obj_x, self._obj_y)))
		return

	def Update_event(self):
		self._event = self._controller.step(action='Pass')

	def _get_reachable(self):
		self._event = self._controller.step(action='GetReachablePositions')
		self._reachable = self._event.metadata['actionReturn']
		for point in self._reachable:
			self._reachable_x.append(point['z'])
			self._reachable_y.append(point['x'])
			self._general_y = point['y']
			self._reachable_position.append([point['z'], point['x']])

		return

	def _get_object(self):
		self.Update_event()
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
				self._node_index_list = self._connect_by_node(self._node_index_list, center_point_index, goal_index)
		print('_node_index_list: ', self._node_index_list)
		print('len(node_index_list): ', len(self._node_index_list))

		object_cover = {}
		for node_index in self._node_index_list:
			index = self._tree_obj.query_ball_point(self._reachable_position[node_index], r=self._node_radius)
			object_cover[node_index] = index
		object_cover_list = list(object_cover.values())
		repeated_cover_node = []
		for index, object_covered in enumerate(object_cover_list):
			if object_cover_list.count(object_covered) > 1:
				repeated_cover_node_temp = []
				for i in range(len(object_cover_list)):
					if object_cover_list[i] == object_covered:
						repeated_cover_node_temp.append(i)
				repeated_cover_node.append(repeated_cover_node_temp)
		# repeated_cover_node = list(set(repeated_cover_node))
		repeated_cover_node_no_dup = []
		node_remove = []
		for i in range(len(repeated_cover_node)):
			if not repeated_cover_node[i] in repeated_cover_node_no_dup:
				repeated_cover_node_no_dup.append(repeated_cover_node[i])
				node_remove.append(self._node_index_list[repeated_cover_node[i][0]])
		# print('repeated_cover_node: ', repeated_cover_node)
		print('repeated_cover_node_no_dup: ', repeated_cover_node_no_dup)

		for i in range(len(node_remove)):
			# print(repeated_cover_node_no_dup[i][0])
			self._node_index_list.remove(node_remove[i])



		print('_node_index_list: ', self._node_index_list)
		print('len(node_index_list): ', len(self._node_index_list))

		# for center_index in range(len(self._cluster_center_point)):
		# 	for goal_center_index in range(center_index + 1, len(self._cluster_center_point)):
		# 		current_center_point_index = self._reachable_position.index(self._cluster_center_point[center_index])
		# 		goal_center_point_index = self._reachable_position.index(self._cluster_center_point[goal_center_index])
		# 		self._node_index_list = self._connect_by_node(self._node_index_list, current_center_point_index, goal_center_point_index)
		# self._node_index_list = list(set(self._node_index_list))
		# print('_node_index_list: ', self._node_index_list)
		# print('len(node_index_list): ', len(self._node_index_list))
		return

if __name__ == '__main__':
	node_generator = Node_generator()
	node_generator.Plot_map()
