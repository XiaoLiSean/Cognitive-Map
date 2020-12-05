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
from matplotlib.patches import Ellipse, Circle

n_clusters = 3


class Node_generator():
	def __init__(self, controller=None, node_radius=2.0, fieldOfView=120):
		self._grid_size = 0.25
		self._fieldOfView = fieldOfView
		if controller is None:
			self._controller = Controller(scene='FloorPlan30', gridSize=self._grid_size, fieldOfView=self._fieldOfView, visibilityDistance=node_radius, agentMode='bot')
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
		self._node_radius = node_radius

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

		# self.Shuffle_scene()
		# self.Update_event()

		# self._get_reachable()
		# self._build_tree()
		# self._get_cluster_center()
		# self._build_graph()
		self.Init_node_generator()
		# self._assign_obj_cluster()

		# self._get_object()
		# self._build_obj_tree()

		# self._get_node_vs_obj()
		# self.Build_node_map()

		# self._node_index_list = [0, 1, 4, 5, 7, 8, 9, 10, 11, 17, 21, 23, 32, 44, 47, 48, 52, 56, 58, 59, 81, 83, 90]

		# self.Get_navigatable_node_pair()

		# self.Get_connected_orientaton_by_overlap_scene()

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

	def Get_connected_orientaton_by_overlap_scene(self):
		print('')
		self.Get_boundary()
		orientations = [[-60, 60], [30, 150], [120, -120], [-150, -30]]

		right = list(range(-60, 61))
		upper = list(range(30, 151))
		left = list(range(120, 181)) + list(range(-180, -119))
		lower = list(range(-150, -31))

		orientations = [right, upper, left, lower]
		subnode_visible_boundary = [[[] for i in range(len(orientations))] for i in range(len(self._node_index_list))]

		beam_step_size = self._grid_size / 5

		for node_pos_index in self._node_index_list:
			node_index = self._node_index_list.index(node_pos_index)
			# if not node_pos_index == 1:# and not node_pos_index == 4 and not node_pos_index == 5:
			# 	continue
			# print('node_pos_index: ', node_pos_index)

			for orientation_index in range(len(orientations)):

				# if not orientation_index == 0:
				# 	continue

				for beam_angle in orientations[orientation_index]:

					beam_path = self._reachable_position[node_pos_index]

					while beam_path[0] < self._x_max + self._grid_size and beam_path[0] > self._x_min - self._grid_size and \
						beam_path[1] < self._y_max + self._grid_size and beam_path[1] > self._y_min - self._grid_size:

						beam_step = [beam_step_size * np.cos(beam_angle / 180 * np.pi), beam_step_size * np.sin(beam_angle / 180 * np.pi)]
						beam_path = list(map(lambda x, y: x + y, beam_step, beam_path))
						# self._test_x.append(beam_path[0])
						# self._test_y.append(beam_path[1])
						beam_boundary_indexes = self._boundary_tree.query_ball_point(beam_path, r=0.55 * self._boundary_grid_size)
						if len(beam_boundary_indexes) > 0:
							for beam_boundary_index in beam_boundary_indexes:
								if not beam_boundary_index in subnode_visible_boundary[node_index][orientation_index]:

									subnode_visible_boundary[node_index][orientation_index].append(beam_boundary_index)
									# self._test_x.append(self._map_boundary[beam_boundary_index][0])
									# self._test_y.append(self._map_boundary[beam_boundary_index][1])
							break
				subnode_visible_boundary[node_index][orientation_index] = list(set(subnode_visible_boundary[node_index][orientation_index]))
				# print('subnode_visible_boundary: ', node_pos_index, orientation_index, len(subnode_visible_boundary[node_index][orientation_index]))

		self._connected_subnodes = []
		self._neighbor_nodes = []
		# print('self._neighbor_nodes: ', self._neighbor_nodes)
		for orientation_index in range(len(orientations)):

			for node_index in range(len(self._node_index_list)):

				for other_node_index in range(node_index, len(self._node_index_list)):

					boundary_pts_in_common_num = 0

					if node_index == other_node_index:
						continue

					for point in subnode_visible_boundary[other_node_index][orientation_index]:

						if point in subnode_visible_boundary[node_index][orientation_index]:
							boundary_pts_in_common_num += 1

					if len(subnode_visible_boundary[node_index][orientation_index]) == 0 or len(subnode_visible_boundary[other_node_index][orientation_index]) == 0:
						continue
					percent_common_view_first = boundary_pts_in_common_num / len(subnode_visible_boundary[node_index][orientation_index])
					percent_common_view_second = boundary_pts_in_common_num / len(subnode_visible_boundary[other_node_index][orientation_index])

					if percent_common_view_first > 0.7 or percent_common_view_second > 0.7:

						node_pair = [self._node_index_list[node_index], self._node_index_list[other_node_index]]
						reverse_node_pair = [self._node_index_list[other_node_index], self._node_index_list[node_index]]

						if not node_pair in self._neighbor_nodes and not reverse_node_pair in self._neighbor_nodes:

							self._neighbor_nodes.append([self._node_index_list[node_index], self._node_index_list[other_node_index]])
							self._connected_subnodes.append([orientation_index])

						else:
							if node_pair in self._neighbor_nodes:

								if not orientation_index in self._connected_subnodes[self._neighbor_nodes.index(node_pair)]:
									self._connected_subnodes[self._neighbor_nodes.index(node_pair)].append(orientation_index)
							elif reverse_node_pair in self._neighbor_nodes:
								if not orientation_index in self._connected_subnodes[self._neighbor_nodes.index(reverse_node_pair)]:
									self._connected_subnodes[self._neighbor_nodes.index(reverse_node_pair)].append(orientation_index)
		print('self._node_index_list: ', self._node_index_list)
		print('self._neighbor_nodes: ', self._neighbor_nodes)
		print('self._connected_subnodes: ', self._connected_subnodes)
		return self._connected_subnodes

	def Get_boundary(self):
		all_neighbors_relative = [[0, self._grid_size], [0, -self._grid_size], [self._grid_size, 0], [-self._grid_size, 0]]
		self._map_boundary = []
		self._map_boundary_x = []
		self._map_boundary_y = []

		for index in range(len(self._reachable_position)):
			all_neighbor_indexes = self._tree.query_ball_point(self._reachable_position[index], r=1.02 * self._grid_size)

			if len(all_neighbor_indexes) < 5:
				# self._map_boundary.append(self._reachable_position[index])
				# self._map_boundary_x.append(self._reachable_position[index][0])
				# self._map_boundary_y.append(self._reachable_position[index][1])
				for neighbor_num in range(len(all_neighbors_relative)):
					neighbor_position = list(map(lambda x, y: x + y, all_neighbors_relative[neighbor_num], self._reachable_position[index]))
					neighbor_index = self._tree.query_ball_point(neighbor_position, r=0.1 * self._grid_size)
					if len(neighbor_index) == 0:
						boundary_point_position = [round(self._reachable_position[index][0], 2) + all_neighbors_relative[neighbor_num][0],
							round(self._reachable_position[index][1], 2) + all_neighbors_relative[neighbor_num][1]]
						if not boundary_point_position in self._map_boundary:
							self._map_boundary.append(boundary_point_position)
							self._map_boundary_x.append(boundary_point_position[0])
							self._map_boundary_y.append(boundary_point_position[1])


		self._boundary_tree = spatial.KDTree(list(zip(self._map_boundary_x, self._map_boundary_y)))

		for boundary_point_position in self._map_boundary:
			neighbot_boundary_index = self._boundary_tree.query_ball_point(boundary_point_position, r=1.02 * self._grid_size)
			diagonal_boundary_indexes_temp = self._boundary_tree.query_ball_point(boundary_point_position, r=1.02 *np.sqrt(2) * self._grid_size)
			diagonal_boundary_indexes = [index for index in diagonal_boundary_indexes_temp if index not in neighbot_boundary_index]
			for diagonal_boundary_index in diagonal_boundary_indexes:
				error = list(map(lambda x, y: x - y, self._map_boundary[diagonal_boundary_index], boundary_point_position))
				xy_errors = [[error[0], 0], [0, error[1]]]
				for xy_error in xy_errors:
					boundary_try_pt = list(map(lambda x, y: x + y, xy_error, boundary_point_position))
					potential_boundary_index = self._boundary_tree.query_ball_point(boundary_try_pt, r=0.1 * self._grid_size)
					potential_reachable_index = self._tree.query_ball_point(boundary_try_pt, r=0.1 * self._grid_size)
					if len(potential_boundary_index) > 0 or len(potential_reachable_index) > 0:
						continue
					else:
						potential_boundary_pt_neighbor_reachable_index = self._tree.query_ball_point(boundary_try_pt, r=1.02 *np.sqrt(2) * self._grid_size)
						if len(potential_boundary_pt_neighbor_reachable_index) == 0 or boundary_try_pt in self._map_boundary:
							continue
						else:
							self._map_boundary.append(boundary_try_pt)
							self._map_boundary_x.append(boundary_try_pt[0])
							self._map_boundary_y.append(boundary_try_pt[1])

		self._smaller_grid_map_boundary = []
		for boundary_point_position in self._map_boundary:

			self._smaller_grid_map_boundary.append(boundary_point_position)
			neighbot_boundary_indexes = self._boundary_tree.query_ball_point(boundary_point_position, r=1.02 * self._grid_size)
			node_num_to_add = self._grid_size / self._boundary_grid_size - 1

			for neighbor_boundary_index in neighbot_boundary_indexes:

				goal_direction_errors = list(map(lambda x, y: x - y, self._map_boundary[neighbor_boundary_index], boundary_point_position))
				goal_direction = []

				for goal_direction_error in goal_direction_errors:
					if goal_direction_error > 0:
						goal_direction.append(self._boundary_grid_size)
					elif goal_direction_error < 0:
						goal_direction.append(-self._boundary_grid_size)
					else:
						goal_direction.append(0)

				for adding_node_index in range(int(node_num_to_add)):

					adding_step = [i * (adding_node_index + 1) for i in goal_direction]
					adding_smaller_boundary_point = list(map(lambda x, y: x + y, boundary_point_position, adding_step))

					if not adding_smaller_boundary_point in self._smaller_grid_map_boundary:
						# print('adding_smaller_boundary_point: ', adding_smaller_boundary_point)
						self._smaller_grid_map_boundary.append(adding_smaller_boundary_point)
						self._map_boundary_x.append(adding_smaller_boundary_point[0])
						self._map_boundary_y.append(adding_smaller_boundary_point[1])

		self._map_boundary_x = []
		self._map_boundary_y = []
		for i in range(len(self._smaller_grid_map_boundary)):
			self._map_boundary_x.append(self._smaller_grid_map_boundary[i][0])
			self._map_boundary_y.append(self._smaller_grid_map_boundary[i][1])

		self._boundary_tree = spatial.KDTree(list(zip(self._map_boundary_x, self._map_boundary_y)))
		self._map_boundary = copy.deepcopy(self._smaller_grid_map_boundary)

		# print(self._boundary_tree.query_ball_point(self._map_boundary[0], r=0.8 * self._grid_size))

	def Get_navigatable_node_pair(self):
		for node_index in range(len(self._node_index_list)):
			for another_node_index in range(node_index + 1, len(self._node_index_list)):
				delta_position = list(map(lambda x, y: x - y, self._reachable_position[self._node_index_list[node_index]],
					self._reachable_position[self._node_index_list[another_node_index]]))
				if np.linalg.norm(delta_position) < 2 * self._node_radius:
					dx, dy = delta_position
					steps = int(max(np.abs(dx) / 0.05, np.abs(dy) / 0.05))
					add = True
					for i in range(steps):
						# print()
						searching_point = list(map(lambda x, y: x + y, self._reachable_position[self._node_index_list[another_node_index]], [dx * i / steps, dy * i / steps]))
						ineighbor_indexes = self._tree.query_ball_point(searching_point, r=0.8 * self._grid_size)
						# if self._node_index_list[node_index] == 212 and self._node_index_list[another_node_index] == 244 or self._node_index_list[node_index] == 244 and self._node_index_list[another_node_index] == 212:
						# if self._node_index_list[node_index] == 212 or self._node_index_list[another_node_index] == 212:
						# 	print('self._node_index_list[node_index]: ', self._node_index_list[node_index])
						# 	print('self._node_index_list[another_node_index]: ', self._node_index_list[another_node_index])
							# print('searching_point: ', searching_point)
							# print('ineighbor_indexes: ', ineighbor_indexes)
						if len(ineighbor_indexes) == 0:
							add = False
							break
					if add:
						self._neighbor_nodes.append([self._node_index_list[node_index], self._node_index_list[another_node_index]])
		print('len(self._neighbor_nodes): ', len(self._neighbor_nodes))
		print('len(self._neighbor_nodes): ', self._neighbor_nodes)
		return

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

		node_percent_common = {}
		for node_index, pos_index in enumerate(node_list):
			for second_node_index, second_pos_index in enumerate(node_list):

				num_obj_common = 0
				for obj_index in self._node_vs_visible_obj[pos_index]:
					if obj_index in self._node_vs_visible_obj[second_pos_index]:
						num_obj_common += 1
				if node_index == second_node_index:
					num_obj_common = - len(self._node_vs_visible_obj[pos_index])
				if not pos_index in list(node_percent_common.keys()):
					node_percent_common[pos_index] = [num_obj_common / len(self._node_vs_visible_obj[pos_index])]
				else:
					node_percent_common[pos_index].append(num_obj_common / len(self._node_vs_visible_obj[pos_index]))

		# print('node_percent_common: ', node_percent_common)
		to_remove_node_pos_index = []
		for pos_index, percents in node_percent_common.items():
			# print('to_remove_node_pos_index: ', to_remove_node_pos_index)
			for node_index, percent in enumerate(percents):
				if percent > 0.9 and not node_list[node_index] in to_remove_node_pos_index:
					# print('percent > 0.8')
					to_remove_node_pos_index.append(pos_index)
					break

		# print('node_list: ', node_list)
		# print('to_remove_node_pos_index: ', to_remove_node_pos_index)

		for pos_index in to_remove_node_pos_index:
			node_list.remove(pos_index)

		# print('node_list: ', node_list)

		node_position = []
		for node_index in node_list:
			node_position.append(self._reachable_position[node_index])

		print('node_position: ', node_position)
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
		map_boundary_x = []
		map_boundary_y = []

		print('self._smaller_grid_map_boundary: ', len(self._smaller_grid_map_boundary))

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
		print('len(self._node_index_list): ', len(self._node_index_list))



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


		# object_cover = {}
		# for node_index in self._node_index_list:
		# 	index = self._tree_obj.query_ball_point(self._reachable_position[node_index], r=self._node_radius)
		# 	object_cover[node_index] = index
		# object_cover_list = list(object_cover.values())

		# print('self._label_vs_node: ', self._label_vs_node)
		# print('object_cover_list: ', object_cover_list)

		# repeated_cover_node = []
		# for index, object_covered in enumerate(object_cover_list):
		# 	if object_cover_list.count(object_covered) > 1:
		# 		repeated_cover_node_temp = []
		# 		for i in range(len(object_cover_list)):
		# 			if object_cover_list[i] == object_covered:
		# 				repeated_cover_node_temp.append(i)
		# 		repeated_cover_node.append(repeated_cover_node_temp)
		# # repeated_cover_node = list(set(repeated_cover_node))

		# print('repeated_cover_node: ', repeated_cover_node)
		# repeated_cover_node_no_dup = []
		# node_remove = []
		# for i in range(len(repeated_cover_node)):
		# 	if not repeated_cover_node[i] in repeated_cover_node_no_dup:
		# 		repeated_cover_node_no_dup.append(repeated_cover_node[i])
		# 		node_remove.append(self._node_index_list[repeated_cover_node[i][0]])
		# # print('repeated_cover_node: ', repeated_cover_node)
		# # print('repeated_cover_node_no_dup: ', repeated_cover_node_no_dup)

		# for i in range(len(node_remove)):
		# 	# print(repeated_cover_node_no_dup[i][0])
		# 	self._node_index_list.remove(node_remove[i])



		for center_index in range(len(self._cluster_center_point)):
			for goal_center_index in range(center_index + 1, len(self._cluster_center_point)):
				current_center_point_index = self._reachable_position.index(self._cluster_center_point[center_index])
				goal_center_point_index = self._reachable_position.index(self._cluster_center_point[goal_center_index])
				self._node_index_list = self._connect_by_node(self._node_index_list, current_center_point_index, goal_center_point_index)
		self._node_index_list = list(set(self._node_index_list))
		print('self._node_index_list: ', self._node_index_list)
		return

if __name__ == '__main__':
	pass
	# test = test_func
	# test()
	# exit()


	# node_generator = Node_generator(controller=Agent_action._controller)
	# node_pair_list = node_generator.Get_neighbor_nodes()
	# # subnodes = node_generator.Get_connected_subnodes()

	# map = Topological_map(controller=node_generator._controller,node_index_list=node_generator._node_index_list, neighbor_nodes_pair=node_pair_list)
	# map.Set_Unit_rotate_func(Agent_action.Unit_rotate)
	# map.Add_all_node()
	# map.Add_all_edges(connected_subnodes=subnodes)
	# map.Plot_graph()

	# exit()

	# node_generator.Plot_map()
	# print('_node_index_list: ', node_generator._node_index_list)
	# for i in range(len(node_generator._node_index_list)):
	# 	print(node_generator._node_index_list[i], node_generator._reachable_position[node_generator._node_index_list[i]])
