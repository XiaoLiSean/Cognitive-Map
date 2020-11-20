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
sys.path.append('../')
from experiment import *
import networkx as nx


class Topological_map():
	def __init__(self, controller=None, node_index_list=None, scene_graph_method=None, neighbor_nodes_pair=None, connected_subnodes=None):
		self._graph = nx.Graph()
		self._data = []
		self._orientations = [0, 90, 180, 270]
		self._controller = controller
		self._neighbor_nodes_pair = neighbor_nodes_pair
		self._node_index_list = node_index_list
		self._reachable = None
		self._connected_subnodes = connected_subnodes
		self._dij_graph = dij.Graph()
		self._grid_size = 0.25
		self._node_radius = 1.5

		self._rotation_cost = 1
		self._action_direction = {'forward': 0, 'right': 1, 'left': 2, 'backward': 3}
		self._action_direction_cost_coeff = [0.5, 1, 1, 0.7]

		self._event = None

		self._scene_graph_method = None
		self._Unit_rotate_func = None
		self._Rotate_to_degree_func = None
		self._Teleport_agent_func = None
		if not self._controller is None:
			self.Get_reachable_coordinate()

	def Get_node_dij_index(self, node_name):
		node_num, orientation = self.Get_node_index_orien(node_name)
		return node_num * len(self._orientations) + self._orientations.index(orientation)

	def Get_node_index_from_dij_index(self, dij_index):
		node_num = int(dij_index / len(self._orientations))
		orientation = int(dij_index % len(self._orientations))
		node_name = self.Get_node_name(node_num=node_num, orientation=self._orientations[orientation])
		return node_name

	def Build_dij_graph(self):

		for current_node_name, neighbor_nodes in self._graph.adj.items():

			current_node_dij_index = self.Get_node_dij_index(node_name=current_node_name)

			for neighbor_node_name, weight_dict in neighbor_nodes.items():

				neighbor_node_dij_index = self.Get_node_dij_index(node_name=neighbor_node_name)
				self._dij_graph.add_edge(current_node_dij_index, neighbor_node_dij_index, weight_dict['weight'])

		return

	def Find_dij_path(self, current_node_index, current_orientation, goal_node_index, goal_orientation):

		current_node_name = self.Get_node_name(node_num=current_node_index, orientation=current_orientation)
		current_dij_index = self.Get_node_dij_index(node_name=current_node_name)

		goal_node_name = self.Get_node_name(node_num=goal_node_index, orientation=goal_orientation)
		goal_dij_index = self.Get_node_dij_index(node_name=goal_node_name)

		result = dij.find_path(self._dij_graph, current_dij_index, goal_dij_index)
		path = result.nodes

		path_nodes_name = []
		for node_dij_index in path:
			path_nodes_name.append(self.Get_node_index_from_dij_index(node_dij_index))

		return path_nodes_name

	def init_data(self):
		if not self._node_index_list is None:
			self._data = [[[] for _ in range(4)] for _ in range(len(self._node_index_list))]

	def Plot_graph(self):
		plt.subplot(121)
		options = {
		    'node_color': 'red',
		    'node_size': 10,
		    'width': 3,
		}
		# print(list(self._graph.nodes))
		# nx.draw(self._graph, with_labels=True, font_weight='bold')
		nx.draw(self._graph, with_labels=False, font_weight='bold', **options)

	def Set_Teleport_agent_func(self, Teleport_agent):
		self._Teleport_agent_func = Teleport_agent

	def Set_scene_graph_method(self, scene_graph_method):
		self._scene_graph_method = scene_graph_method

	def Set_Unit_rotate_func(self, Unit_rotate):
		self._Unit_rotate_func = Unit_rotate

	def Set_Rotate_to_degree_func(self, Rotate_to_degree_func):
		self._Rotate_to_degree_func = Rotate_to_degree_func

	def Set_controller(self, controller):
		self._controller = controller

	def Set_neighbor_node(self, neighbor_nodes_pair):
		self._neighbor_nodes_pair = neighbor_nodes_pair

	def Set_node_index_list(self, node_index_list):
		self._node_index_list = node_index_list

	def Update_event(self):
		self._event = self._controller.step('Pass')

	def Get_reachable_coordinate(self):
		self._event = self._controller.step(action='GetReachablePositions')
		self._reachable = self._event.metadata['actionReturn']
		return self._event.metadata['actionReturn']

	def Get_agent_rotation(self):
		self.Update_event()
		return self._event.metadata['agent']['rotation']

	def Get_frame(self):
		self.Update_event()
		return self._event.frame

	def Add_all_node(self, node_index_list=None):

		if not node_index_list is None:
			self._node_index_list = node_index_list
		# print('self._node_index_list: ', self._node_index_list)
		for node_index in range(len(self._node_index_list)):
			# print('node_index: ', node_index)
			frames = []
			scene_graphs = []
			node_position = list(self._reachable[self._node_index_list[node_index]].values())

			self._Teleport_agent_func(position=node_position, useful=True)

			for orien_index, orientation in enumerate(self._orientations):

				self._Rotate_to_degree_func(goal_degree=orientation)
				frames.append(self.Get_frame())
				self._data[node_index][orien_index].append({'image': [], 'scene_graph': []})
				if not self._scene_graph_method is None:
					scene_graphs.append(self._scene_graph_method())

				# self._data[node_index][orien_index].append({'image': self.Get_frame(), 'scene_graph': self._scene_graph_method()})
			if len(scene_graphs) > 0:
				self.Add_node(node_num=node_index, position=node_position, frames=frames, scene_graphs=scene_graphs)
			else:
				self.Add_node(node_num=node_index, position=node_position, frames=frames)

		return

# 	Assume frames always comes as a list of length 4
	def Add_node(self, node_num, position, frames, scene_graphs=None):

		for sub_node_index in range(len(frames)):

			if scene_graphs is None:
				self.Add_sub_node(node_num=node_num, position=position, orientation=self._orientations[sub_node_index], frame=frames[sub_node_index])

			else:
				self.Add_sub_node(node_num=node_num, position=position, orientation=self._orientations[sub_node_index], frame=frames[sub_node_index], scene_graph=scene_graphs[sub_node_index])

		for sub_node_index in range(len(frames)):

			if sub_node_index < len(self._orientations) - 1:
				self.Add_edge(node_1=node_num, orientation_1=self._orientations[sub_node_index],
				node_2=node_num, orientation_2=self._orientations[sub_node_index + 1])

			elif sub_node_index == len(self._orientations) - 1:
				self.Add_edge(node_1=node_num, orientation_1=self._orientations[sub_node_index],
				node_2=node_num, orientation_2=self._orientations[0])

		return

	def Add_sub_node(self, node_num, position, orientation, frame, scene_graph=None):
		self._graph.add_nodes_from([
		(self.Get_node_name(node_num, orientation), {'position': position, 'image': frame, 'scene_graph': scene_graph})
		])
		# print(self._graph.nodes[self.Get_node_name(node_num, orientation)])
		return

	def Add_all_edges(self, connected_subnodes=None):

		if not connected_subnodes is None:
			self._connected_subnodes = connected_subnodes

		for nodes_pair_index in range(len(self._connected_subnodes)):
			self.Add_edge_between_nodes(node_pair_index=nodes_pair_index, orientations=connected_subnodes[nodes_pair_index])

	def Add_edge_between_nodes(self, node_pair_index, orientations):

		first_node_position = list(self._reachable[self._neighbor_nodes_pair[node_pair_index][0]].values())
		second_node_position = list(self._reachable[self._neighbor_nodes_pair[node_pair_index][1]].values())

		node_position_diff = list(map(lambda x, y: np.abs(x - y), first_node_position, second_node_position))
		# print('node_position_diff: ',node_position_diff)
		max_diff_index = node_position_diff.index(max(node_position_diff))

		if max_diff_index == 0:
			weight_coeff = [1, 0.5, 0.5, 1]
		elif max_diff_index == 2:
			weight_coeff = [0.5, 1, 1, 0.5]
		weight = sum(node_position_diff)

		if weight > 6 * self._grid_size:
			weight *= 2

		for orientation in orientations:
			# print('orientation: ', orientation)
			# print('weight * weight_coeff[self._orientations.index(orientation)]: ', weight * weight_coeff[orientation])

			self.Add_edge(node_1=self._node_index_list.index(self._neighbor_nodes_pair[node_pair_index][0]), orientation_1=self._orientations[orientation],
				node_2=self._node_index_list.index(self._neighbor_nodes_pair[node_pair_index][1]), orientation_2=self._orientations[orientation], weight=weight * weight_coeff[orientation])

		return

	def Add_edge(self, node_1, orientation_1, node_2, orientation_2, weight=0.25):
		self._graph.add_weighted_edges_from([
			(self.Get_node_name(node_1, orientation_1), self.Get_node_name(node_2, orientation_2), weight)
			])
		return

	def Get_node_name(self, node_num, orientation):
		return 'node_' + str(node_num) + '_degree_' + str(orientation)

	def Get_node_index_orien(self, node_name):
		name_split = node_name.split('_')
		return int(name_split[1]), int(name_split[3])

	def Get_node_value_by_name(self, node_name):
		return self._graph.nodes[node_name]

	def Get_node_value(self, node_num, orientation):
		return self._graph.nodes[self.Get_node_name(node_num, orientation)]

	def get_scene_bbox(self):
		self.Update_event()
		data = self._event.metadata['sceneBounds']
		center_x = data['center']['x']
		center_z = data['center']['z']
		size_x = data['size']['x']
		size_z = data['size']['z']

		bbox_x = [center_x-size_x*0.5, center_x+size_x*0.5, center_x+size_x*0.5, center_x-size_x*0.5, center_x-size_x*0.5]
		bbox_z = [center_z+size_z*0.5, center_z+size_z*0.5, center_z-size_z*0.5, center_z-size_z*0.5, center_z+size_z*0.5]

		return (bbox_x, bbox_z)

	def show_map(self, show_nodes=False, show_edges=False, show_weight=False):
		self.Update_event()
		# Plot reachable points
		points = self.Get_reachable_coordinate()
		X = [p['x'] for p in points]
		Z = [p['z'] for p in points]

		fig, ax = plt.subplots()

		plt.plot(X, Z, 'o', color='lightskyblue',
		         markersize=5, linewidth=4,
		         markerfacecolor='white',
		         markeredgecolor='lightskyblue',
		         markeredgewidth=2)

		# Plot rectangle bounding the entire scene
		scene_bbox = self.get_scene_bbox()
		plt.plot(scene_bbox[0], scene_bbox[1], '-', color='orangered', linewidth=4)

		# Plot objects 2D boxs
		for obj in self._event.metadata['objects']:
			size = obj['axisAlignedBoundingBox']['size']
			center = obj['axisAlignedBoundingBox']['center']
			rect = Rectangle(xy=(center['x'] - size['x']*0.5, center['z'] - size['z']*0.5), width=size['x'], height=size['z'], fill=True, alpha=0.3, color='darkgray', hatch='//')
			ax.add_patch(rect)

		# Setup plot parameters
		plt.xticks(np.arange(np.floor(min(scene_bbox[0])/self._grid_size), np.ceil(max(scene_bbox[0])/self._grid_size)+1, 1) * self._grid_size, rotation=90)
		plt.yticks(np.arange(np.floor(min(scene_bbox[1])/self._grid_size), np.ceil(max(scene_bbox[1])/self._grid_size)+1, 1) * self._grid_size)
		plt.xlabel("x coordnates, [m]")
		plt.xlabel("z coordnates, [m]")
		# plt.title("{}: Node radius {} [m]".format(self._scene_name, str(self._node_radius)))
		plt.xlim(min(scene_bbox[0])-self._grid_size, max(scene_bbox[0])+self._grid_size)
		plt.ylim(min(scene_bbox[1])-self._grid_size, max(scene_bbox[1])+self._grid_size)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.grid(True)

		subnodes_offset_x = [0, self._grid_size / 2.0, 0, -self._grid_size / 2.0]
		subnodes_offset_y = [self._grid_size / 2.0, 0, -self._grid_size / 2.0, 0]

		# subnodes_offset_x = [0, 0, 0, 0]
		# subnodes_offset_y = [self._grid_size / 2.0, 0, -self._grid_size / 2.0, 0]


		# Plot nodes
		if show_nodes:
			nodes_x = []
			nodes_y = []
			subnodes_x = []
			subnodes_y = []


			for node_name in self._graph.nodes:
				node_position = self._graph.nodes[node_name]['position']
				cir = Circle(xy=(node_position[0], node_position[2]), radius=self._node_radius, alpha=0.05)
				ax.add_patch(cir)
				cir = Circle(xy=(node_position[0], node_position[2]), radius=self._grid_size / 2.0, color='red', alpha=0.3)
				ax.add_patch(cir)

				# print('node_position: ', node_position)
				nodes_x.append(node_position[0])
				nodes_y.append(node_position[2])
				for i in range(len(subnodes_offset_x)):
					subnodes_x.append(node_position[0] + subnodes_offset_x[i])
					subnodes_y.append(node_position[2] + subnodes_offset_y[i])
			print('nodes_x: ', nodes_x)
			print('nodes_y: ', nodes_y)
			plt.plot(nodes_x, nodes_y, 'o', color="None",
			         markersize=5, linewidth=4,
			         markerfacecolor='red',
			         markeredgecolor="None",
			         markeredgewidth=2)
			plt.plot(subnodes_x, subnodes_y, 'o', color="None",
			         markersize=3, linewidth=4,
			         markerfacecolor='red',
			         markeredgecolor="None",
			         markeredgewidth=2)
		if show_edges:
			for node_name, nbrs in self._graph.adj.items():
				# print('n: ', n)
				node_position = copy.deepcopy(self._graph.nodes[node_name]['position'])

				node_num, node_orien = self.Get_node_index_orien(node_name)


				orientation_index = self._orientations.index(node_orien)
				node_position[0] += subnodes_offset_x[orientation_index]
				node_position[2] += subnodes_offset_y[orientation_index]
				# print('------------------------------')
				# print('node_name: ', node_name)
				# print('node_orien: ', node_orien)
				# print('node_offset: ', subnodes_offset_x[orientation_index], subnodes_offset_y[orientation_index])


				for neighbor_node_name, weight_dict in nbrs.items():

					neighbor_node_position = copy.deepcopy(self._graph.nodes[neighbor_node_name]['position'])

					neighbor_node_num, neighbor_node_orien = self.Get_node_index_orien(neighbor_node_name)
					# if not node_num == neighbor_node_num:
					# 	continue

					neighbor_orientation_index = self._orientations.index(neighbor_node_orien)
					neighbor_node_position[0] += subnodes_offset_x[neighbor_orientation_index]
					neighbor_node_position[2] += subnodes_offset_y[neighbor_orientation_index]
					# print('neighbor_node_name: ', neighbor_node_name)
					# print('neighbor_node_offset: ', subnodes_offset_x[neighbor_orientation_index], subnodes_offset_y[neighbor_orientation_index])
					# print('node_position: ', node_position)
					# print('neighbor_node_position: ', neighbor_node_position)
					ax.plot([node_position[0], neighbor_node_position[0]], [node_position[2], neighbor_node_position[2]], 'r--', linewidth=2.0)
					if show_weight:
						ax.text((node_position[0]+neighbor_node_position[0]) / 2.0, (node_position[2]+neighbor_node_position[2]) / 2.0, weight_dict['weight'], size=6,
					        ha="center", va="center",
					        bbox=dict(boxstyle="round",
					                  ec=(0.5, 0.25, 0.25),
					                  fc=(0.5, 0.4, 0.4),
					                  )
					        )
		plt.show()


if __name__ == '__main__':

	Agent_action = Agent_action(AI2THOR=False, scene_type=1, scene_num=1, grid_size=0.25, rotation_step=30, sleep_time=0.1, save_directory='./data')
	topo_map = Topological_map(controller=Agent_action._controller)

	topo_map.init_data()
	topo_map.Set_Teleport_agent_func(Agent_action.Teleport_agent)
	topo_map.Set_Rotate_to_degree_func(Agent_action.Rotate_to_degree)

	topo_map.Add_all_node()
	topo_map.Add_all_edges(connected_subnodes=subnodes)
	topo_map.Build_dij_graph()

	path = topo_map.Find_dij_path(current_node_index=13, current_orientation=0, goal_node_index=18, goal_orientation=90)
	topo_map.show_map(show_nodes=True, show_edges=True)
