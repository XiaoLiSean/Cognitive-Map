from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from dijkstar import Graph, find_path
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
# from node_generation import *
import networkx as nx


class Topological_map():
	def __init__(self, controller=None, node_index_list=None, neighbor_nodes_pair=None, connected_subnodes=None):
		self._graph = nx.Graph()
		self._frames = {}
		self._orientations = [0, 90, 180, 270]
		self._controller = controller
		self._neighbor_nodes_pair = neighbor_nodes_pair
		self._node_index_list = node_index_list
		self._reachable = None
		self._connected_subnodes = connected_subnodes
		if not self._controller is None:
			self.Get_reachable_coordinate()

	def Plot_graph(self):
		plt.subplot(121)
		options = {
		    'node_color': 'red',
		    'node_size': 10,
		    'width': 3,
		}
		print(list(self._graph.nodes))
		# nx.draw(self._graph, with_labels=True, font_weight='bold')
		nx.draw(self._graph, with_labels=False, font_weight='bold', **options)

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

	def Rotate_to_degree(self, degree):
		current_orientation = self.Get_agent_rotation()['y']
		orientation_error = degree - current_orientation
		self.Unit_rotate(orientation_error)
		return

	def Unit_rotate(self, degree):
		if np.abs(degree) < 2:
			return None
		degree_corrected = degree
		while degree_corrected > 180:
			degree_corrected -= 360
		while degree_corrected < -180:
			degree_corrected += 360
		if degree_corrected > 0:
			self._event = self._controller.step(action='RotateRight', degrees=np.abs(degree_corrected))
		else:
			self._event = self._controller.step(action='RotateLeft', degrees=np.abs(degree_corrected))

	def Teleport_agent(self, position):
		self.Update_event()
		if isinstance(position, dict):
			position_list = list(position.values())
		else:
			position_list = copy.deepcopy(position)
		self._event = self._controller.step(action='Teleport', x=position_list[0], y=position_list[1], z=position_list[2])

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
			self.Teleport_agent(self._reachable[self._node_index_list[node_index]])
			for orientation in self._orientations:
				self.Rotate_to_degree(orientation)
				frames.append(self.Get_frame())
			self.Add_node(node_num=node_index, frame=frames)
		return

# 	Assume frame always comes as a list of length 4
	def Add_node(self, node_num, frame):
		for sub_node_index in range(len(frame)):
			self.Add_sub_node(node_num=node_num, orientation=self._orientations[sub_node_index], frame=frame[sub_node_index])
		for sub_node_index in range(len(frame)):
			if sub_node_index < 3:
				self.Add_edge(node_1=node_num, orientation_1=self._orientations[sub_node_index],
				node_2=node_num, orientation_2=self._orientations[sub_node_index + 1])
			elif sub_node_index == 3:
				self.Add_edge(node_1=node_num, orientation_1=self._orientations[sub_node_index],
				node_2=node_num, orientation_2=self._orientations[0])
		return

	def Add_all_edges(self, connected_subnodes=None):
		if not connected_subnodes is None:
			self._connected_subnodes = connected_subnodes
		# print('self._neighbor_nodes_pair: ', self._neighbor_nodes_pair)
		# print('self._connected_subnodes: ', self._connected_subnodes)
		for nodes_pair_index in range(len(self._connected_subnodes)):
			self.Add_edge_between_nodes(node_pair_index=nodes_pair_index, orientations=connected_subnodes[nodes_pair_index])

	def Add_edge_between_nodes(self, node_pair_index, orientations):
		for orientation in orientations:
			self.Add_edge(node_1=self._node_index_list.index(self._neighbor_nodes_pair[node_pair_index][0]), orientation_1=self._orientations[orientation],
				node_2=self._node_index_list.index(self._neighbor_nodes_pair[node_pair_index][1]), orientation_2=self._orientations[orientation])
		return

	def Add_sub_node(self, node_num, orientation, frame):
		self._graph.add_node(self.Get_node_name(node_num, orientation))
		self._frames[self.Get_node_name(node_num, orientation)] = copy.deepcopy(frame)
		return

	def Add_edge(self, node_1, orientation_1, node_2, orientation_2):
		self._graph.add_edge(self.Get_node_name(node_1, orientation_1), self.Get_node_name(node_2, orientation_2))
		return

	def Get_node_name(self, node_num, orientation):
		return 'node_' + str(node_num) + '_degree_' + str(orientation)
