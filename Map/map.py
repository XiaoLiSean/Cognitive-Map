from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import dijkstar as dij
from termcolor import colored
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
from lib.scene_graph_generation import Scene_Graph
sys.path.append('../')
from experiment import *
import networkx as nx
from Map.Frelement import *


class Topological_map():
	def __init__(self, controller=None, node_index_list=None, scene_graph_method=None, neighbor_nodes_pair=None, connected_subnodes=None):
		self._graph = nx.Graph()
		self._data = []
		self._orientations = [0, 90, 180, 270]
		self._controller = controller
		self._neighbor_nodes_pair = neighbor_nodes_pair
		self._node_index_list = node_index_list
		self._updated_needed = False
		if not self._node_index_list is None:
			self.init_data()
			self._updated_needed = True

		self._reachable = None
		self._connected_subnodes = connected_subnodes
		self._dij_graph = dij.Graph()
		self._grid_size = 0.25
		self._node_radius = 1.5
		self._rotation_cost = 0.0001
		self._action_direction = {'forward': 0, 'right': 1, 'left': 2, 'backward': 3}
		self._action_direction_cost_coeff = [0.5, 1, 1, 0.7]

		self._event = None
		# This two variable is used for localization in dynamics env using Scene Graph
		self._CurrentSceneGraph = Scene_Graph()
		self._CurrentImg = []


		self._Unit_rotate_func = None
		self._Rotate_to_degree_func = None
		self._Teleport_agent_func = None
		if not self._controller is None:
			self.Get_reachable_coordinate()
		self._subnode_plan = False

		self._subnode_records = []
		self._orderi = 12  #order of Frouier transform used in dynamic map, max 24

	def init_subnode_records(self):
		self._subnode_records = [subnode_dynamic_record() for _ in range(len(self._graph.nodes))]

	def build_frelement_subnode_object(self, obj_name, times, signal, node_name):

		if obj_name not in list(self._graph.nodes[node_name]['frelements'].keys()):
			self._graph.nodes[node_name]['frelements'][obj_name] = Frelement()
		self._graph.nodes[node_name]['frelements'][obj_name].build(times=times, signal=signal, length=len(times), orderi=self._orderi)

	def build_frelements(self):

		for subnode_record in self._subnode_records:
			for object_index in range(len(subnode_record._subnode_name)):
				self.build_frelement_subnode_object(obj_name=subnode_record._subnode_name[object_index],
					times=subnode_record._times[object_index], signal=subnode_record._signals[object_index], node_name=subnode_record._subnode_name)

# 	Add new observations of some time to the frelements,
	def add_obs_frelements(self, objs, time_current, node_name):

		objs_obs_list = []

		for obj in objs:

			objs_obs_list.append(obj['name'])

			if obj['name'] not in list(self._graph.nodes[node_name]['frelements'].keys()):
				self._graph.nodes[node_name]['frelements'][obj['name']] = Frelement()

		for obj_in_node in list(self._graph.nodes[node_name]['frelements'].keys()):

			if obj_in_node in objs_obs_list:
				self._graph.nodes[node_name]['frelements'][obj['name']].add(times=[time_current], states=[1])
			else:
				self._graph.nodes[node_name]['frelements'][obj['name']].add(times=[time_current], states=[0])

# 	Estimate the probability of certain object showing in certain node in certain time
	def estimate_prob_obj_vs_node(self, obj_name, time_estimate, node_name):

		if obj_name not in list(self._graph.nodes[node_name]['frelements'].keys()):
			return 0

		else:
			return self._graph.nodes[node_name]['frelements'][obj_name].estimate(times=[time_estimate])

	def estimate_obj_node(self, obj_name, time_estimate):

		all_nodes_list = list(self._graph.nodes)
		node_probs_list = []

		for node_name in all_nodes_list:
			node_probs_list.append(self.estimate_prob_obj_vs_node(obj_name=obj_name, time_estimate=time_estimate, node_name=node_name))

		most_prob_node_index = node_probs_list.index(max(node_probs_list))
		return all_nodes_list[most_prob_node_index]

	def get_scene_graph(self, visible_filter=True, scan_entire_corn=False):
		SG = Scene_Graph()
		self.Update_event()

		return SG.get_SG_as_dict()

	def Set_env_from_Robot(self, Robot):
		self._controller = Robot._AI2THOR_controller._controller
		self.Set_Teleport_agent_func(Robot._AI2THOR_controller.Teleport_agent)
		self.Set_Rotate_to_degree_func(Robot._AI2THOR_controller.Rotate_to_degree)

	def Clear_graph(self):
		self._graph = nx.Graph()

	def Export_graph(self):
		if self._updated_needed:
			logging.warning('Graph needs to be updated to match node list')
		return copy.deepcopy(self._graph)

	def Update_topo_map(self, node_index_list=None, node_pair_list=None, connected_subnodes=None):
		if self._Rotate_to_degree_func is None or self._Teleport_agent_func is None:
			logging.error('Action function not set in topo map class')
			return
		self.Get_reachable_coordinate()
		if self._controller is None:
			logging.error('Controller not set in topo map class')
			return

		if not node_index_list is None:
			self.Set_node_index_list(node_index_list=node_index_list)

		if not node_pair_list is None:
			self.Set_neighbor_node(neighbor_nodes_pair=node_pair_list)

		if not connected_subnodes is None:
			self.Set_subnode_connection(connected_subnodes=connected_subnodes)

		if self._node_index_list is None or self._connected_subnodes is None:
			logging.error('Node list or subnode connection is not set')
			return

		self.Add_all_node()
		self.Add_all_edges()
		self._updated_needed = False

	def _wrap_to_360(self, degree):
		while degree >= 360:
			degree -=360
		while degree < 0:
			degree +=360
		return int(degree)

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
		self._updated_needed = True
		self.init_data()

	def Set_subnode_connection(self, connected_subnodes):
		self._connected_subnodes = connected_subnodes
		self._updated_needed = True

	def Update_event(self):
		self._event = self._controller.step('Pass')
		objs = self._event.metadata['objects']
		self._CurrentImg = self._event.frame
		Img = Image.fromarray(self._CurrentImg, 'RGB')
		self._CurrentSceneGraph.reset()
		objs = self._CurrentSceneGraph.visibleFilter_by_2Dbbox(objs, self._event.instance_detections2D)
		self._CurrentSceneGraph.update_from_data(objs, image_size=Img.size[0])

	def Get_reachable_coordinate(self):
		self._event = self._controller.step(action='GetReachablePositions')
		self._reachable = self._event.metadata['actionReturn']
		return self._event.metadata['actionReturn']

	def Get_agent_orientation(self):
		self.Update_event()
		return self._event.metadata['agent']['rotation']

	def Get_frame(self):
		self.Update_event()
		return self._CurrentImg

	def Get_SceneGraph(self):
		self.Update_event()
		return self._CurrentSceneGraph.get_SG_as_dict()

	# ==========================================================================
	# Clear the current graph and add all nodes in graph
	# ==========================================================================
	def Add_all_node(self, node_index_list=None):

		if not node_index_list is None:
			self.Set_node_index_list(node_index_list)

		if self._node_index_list is None or len(self._node_index_list) == 0:
			logging.error('There is no node in map')
			return

		self.Clear_graph()

		for node_index in range(len(self._node_index_list)):
			frames = []
			scene_graphs = []
			# print('self._node_index_list[node_index]: ', self._node_index_list[node_index])
			node_position = list(self._reachable[self._node_index_list[node_index]].values())

			self._Teleport_agent_func(position=node_position, save_image=False)

			for orien_index, orientation in enumerate(self._orientations):

				self._Rotate_to_degree_func(goal_degree=orientation)
				frames.append(self.Get_frame())
				self._data[node_index][orien_index].append({'image': [], 'scene_graph': []})
				scene_graphs.append(self.Get_SceneGraph())

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

# 	Add subnode to the graph(networkx), whose attributes includes 1. position 2. rgb image 3. scene graph 4. frelements
	def Add_sub_node(self, node_num, position, orientation, frame, scene_graph=None):
		self._graph.add_nodes_from([
		(self.Get_node_name(node_num, orientation), {'position': position, 'image': frame, 'scene_graph': scene_graph, 'frelements': {}})
		])
		return

	def Add_all_edges(self, connected_subnodes=None):

		if not connected_subnodes is None:
			self.Set_subnode_connection(connected_subnodes=connected_subnodes)

		for nodes_pair_index in range(len(self._connected_subnodes)):
			self.Add_edge_between_nodes(node_pair_index=nodes_pair_index, orientations=self._connected_subnodes[nodes_pair_index])

	def Add_edge_between_nodes(self, node_pair_index, orientations):

		first_node_position = list(self._reachable[self._neighbor_nodes_pair[node_pair_index][0]].values())
		second_node_position = list(self._reachable[self._neighbor_nodes_pair[node_pair_index][1]].values())

		node_position_diff = list(map(lambda x, y: np.abs(x - y), first_node_position, second_node_position))
		max_diff_index = node_position_diff.index(max(node_position_diff))

		if max_diff_index == 0:
			weight_coeff = [1, 0.5, 0.5, 1]
		elif max_diff_index == 2:
			weight_coeff = [0.5, 1, 1, 0.5]
		weight = sum(node_position_diff)

		if weight > 5 * self._grid_size:
			weight *= 2

		# print('self._neighbor_nodes_pair: ', self._neighbor_nodes_pair)
		# print('self._connected_subnodes: ', self._connected_subnodes)


		for orientation in orientations:

			# print('orientation: ', orientation)

			self.Add_edge(node_1=self._node_index_list.index(self._neighbor_nodes_pair[node_pair_index][0]), orientation_1=self._orientations[orientation],
						  node_2=self._node_index_list.index(self._neighbor_nodes_pair[node_pair_index][1]), orientation_2=self._orientations[orientation],
						  weight=weight * weight_coeff[orientation])
		# exit()
		return

	def Get_object_closest_node(self):
		pass

	def Get_adj(self):
		return self._graph.adj.items()

	def Get_node_adj_by_name(self, node_name):
		node_adj = self._graph.adj
		return node_adj[node_name]

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

	def Get_node_value_dict_by_name(self, node_name):
		return self._graph.nodes[node_name]

	def Get_node_value_dict(self, node_num, orientation):
		return self._graph.nodes[self.Get_node_name(node_num, orientation)]

	def Get_node_value_by_name(self, node_name):
		values = self._graph.nodes[node_name]
		return values['position'], values['image'], values['scene_graph']

	def Get_node_value(self, node_num, orientation):
		values = self._graph.nodes[self.Get_node_name(node_num, orientation)]
		return values['position'], values['image'], values['scene_graph']
