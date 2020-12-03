from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from distutils.util import strtobool
import numpy as np
import time
import copy
import random
import logging
import os
import sys
sys.path.append('../')
from experiment import *
from .map import Topological_map
import networkx as nx

class Plotter():
	def __init__(self, topo_map=None):
		self._map = topo_map

	def Update_topo_map(self, topo_map):
		self._map = topo_map

	def get_scene_bbox(self):
		self._map.Update_event()
		data = self._map._event.metadata['sceneBounds']
		center_x = data['center']['x']
		center_z = data['center']['z']
		size_x = data['size']['x']
		size_z = data['size']['z']

		bbox_x = [center_x-size_x*0.5, center_x+size_x*0.5, center_x+size_x*0.5, center_x-size_x*0.5, center_x-size_x*0.5]
		bbox_z = [center_z+size_z*0.5, center_z+size_z*0.5, center_z-size_z*0.5, center_z-size_z*0.5, center_z+size_z*0.5]

		return (bbox_x, bbox_z)

	def show_map(self, show_nodes=True, show_edges=False, show_weight=False):
		self._map.Update_event()
		# Plot reachable points
		points = self._map.Get_reachable_coordinate()
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
		for obj in self._map._event.metadata['objects']:
			size = obj['axisAlignedBoundingBox']['size']
			center = obj['axisAlignedBoundingBox']['center']
			rect = Rectangle(xy=(center['x'] - size['x']*0.5, center['z'] - size['z']*0.5), width=size['x'], height=size['z'], fill=True, alpha=0.3, color='darkgray', hatch='//')
			ax.add_patch(rect)

		# Setup plot parameters
		plt.xticks(np.arange(np.floor(min(scene_bbox[0])/self._map._grid_size), np.ceil(max(scene_bbox[0])/self._map._grid_size)+1, 1) * self._map._grid_size, rotation=90)
		plt.yticks(np.arange(np.floor(min(scene_bbox[1])/self._map._grid_size), np.ceil(max(scene_bbox[1])/self._map._grid_size)+1, 1) * self._map._grid_size)
		plt.xlabel("x coordnates, [m]")
		plt.xlabel("z coordnates, [m]")
		# plt.title("{}: Node radius {} [m]".format(self._scene_name, str(self._node_radius)))
		plt.xlim(min(scene_bbox[0])-self._map._grid_size, max(scene_bbox[0])+self._map._grid_size)
		plt.ylim(min(scene_bbox[1])-self._map._grid_size, max(scene_bbox[1])+self._map._grid_size)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.grid(True)

		subnodes_offset_x = [0, self._map._grid_size / 2.0, 0, -self._map._grid_size / 2.0]
		subnodes_offset_y = [self._map._grid_size / 2.0, 0, -self._map._grid_size / 2.0, 0]

		# subnodes_offset_x = [0, 0, 0, 0]
		# subnodes_offset_y = [self._grid_size / 2.0, 0, -self._grid_size / 2.0, 0]
		

		# Plot nodes
		if show_nodes:
			nodes_x = []
			nodes_y = []
			subnodes_x = []
			subnodes_y = []
			

			for node_name in self._map._graph.nodes:
				node_position = self._map._graph.nodes[node_name]['position']
				cir = Circle(xy=(node_position[0], node_position[2]), radius=self._map._node_radius, alpha=0.05)
				ax.add_patch(cir)
				cir = Circle(xy=(node_position[0], node_position[2]), radius=self._map._grid_size / 2.0, color='red', alpha=0.3)
				ax.add_patch(cir)

				# print('node_position: ', node_position)
				nodes_x.append(node_position[0])
				nodes_y.append(node_position[2])
				for i in range(len(subnodes_offset_x)):
					subnodes_x.append(node_position[0] + subnodes_offset_x[i])
					subnodes_y.append(node_position[2] + subnodes_offset_y[i])
			# print('nodes_x: ', nodes_x)
			# print('nodes_y: ', nodes_y)
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
			for node_name, nbrs in self._map._graph.adj.items():
				# print('n: ', n)
				node_position = copy.deepcopy(self._map._graph.nodes[node_name]['position'])

				node_num, node_orien = self._map.Get_node_index_orien(node_name)


				orientation_index = self._map._orientations.index(node_orien)
				node_position[0] += subnodes_offset_x[orientation_index]
				node_position[2] += subnodes_offset_y[orientation_index]
				# print('------------------------------')
				# print('node_name: ', node_name)
				# print('node_orien: ', node_orien)
				# print('node_offset: ', subnodes_offset_x[orientation_index], subnodes_offset_y[orientation_index])
				

				for neighbor_node_name, weight_dict in nbrs.items():

					neighbor_node_position = copy.deepcopy(self._map._graph.nodes[neighbor_node_name]['position'])

					neighbor_node_num, neighbor_node_orien = self._map.Get_node_index_orien(neighbor_node_name)
					# if not node_num == neighbor_node_num:
					# 	continue

					neighbor_orientation_index = self._map._orientations.index(neighbor_node_orien)
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

				# print('nbrs: ', nbrs)
				# print('weight: ', nbrs[])
		# ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'r--', linewidth=2.0)
		# ax.text((node_i[0]+node_j[0]) / 2.0, (node_i[1]+node_j[1]) / 2.0, int(cost), size=8,
		#         ha="center", va="center",
		#         bbox=dict(boxstyle="round",
		#                   ec=(1., 0.5, 0.5),
		#                   fc=(1., 0.8, 0.8),
		#                   )
		#         )
		plt.show()