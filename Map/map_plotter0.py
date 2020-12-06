from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from distutils.util import strtobool
import numpy as np
import time

class Plotter():
	def __init__(self, scene_name, scene_bbox, grid_size, reachable_points):
		self._map = topo_map
		_, self._ax = plt.subplots()
		self.scene_name = scene_name
		self.scene_bbox = scene_bbox
		self.grid_size = grid_size
		self.reachable_points = reachable_point

	def show_map(self, show_nodes=True, show_edges=True, show_weight=False):

		self._ax.plot(self.scene_bbox[0], self.scene_bbox[1], '-', color='orangered', linewidth=4)
		# Overlay map image
		self._ax.imshow(plt.imread('icon/' + self.scene_name + '.png'), extent=[self.scene_bbox[0][0], self.scene_bbox[0][1], self.scene_bbox[1][3], self.scene_bbox[1][4]])

		# Setup plot parameters
		plt.xticks(np.arange(np.floor(min(self.scene_bbox[0])/self.grid_size), np.ceil(max(self.scene_bbox[0])/self.grid_size)+1, 1) * self.grid_size, rotation=90)
		plt.yticks(np.arange(np.floor(min(self.scene_bbox[1])/self.grid_size), np.ceil(max(self.scene_bbox[1])/self.grid_size)+1, 1) * self.grid_size)
		plt.xlabel("x coordnates, [m]")
		plt.xlabel("z coordnates, [m]")
		# plt.title("{}: Node radius {} [m]".format(self._scene_name, str(self._node_radius)))
		plt.xlim(min(self.scene_bbox[0])-self.grid_size, max(self.scene_bbox[0])+self.grid_size)
		plt.ylim(min(self.scene_bbox[1])-self.grid_size, max(self.scene_bbox[1])+self.grid_size)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.grid(True)

		# Plot nodes
		if show_nodes:
			nodes_x = []
			nodes_y = []
			subnodes_x = []
			subnodes_y = []
			for node_name in self._map._graph.nodes:
				node_position = self._map._graph.nodes[node_name]['position']
				cir = Circle(xy=(node_position[0], node_position[2]), radius=0.2*self._map._node_radius, alpha=0.3)
				self._ax.add_patch(cir)
				cir = Circle(xy=(node_position[0], node_position[2]), radius=self._map._grid_size / 2.0, color='green', alpha=0.03)
				self._ax.add_patch(cir)

				# print('node_position: ', node_position)
				nodes_x.append(node_position[0])
				nodes_y.append(node_position[2])
				for i in range(len(subnodes_offset_x)):
					subnodes_x.append(node_position[0] + subnodes_offset_x[i])
					subnodes_y.append(node_position[2] + subnodes_offset_y[i])
			# print('nodes_x: ', nodes_x)
			# print('nodes_y: ', nodes_y)
			self._ax.plot(nodes_x, nodes_y, 'o', color="None",
				          markersize=5, linewidth=4,
				          markerfacecolor='red',
				          markeredgecolor="None",
				          markeredgewidth=2)

		if show_edges:
			for node_name, nbrs in self._map._graph.adj.items():
				node_position = copy.deepcopy(self._map._graph.nodes[node_name]['position'])
				for neighbor_node_name, weight_dict in nbrs.items():
					neighbor_node_position = copy.deepcopy(self._map._graph.nodes[neighbor_node_name]['position'])
					self._ax.plot([node_position[0], neighbor_node_position[0]], [node_position[2], neighbor_node_position[2]], 'r--', linewidth=2.0)


		plt.show()
