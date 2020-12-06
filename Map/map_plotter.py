from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle, Wedge
from lib.params import NODES, ADJACENT_NODES_SHIFT_GRID, VISBILITY_DISTANCE
from termcolor import colored
from PIL import Image
import numpy as np
import time

class Plotter():
	def __init__(self, scene_name, scene_bbox, grid_size, reachable_points, client=None, comfirmed=None):
		fig = plt.figure(figsize=(12,8), constrained_layout=True)
		gs = GridSpec(3, 3, figure=fig)
		self.toggleMap = fig.add_subplot(gs[0:-1, :])
		self.currentView = fig.add_subplot(gs[2, 0])
		self.goalView = fig.add_subplot(gs[2, 1])
		self.scene_name = scene_name
		self.scene_bbox = scene_bbox
		self.grid_size = grid_size
		self.node_radius = VISBILITY_DISTANCE
		self.reachable_points = reachable_points
		self.multithread_node = dict(client=client, comfirmed=comfirmed)

	def is_reachable(self, pi, pj):
		map = self.reachable_points
		diff = (np.array(pj) - np.array(pi)) / self.grid_size
		sign = np.sign(diff)
		diff = np.abs(diff.astype(int))
		current_pose = dict(x=pi[0], y=map[0]['y'], z=pi[1])
		count = 0
		for i in range(1, diff[0]+1):
			current_pose['x'] += sign[0]*self.grid_size
			if current_pose in map:
				count += 1
		for j in range(1, diff[1]+1):
			current_pose['z'] += sign[1]*self.grid_size
			if current_pose in map:
				count += 1
		if count == (diff[0] + diff[1]):
			return True


		current_pose = dict(x=pi[0], y=map[0]['y'], z=pi[1])
		count = 0
		for j in range(1, diff[1]+1):
			current_pose['z'] += sign[1]*self.grid_size
			if current_pose in map:
				count += 1
		for i in range(1, diff[0]+1):
			current_pose['x'] += sign[0]*self.grid_size
			if current_pose in map:
				count += 1
		if count == diff[0] + diff[1]:
			return True

		return False


	def add_edges(self, nodes):
		edges = []
		# Iterature through nodes to generate edges
		for i in range(len(nodes)-1):
			node_i = nodes[i]
			for j in range(i+1, len(nodes)):
				node_j = nodes[j]
				diff = np.abs(np.array(node_i) - np.array(node_j))
				is_edge = False

				if diff[0] < self.node_radius:
					if diff[1] <= ADJACENT_NODES_SHIFT_GRID * self.grid_size:
						is_edge = self.is_reachable(node_i, node_j)
						if is_edge:
							cost = (diff[0] + diff[1]) / self.grid_size
							edges.append((node_i, node_j, int(cost)))


				if diff[1] < self.node_radius:
					if diff[0] <= ADJACENT_NODES_SHIFT_GRID * self.grid_size:
						is_edge = self.is_reachable(node_i, node_j)
						if is_edge:
							cost = (diff[0] + diff[1]) / self.grid_size
							edges.append((node_i, node_j, int(cost)))


				if is_edge:
					self.toggleMap.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'r--', linewidth=2.0)

	def add_nodes(self):
		nodes_x = []
		nodes_y = []
		points = NODES[self.scene_name]
		for idx, p in enumerate(points):
			circ = Circle(xy = (p[0], p[1]), radius=0.2*self.node_radius, alpha=0.3)
			self.toggleMap.add_patch(circ)
			nodes_x.append(p[0])
			nodes_y.append(p[1])
			self.toggleMap.text(p[0], p[1], str(idx))

		return (nodes_x, nodes_y)

	def show_map(self, show_nodes=True, show_edges=True):
		# Plot reachable points
		points = self.reachable_points
		X = [p['x'] for p in points]
		Z = [p['z'] for p in points]

		self.toggleMap.plot(X, Z, 'o', color='lightskyblue',
			         markersize=5, linewidth=4,
			         markerfacecolor='white',
			         markeredgecolor='lightskyblue',
			         markeredgewidth=2)

		self.toggleMap.plot(self.scene_bbox[0], self.scene_bbox[1], '-', color='orangered', linewidth=4)
		# Overlay map image
		# self.toggleMap.imshow(plt.imread('icon/' + self.scene_name + '.png'), extent=[self.scene_bbox[0][0], self.scene_bbox[0][1], self.scene_bbox[1][3], self.scene_bbox[1][4]])

		# Setup plot parameters
		self.toggleMap.set_xticks(np.arange(np.floor(min(self.scene_bbox[0])/self.grid_size), np.ceil(max(self.scene_bbox[0])/self.grid_size)+1, 1) * self.grid_size)
		self.toggleMap.set_yticks(np.arange(np.floor(min(self.scene_bbox[1])/self.grid_size), np.ceil(max(self.scene_bbox[1])/self.grid_size)+1, 1) * self.grid_size)
		self.toggleMap.set_xticklabels(np.arange(np.floor(min(self.scene_bbox[0])/self.grid_size), np.ceil(max(self.scene_bbox[0])/self.grid_size)+1, 1) * self.grid_size, rotation=90)
		self.toggleMap.set_xlabel("x coordnates, [m]")
		self.toggleMap.set_ylabel("z coordnates, [m]")
		self.toggleMap.set_xlim(min(self.scene_bbox[0])-self.grid_size, max(self.scene_bbox[0])+self.grid_size)
		self.toggleMap.set_ylim(min(self.scene_bbox[1])-self.grid_size, max(self.scene_bbox[1])+self.grid_size)
		self.toggleMap.set_aspect('equal', 'box')
		# plt.gca().set_aspect('equal', adjustable='box')

		# Plot nodes
		if show_nodes and self.scene_name in NODES:
			nodes = self.add_nodes()
			self.toggleMap.plot(nodes[0], nodes[1], 'o', color="None",
						        markersize=5, linewidth=4,
						        markerfacecolor='red',
						        markeredgecolor="None",
						        markeredgewidth=2)
			if show_edges:
				self.add_edges(NODES[self.scene_name])

		# plt.show(block=False)
		# ----------------------------------------------------------------------
		# Plot map and agent motion in real time
		# ----------------------------------------------------------------------
		is_initial_map = True
		rob_icon = Image.open('icon/robot.png')
		while True:
			info = self.multithread_node['client'].recv()
			if not is_initial_map:
				rob.remove()
				del rob
				wedge_cur.remove()
				del wedge_cur

			# plot robot icon
			scale = 2*self.grid_size
			pose_cur = info['cur_pose']

			rob = self.toggleMap.imshow(rob_icon.rotate(-pose_cur[2]), extent=[pose_cur[0] - scale, pose_cur[0] + scale, pose_cur[1] - scale, pose_cur[1] + scale])
			# plot current robot field of view
			wedge_cur = Wedge((pose_cur[0], pose_cur[1]), 1.2*self.grid_size, - pose_cur[2] + 90 - 60, - pose_cur[2] + 90 + 60, width=self.grid_size, color='red')
			self.toggleMap.add_patch(wedge_cur)

			plt.show(block=False)
			plt.pause(1)

			self.multithread_node['comfirmed'].value = 1
			is_initial_map = False

		plt.show()
