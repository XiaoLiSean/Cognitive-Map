from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.lines import Line2D
from lib.params import NODES, ADJACENT_NODES_SHIFT_GRID, VISBILITY_DISTANCE
from termcolor import colored
from PIL import Image
import numpy as np
import time

class Plotter():
	def __init__(self, scene_name, scene_bbox, grid_size, reachable_points, objects, client=None, comfirmed=None):
		self.fig = plt.figure(figsize=(17,8), constrained_layout=True)
		gs = GridSpec(2, 3, figure=self.fig)
		self.toggleMap = self.fig.add_subplot(gs[:, 0:-1])
		self.currentView = self.fig.add_subplot(gs[0, -1])
		self.goalView = self.fig.add_subplot(gs[1, -1])
		self.scene_name = scene_name
		self.scene_bbox = scene_bbox
		self.grid_size = grid_size
		self.node_radius = VISBILITY_DISTANCE
		self.reachable_points = reachable_points
		self.objects = objects
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
					self.toggleMap.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'r--', linewidth=2.0, alpha=0.3)

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

	def show_map(self, show_nodes=True, show_edges=True, show_toggle_map=False):
		# ----------------------------------------------------------------------
		# Plot reachable points
		# ----------------------------------------------------------------------
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
		if show_toggle_map:
			self.toggleMap.imshow(plt.imread('icon/' + self.scene_name + '.png'), extent=[self.scene_bbox[0][0], self.scene_bbox[0][1], self.scene_bbox[1][3], self.scene_bbox[1][4]])
		else:
			for obj in self.objects:
				if obj['objectType'] == 'Floor':
					continue
				size = obj['axisAlignedBoundingBox']['size']
				center = obj['axisAlignedBoundingBox']['center']
				rect = Rectangle(xy = (center['x'] - size['x']*0.5, center['z'] - size['z']*0.5), width=size['x'], height=size['z'], fill=True, alpha=0.3, color='darkgray', hatch='//')
				self.toggleMap.add_patch(rect)

		# ----------------------------------------------------------------------
		# Setup plot parameters
		# ----------------------------------------------------------------------
		self.toggleMap.set_xticks(np.arange(np.floor(min(self.scene_bbox[0])/self.grid_size), np.ceil(max(self.scene_bbox[0])/self.grid_size)+1, 1) * self.grid_size)
		self.toggleMap.set_yticks(np.arange(np.floor(min(self.scene_bbox[1])/self.grid_size), np.ceil(max(self.scene_bbox[1])/self.grid_size)+1, 1) * self.grid_size)
		self.toggleMap.set_xticklabels(np.arange(np.floor(min(self.scene_bbox[0])/self.grid_size), np.ceil(max(self.scene_bbox[0])/self.grid_size)+1, 1) * self.grid_size, rotation=90)
		self.toggleMap.set_xlabel("x coordnates, [m]")
		self.toggleMap.set_ylabel("z coordnates, [m]")
		self.toggleMap.set_xlim(min(self.scene_bbox[0])-self.grid_size, max(self.scene_bbox[0])+self.grid_size)
		self.toggleMap.set_ylim(min(self.scene_bbox[1])-self.grid_size, max(self.scene_bbox[1])+self.grid_size)
		self.toggleMap.set_title("{}: Node radius {} [m]\n\n\n".format(self.scene_name, str(self.node_radius)))
		self.toggleMap.set_aspect('equal', 'box')
		legend_elements = [Wedge((0.0, 0.0), 1.2*self.grid_size, 90 - 60, 90 + 60, width=self.grid_size, color='lightskyblue', alpha=0.3, label='Topological Nodes'),
						   Line2D([0], [0], linestyle='--', color='red', lw=2, alpha=0.3, label='Topological Edges'),
						   Wedge((0.0, 0.0), 1.2*self.grid_size, 90 - 60, 90 + 60, width=self.grid_size, color='red', alpha=0.5, label='Goal Subnodes'),
						   Wedge((0.0, 0.0), 1.2*self.grid_size, 90 - 60, 90 + 60, width=self.grid_size, color='yellow', alpha=0.5, label='Robot Pose'),
						   Wedge((0.0, 0.0), 1.2*self.grid_size, 90 - 60, 90 + 60, width=self.grid_size, color='green', alpha=0.5, label='Reached Goal Subnodes')]

		self.toggleMap.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=5)
		if not show_toggle_map:
			self.toggleMap.grid(True)

		self.goalView.axis('off')
		self.goalView.set_title('Goal View')
		self.currentView.axis('off')
		self.currentView.set_title('Current Robot View')

		# ----------------------------------------------------------------------
		# Plot nodes
		# ----------------------------------------------------------------------
		if show_nodes and self.scene_name in NODES:
			nodes = self.add_nodes()
			self.toggleMap.plot(nodes[0], nodes[1], 'o', color="None",
						        markersize=5, linewidth=4,
						        markerfacecolor='red',
						        markeredgecolor="None",
						        markeredgewidth=2)
			if show_edges:
				self.add_edges(NODES[self.scene_name])

		plt.show(block=False)
		# ----------------------------------------------------------------------
		# Plot map and agent motion in real time
		# ----------------------------------------------------------------------
		is_initial_map = True
		rob_icon = Image.open('icon/robot.png')
		step_idx = 0
		while True:
			# ------------------------------------------------------------------
			# Delect robot icon for new position plot
			# ------------------------------------------------------------------
			if not is_initial_map:
				rob.remove()
				del rob
				wedge_cur.remove()
				del wedge_cur
				if info['is_reached']:
					bbox_reached_robot.remove()
					del bbox_reached_robot
					bbox_reached_goal.remove()
					del bbox_reached_goal
				else:
					wedge_goal.remove()
					del wedge_goal
					bbox_robot.remove()
					del bbox_robot
					bbox_goal.remove()
					del bbox_goal

			info = self.multithread_node['client'].recv()
			# ------------------------------------------------------------------
			# plot robot icon
			# ------------------------------------------------------------------
			scale = self.grid_size
			pose_cur = info['cur_pose']
			rob = self.toggleMap.imshow(rob_icon.rotate(-pose_cur[2]), extent=[pose_cur[0] - scale, pose_cur[0] + scale, pose_cur[1] - scale, pose_cur[1] + scale])
			# plot current robot field of view
			wedge_cur = Wedge((pose_cur[0], pose_cur[1]), 1.2*self.grid_size, - pose_cur[2] + 90 - 60, - pose_cur[2] + 90 + 60, width=self.grid_size, color='yellow', alpha=0.5)
			self.toggleMap.add_patch(wedge_cur)
			# ------------------------------------------------------------------
			# Plot Robot goal pose
			# ------------------------------------------------------------------
			pose_goal = info['goal_pose']
			# plot goal robot field of view
			if info['is_reached']:
				wedge_goal = Wedge((pose_goal[0], pose_goal[1]), 1.2*self.grid_size, - pose_goal[2] + 90 - 60, - pose_goal[2] + 90 + 60, width=self.grid_size, color='green', alpha=0.5)
			else:
				wedge_goal = Wedge((pose_goal[0], pose_goal[1]), 1.2*self.grid_size, - pose_goal[2] + 90 - 60, - pose_goal[2] + 90 + 60, width=self.grid_size, color='red', alpha=0.5)

			self.toggleMap.add_patch(wedge_goal)
			# ------------------------------------------------------------------
			# Plot Images
			# ------------------------------------------------------------------
			cur_img = self.currentView.imshow(info['cur_img'])
			goal_img = self.goalView.imshow(info['goal_img'])
			# Image.fromarray(info['cur_img']).save('pathImg/'+str(step_idx)+'current.jpg')
			# Image.fromarray(info['goal_img']).save('pathImg/'+str(step_idx)+'goal.jpg')

			width = info['cur_img'].shape[1]
			height = info['cur_img'].shape[0]
			bbox_reached_robot = Rectangle((0.0,0.0), width, height, ec='green', fill=False, linewidth=10)
			bbox_reached_goal = Rectangle((0.0,0.0), width, height, ec='green', fill=False, linewidth=10)
			bbox_robot = Rectangle((0.0,0.0), width, height, ec='yellow', fill=False, linewidth=10)
			bbox_goal = Rectangle((0.0,0.0), width, height, ec='red',  fill=False, linewidth=10)

			if info['is_reached']:
				self.currentView.add_patch(bbox_reached_robot)
				self.goalView.add_patch(bbox_reached_goal)
			else:
				self.currentView.add_patch(bbox_robot)
				self.goalView.add_patch(bbox_goal)
			# ------------------------------------------------------------------

			plt.show(block=False)
			plt.pause(0.5)

			self.multithread_node['comfirmed'].value = 1
			is_initial_map = False
			step_idx += 1

		plt.show()
