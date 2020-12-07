# Module for iTHOR env set up and simple navigation
from ai2thor.controller import Controller
from termcolor import colored
from PIL import Image
from math import floor, ceil
from matplotlib.patches import Circle, Rectangle
from lib.params import SIM_WINDOW_HEIGHT, SIM_WINDOW_WIDTH, VISBILITY_DISTANCE, FIELD_OF_VIEW, NODES, ADJACENT_NODES_SHIFT_GRID, DOOR_NODE
from lib.scene_graph_generation import Scene_Graph
from lib.object_dynamics import shuffle_scene_layout
import matplotlib.pyplot as plt
import numpy as np
import time, copy, sys, random, os

# Class for agent and nodes in simulation env
class Agent_Sim():
	def __init__(self, scene_type='Kitchen', scene_num=1, grid_size=0.25, node_radius=VISBILITY_DISTANCE, default_resol=True, ToggleMapView=False, applyActionNoise=False):
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._grid_size = grid_size
		self._node_radius = node_radius
		self._SG = Scene_Graph()

		# Kitchens: FloorPlan1 - FloorPlan30
		# Living rooms: FloorPlan201 - FloorPlan230
		# Bedrooms: FloorPlan301 - FloorPlan330
		# Bathrooms: FloorPLan401 - FloorPlan430

		if (scene_num<1) or (scene_num>30):
			sys.stderr.write(colored('ERROR: ','red')
							 + "Expect scene_num within [1,30] while get '{}'\n".format(scene_num))

		if scene_type == 'Kitchen':
			add_on = 0
		elif scene_type == 'Living room':
			add_on = 200
		elif scene_type == 'Bedroom':
			add_on = 300
		elif scene_type == 'Bathroom':
			add_on = 400
		else:
			sys.stderr.write(colored('ERROR: ','red')
							 + "Expect scene_type 'Kitchen', 'Living room', 'Bedroom' or 'Bathroom' while get '{}'\n".format(scene_type))
			sys.exit(1)


		self._scene_name = 'FloorPlan' + str(add_on + self._scene_num)

		self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE, fieldOfView=FIELD_OF_VIEW, applyActionNoise=applyActionNoise)

		if not default_resol:
			self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)  # Change simulation window size

		if ToggleMapView:   # Top view of the map to see the objets layout. issue: no SG can be enerated
			self._controller.step({"action": "ToggleMapView"})
			self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)  # Change simulation window size

		self._event = self._controller.step('Pass')


	def update_event(self):
		self._event = self._controller.step('Pass')
		self._SG.reset()
		self._SG.update_from_data([obj for obj in self._event.metadata['objects'] if obj['visible']])
		# self._SG.visualize_SG()

	def get_agent_position(self):
		self.update_event()
		return self._event.metadata['agent']['position']

	def get_agent_rotation(self):
		self.update_event()
		return self._event.metadata['agent']['rotation']

	def get_reachable_coordinate(self):
		self._event = self._controller.step(action='GetReachablePositions')
		return self._event.metadata['actionReturn']

	def reset_scene(self, scene_type, scene_num, ToggleMapView=False, Show_doorway=False, shore_toggle_map=False):
		if scene_type == 'Kitchen':
		# This code is used to store map in toggle view to visualize the task flow
		# self.store_toggled_map()

			add_on = 0
		elif scene_type == 'Living room':
			add_on = 200
		elif scene_type == 'Bedroom':
			add_on = 300
		elif scene_type == 'Bathroom':
			add_on = 400
		else:
			sys.stderr.write(colored('ERROR: ','red')
							 + "Expect scene_type 'Kitchen', 'Living room', 'Bedroom' or 'Bathroom' while get '{}'\n".format(scene_type))
			sys.exit(1)
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._scene_name = 'FloorPlan' + str(add_on + self._scene_num)
		self._controller.reset(self._scene_name)

		if Show_doorway and self._scene_name in DOOR_NODE:
			if DOOR_NODE[self._scene_name] != None:
				node_idx = DOOR_NODE[self._scene_name][0]
				node = NODES[self._scene_name][node_idx]
				subnode = DOOR_NODE[self._scene_name][1]
				self._controller.step(action='TeleportFull', x=node[0], y=self.get_agent_position()['y'], z=node[1], rotation=dict(x=0.0, y=subnode, z=0.0))
				time.sleep(5)

		if ToggleMapView:   # Top view of the map to see the objets layout. issue: no SG can be enerated
			self._controller.step({"action": "ToggleMapView"})
			# self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)  # Change simulation window size
			if shore_toggle_map:
				self._controller.step({"action": "Initialize", "makeAgentsVisible": False,})
				map = self.get_current_fram()
				map.save('./icon/' + self._scene_name + '.png')

		self.update_event()

	# --------------------------------------------------------------------------
	'''
	Following functions is used to visualize map of a certain scene for
	manually topological map construction
	'''
	# --------------------------------------------------------------------------
	def get_floor_bbox(self):
		floor = [obj for obj in self._event.metadata['objects'] if obj['objectType'] == 'Floor'][0]
		data = floor['axisAlignedBoundingBox']
		center_x = data['center']['x']
		center_z = data['center']['z']
		size_x = data['size']['x']
		size_z = data['size']['z']

		bbox_x = [center_x-size_x*0.5, center_x+size_x*0.5, center_x+size_x*0.5, center_x-size_x*0.5, center_x-size_x*0.5]
		bbox_z = [center_z+size_z*0.5, center_z+size_z*0.5, center_z-size_z*0.5, center_z-size_z*0.5, center_z+size_z*0.5]

		return (bbox_x, bbox_z)

	def get_scene_bbox(self):
		data = self._event.metadata['sceneBounds']
		center_x = data['center']['x']
		center_z = data['center']['z']
		size_x = data['size']['x']
		size_z = data['size']['z']

		bbox_x = [center_x-size_x*0.5, center_x+size_x*0.5, center_x+size_x*0.5, center_x-size_x*0.5, center_x-size_x*0.5]
		bbox_z = [center_z+size_z*0.5, center_z+size_z*0.5, center_z-size_z*0.5, center_z-size_z*0.5, center_z+size_z*0.5]

		return (bbox_x, bbox_z)

	def add_nodes(self, ax):
		nodes_x = []
		nodes_y = []
		points = NODES[self._scene_name]
		for idx, p in enumerate(points):
			circ = Circle(xy = (p[0], p[1]), radius=self._node_radius, alpha=0.3)
			ax.add_patch(circ)
			nodes_x.append(p[0])
			nodes_y.append(p[1])
			ax.text(p[0], p[1], str(idx))

		return (nodes_x, nodes_y)

	def is_node(self, pose, threshold=1e-6):
		is_node = False
		node_identity = -1
		for node_i, node in enumerate(NODES[self._scene_name]):
			dis_sq = (pose[0] - node[0])**2 + (pose[1] - node[1])**2
			if dis_sq < threshold**2:
				node_identity = node_i
				is_node = True
		return is_node, node_identity


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


	def add_edges(self, nodes, ax=None):
		edges = []
		# Iterature through nodes to generate edges
		for i in range(len(nodes)-1):
			node_i = nodes[i]
			for j in range(i+1, len(nodes)):
				node_j = nodes[j]
				diff = np.abs(np.array(node_i) - np.array(node_j))
				is_edge = False

				if diff[0] < self._node_radius:
					if diff[1] <= ADJACENT_NODES_SHIFT_GRID * self._grid_size:
						is_edge = self.is_reachable(node_i, node_j)
						if is_edge:
							cost = (diff[0] + diff[1]) / self._grid_size
							edges.append((node_i, node_j, int(cost)))


				if diff[1] < self._node_radius:
					if diff[0] <= ADJACENT_NODES_SHIFT_GRID * self._grid_size:
						is_edge = self.is_reachable(node_i, node_j)
						if is_edge:
							cost = (diff[0] + diff[1]) / self._grid_size
							edges.append((node_i, node_j, int(cost)))


				if is_edge and ax != None:
					ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'r--', linewidth=2.0)
					ax.text((node_i[0]+node_j[0]) / 2.0, (node_i[1]+node_j[1]) / 2.0, int(cost), size=8,
					        ha="center", va="center",
					        bbox=dict(boxstyle="round",
					                  ec=(1., 0.5, 0.5),
					                  fc=(1., 0.8, 0.8),
					                  )
					        )


		return edges


	def show_map(self, show_nodes=False, show_edges=False):
		self.update_event()
		# Plot reachable points
		points = self.get_reachable_coordinate()
		X = [p['x'] for p in points]
		Z = [p['z'] for p in points]

		fig, ax = plt.subplots()

		plt.plot(X, Z, 'o', color='lightskyblue',
		         markersize=5, linewidth=4,
		         markerfacecolor='white',
		         markeredgecolor='lightskyblue',
		         markeredgewidth=2)

		# Plot rectangle bounding the entire scene
		scene_bbox = self.get_floor_bbox()
		plt.plot(scene_bbox[0], scene_bbox[1], '-', color='orangered', linewidth=4)

		# Plot objects 2D boxs
		for obj in self._event.metadata['objects']:
			size = obj['axisAlignedBoundingBox']['size']
			center = obj['axisAlignedBoundingBox']['center']
			rect = Rectangle(xy = (center['x'] - size['x']*0.5, center['z'] - size['z']*0.5), width=size['x'], height=size['z'], fill=True, alpha=0.3, color='darkgray', hatch='//')
			ax.add_patch(rect)

		# Setup plot parameters
		plt.xticks(np.arange(floor(min(scene_bbox[0])/self._grid_size), ceil(max(scene_bbox[0])/self._grid_size)+1, 1) * self._grid_size, rotation=90)
		plt.yticks(np.arange(floor(min(scene_bbox[1])/self._grid_size), ceil(max(scene_bbox[1])/self._grid_size)+1, 1) * self._grid_size)
		plt.xlabel("x coordnates, [m]")
		plt.xlabel("z coordnates, [m]")
		plt.title("{}: Node radius {} [m]".format(self._scene_name, str(self._node_radius)))
		plt.xlim(min(scene_bbox[0])-self._grid_size, max(scene_bbox[0])+self._grid_size)
		plt.ylim(min(scene_bbox[1])-self._grid_size, max(scene_bbox[1])+self._grid_size)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.grid(True)

		# Plot nodes
		if show_nodes and self._scene_name in NODES:
			nodes = self.add_nodes(ax)
			plt.plot(nodes[0], nodes[1], 'o', color="None",
			         markersize=5, linewidth=4,
			         markerfacecolor='red',
			         markeredgecolor="None",
			         markeredgewidth=2)
			if show_edges:
				self.add_edges(NODES[self._scene_name], ax=ax)

		plt.show()

	# --------------------------------------------------------------------------
	'''
	Following functions is used to collect data with and without dynamcis
	for localization/retrieval network
	'''
	# --------------------------------------------------------------------------
	def get_current_fram(self):
		self.update_event()
		frame = self._event.frame
		img = Image.fromarray(frame, 'RGB')
		return img

	def get_current_data(self):
		img = self.get_current_fram()
		SG_data = self._SG.get_SG_as_dict()
		return img, SG_data

	def save_current_fram(self, FILE_PATH, file_name):
		img, SG_data = self.get_current_data()
		img.save(FILE_PATH + '/' + file_name + '.png')
		np.save(FILE_PATH + '/' + file_name + '.npy', SG_data)

	def get_nodes_center(self, visualization=False):
		points = self.get_reachable_coordinate()
		# Get [[x1, z1], [x2, z2]] array for all the reachable points
		points_arr = np.zeros((len(points),2))
		for idx, p in enumerate(points):
			points_arr[idx,0] = p['x']
			points_arr[idx,1] = p['z']
		# get enclose rectangle which contain all the points
		nodes_points = []
		print(len(points))
		if visualization:
			fig, ax = plt.subplots()
			ax.scatter(points_arr[:,0], points_arr[:,1], c='tab:green', s=10)
			ax.grid(True)
			plt.show()

	def get_near_grids(self, node, step=2):
		nodes = []
		for i in range(-step, step+1):
			dx = i * self._grid_size
			for j in range(-step, step+1):
				dz = j * self._grid_size
				x = node['x'] + dx
				y = node['y']
				z = node['z'] + dz
				nodes.append({'x':x, 'y':y, 'z':z})
		return nodes

	def coordnates_patroling(self, saving_data=False, file_path=None, dynamics_rounds=1, grid_steps=2, is_test=False):
		rotations = [dict(x=0.0, y=0.0, z=0.0),
					 dict(x=0.0, y=90.0, z=0.0),
					 dict(x=0.0, y=180.0, z=0.0),
					 dict(x=0.0, y=270.0, z=0.0)]

		if is_test and self._scene_name not in NODES:
			print('No nodes data available for {}'.format(self._scene_name ))
			return

		# define file path
		file_path = file_path + '/' + self._scene_name
		if not os.path.isdir(file_path):
			os.mkdir(file_path)
		else:
			# case the dataset is already exists
			# comment out to allow incrementally update the dataset
			print('Skip: ', self._scene_name)
			return

		# random get fractional points as node in train and validation scene
		rand_pts_fration = (1 / (2*grid_steps + 1)**2)


		for round in range(dynamics_rounds):
			# change obj layout for the 2-end rounds
			if round != 0:
				# change object layout
				shuffle_scene_layout(self._controller)
				self.update_event()

			# random get fractional points as node in train and validation scene
			if not is_test:
				map = self.get_reachable_coordinate()
				if rand_pts_fration != None:
					random.shuffle(map)
					points = copy.deepcopy(map[0:int(rand_pts_fration*len(map))])

				# get points near the node
				nodes_tmp = copy.deepcopy(points)
				for node in nodes_tmp:
					grids = self.get_near_grids(node, step=grid_steps)
					for grid in grids:
						if grid not in points and grid in map:
							points.append(grid)

				nodes_num = int(rand_pts_fration*len(map))

			# get nodes in test scene
			else:
				points = []
				universal_y = self.get_agent_position()['y']
				for node in NODES[self._scene_name]:
					points.append({'x': node[0], 'y': universal_y, 'z': node[1]})
				nodes_num = len(NODES[self._scene_name])


			print('{}: {}/{} round, {} data points per round with {} nodes'.format(self._scene_name, round, dynamics_rounds, len(points)*len(rotations), nodes_num))
			# store image and SG
			for p in points:
				for r in rotations:
					self._controller.step(action='TeleportFull', x=p['x'], y=p['y'], z=p['z'], rotation=r)
					file_name = 'round' + str(round) + '_' + str(p['x']) + '_' + str(p['z']) + '_' + str(r['y']) + '_' + 'end'
					if saving_data:
						self.save_current_fram(file_path, file_name)
					else:
						self.update_event()
