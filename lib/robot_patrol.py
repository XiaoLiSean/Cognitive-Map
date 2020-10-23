# Module for iTHOR env set up and simple navigation
from ai2thor.controller import Controller
from termcolor import colored
from PIL import Image
from lib.params import SIM_WINDOW_HEIGHT, SIM_WINDOW_WIDTH, VISBILITY_DISTANCE, FIELD_OF_VIEW
import matplotlib.pyplot as plt
import numpy as np
import time, copy, sys, random, os

# Class for agent and nodes in simulation env
class Agent_Sim():
	def __init__(self, scene_type='Kitchen', scene_num=1, grid_size=0.25, node_radius=1.5, ToggleMapView=False, applyActionNoise=False):
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._grid_size = grid_size
		self._node_radius = node_radius

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
		self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)  # Change simulation window size

		if ToggleMapView:   # Top view of the map to see the objets layout. issue: no SG can be enerated
			self._controller.step({"action": "ToggleMapView"})

		self._event = self._controller.step('Pass')


	def update_event(self):
		self._event = self._controller.step('Pass')

	def get_agent_position(self):
		self.update_event()
		return self._event.metadata['agent']['position']

	def get_agent_rotation(self):
		self.update_event()
		return self._event.metadata['agent']['rotation']

	def get_reachable_coordinate(self):
		self._event = self._controller.step(action='GetReachablePositions')
		return self._event.metadata['actionReturn']

	def reset_scene(self, scene_type, scene_num):
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
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._scene_name = 'FloorPlan' + str(add_on + self._scene_num)
		self._controller.reset(self._scene_name)
		self.update_event()

	def save_current_fram(self, FILE_PATH, RGB_file_name):
		self.update_event()
		frame = self._event.frame
		img = Image.fromarray(frame, 'RGB')
		img.save(FILE_PATH + '/' + RGB_file_name + '.png')

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

	def coordnates_patroling(self, saving_data=False, file_path=None, grid_steps=2):
		map = self.get_reachable_coordinate()
		rotations = [dict(x=0.0, y=0.0, z=0.0),
					 dict(x=0.0, y=90.0, z=0.0),
					 dict(x=0.0, y=180.0, z=0.0),
					 dict(x=0.0, y=270.0, z=0.0)]

		# random get fractional points as node
		rand_pts_fration = 1 / (2*grid_steps + 1)**2
		print(rand_pts_fration, len(map))
		if rand_pts_fration != None:
			random.shuffle(map)
			points = copy.deepcopy(map[0:int(rand_pts_fration*len(map))])

		# get points near the node
		for node in points:
			grids = self.get_near_grids(node, step=grid_steps)
			for grid in grids:
				if grid not in points and grid in map:
					points.append(grid)

		# define file path
		file_path = file_path + '/' + self._scene_name
		if not os.path.isdir(file_path):
			os.mkdir(file_path)

		# store image
		for p in points:
			for r in rotations:
				self._controller.step(action='TeleportFull', x=p['x'], y=p['y'], z=p['z'], rotation=r)
				file_name = self._scene_name + '_' + str(p['x']) + '_' + str(p['z']) + '_' + str(r['y']) + '_' + 'end'
				if saving_data:
					self.save_current_fram(file_path, file_name)
				else:
					self.update_event()
