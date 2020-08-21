from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from dijkstar import Graph, find_path
import numpy as np
from PIL import Image
import time
import copy
import argparse
import random
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scene_type", type=int, default=1,  help="Choose scene type for simulation, 1 for Kitchens, 2 for Living rooms, 3 for Bedrooms, 4 for Bathrooms")
parser.add_argument("--scene_num", type=int, default=0,  help="Choose scene num for simulation, from 1 - 30")
parser.add_argument("--grid_size", type=float, default=0.25,  help="Grid size of AI2THOR simulation")
parser.add_argument("--rotation_step", type=float, default=10,  help="Rotation step of AI2THOR simulation")
parser.add_argument("--sleep_time", type=float, default=0.1,  help="Sleep time between two actions")
parser.add_argument("--save_directory", type=str, default='./data',  help="Data saving directory")
parser.add_argument("--log_level", type=int, default=5,  help="Level of showing log 1-5 where 5 is most detailed")

args = parser.parse_args()


if args.scene_num == 0:
	scene_num = random.randint(1, 30)
scene_setting = {1: '', 2: '2', 3: '3', 4: '4'}

log_setting = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO, 5: logging.DEBUG}

logging.basicConfig(level=log_setting[args.log_level])


class Dumb_Navigetion():
	def __init__(self, coordinate_dict, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory):
		self._coordinate_dict = coordinate_dict
		self._map = {}
		self._map_searched = [False] * len(self._coordinate_dict)
		self._point_list = []
		self._grid_size = grid_size
		self._point_num = 0
		self._build_map()
		self._Agent_action = Agent_action(scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory)
		self._starting_point = self._Agent_action.Get_agent_position()

	def _build_map(self):
		self._point_list.append(list(self._starting_point.values()))
		self._map[self._point_num] = []
		self._map_searched[self._point_num] = True
		self._point_num += 1
		for point_adding in self._coordinate_dict:
			if self._starting_point == point_adding:
				continue
			self._point_list.append([point_adding['x'], point_adding['y'], point_adding['z']])
			self._map[self._point_num] = []
			self._point_num += 1
			for point_added_index in range(self._point_num - 1):
				point_added = self._point_list[point_added_index]
				distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, point_added, self._point_list[self._point_num - 1]))))
				if distance < self._grid_size + 0.1 * self._grid_size:
					self._map[self._point_num - 1].append(point_added_index)
					self._map[point_added_index].append(self._point_num - 1)
		return

		# Assume goal_position is dict
	def Dumb_navigate(self, goal_position):
		graph = Graph()
		nav_starting_point = self._Agent_action.Get_agent_position()
		nav_starting_point = list(nav_starting_point.values())
		nav_starting_point_index = self._point_list.index(nav_starting_point)

		goal_point = list(goal_position.values())
		goal_point_index = self._point_list.index(goal_point)

		connected_point_index = self._map[goal_point_index]
		nearest_reachable_index = None
		for index in connected_point_index:
			if self._map_searched[index]:
				nearest_reachable_index = index
				break
		if nearest_reachable_index is None:
			logging.error('Can not reach the point by existing map')

		for index in range(len(self._map)):
			for connected_index in range(len(self._map[index])):
				if self._map_searched[self._map[index][connected_index]]:
					graph.add_edge(index, self._map[index][connected_index], 1)
		result = find_path(graph, nav_starting_point_index, nearest_reachable_index)

		path = result.nodes
		for mid_point_index in range(1, len(path)):
			mid_point_pose = {'position': [], 'rotation': []}
			mid_point_pose['position'] = self._point_list[mid_point_index]
			mid_point_pose['rotation'] = [0, 0, 0]
			self.__Agent_action.Move_toward(mid_point_pose, rotation_care=False)

		self._Agent_action.Move_toward({'position': self._point_list[goal_point_index], 'rotation': [0, 0, 0]}, rotation_care=False)
		return


class Agent_action():
	def __init__(self, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory):
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._grid_size = grid_size
		self._rotation_step = rotation_step
		self._sleep_time = sleep_time
		self._scene_name = 'FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num)
		self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size)
		self._save_directory = save_directory
		self._event = self._controller.step('Pass')
		self._start_time = time.time()

	def Update_event(self):
		self._event = self._controller.step('Pass')

	def Save_RGB(self):
		self.Update_event()
		frame = self._event.frame
		img = Image.fromarray(frame, 'RGB')
		if not os.path.exists(self._save_directory + '/FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num)):
			os.makedirs(self._save_directory + '/FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num))
		img.save(self._save_directory + '/FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num) +
				'/' + str(time.time() - self._start_time) + '.png')

	def Get_agent_position(self):
		self.Update_event()
		return self._event.metadata['agent']['position']

	def Get_agent_rotation(self):
		self.Update_event()
		return self._event.metadata['agent']['rotation']

	def Get_reachable_coordinate(self):
		self._event = self._controller.step(action='GetReachablePositions')
		return self._event.metadata['actionReturn']

	def Get_object(self):
		self.Update_event()
		return event.metadata['objects']

	def Unit_move(self):
		self._event = self._controller.step(action='MoveAhead')

	def Unit_rotate(self, degree):
		degree_corrected = degree
		while degree_corrected > 360:
			degree_corrected -= 360
		while degree_corrected < -360:
			degree_corrected += 360
		if degree > 0:
			self._event = self._controller.step(action='RotateRight', degrees=np.abs(degree_corrected))
		else:
			self._event = self._controller.step(action='RotateLeft', degrees=np.abs(degree_corrected))

	def Set_object_pose(self, object_name, original_position, pose):
		objects = self.Get_object()
		object_poses = copy.deepcopy(objects)
		object_name_exact = []
		nearest_name = None
		distance = 100

		goal_position = pose['position']
		goal_rotation = pose['rotation']
		if isinstance(goal_position, dict):
			goal_position = list(goal_position.values())
			goal_rotation = list(goal_rotation.values())

		target_index = -1
		for object_index in range(len(objects)):
			if object_name in objects[object_index]['name'] or object_name.capitaliz in objects[object_index]['name']:
				object_name_exact.append(objects[object_index]['name'])
				distance_list = list(objects[object_index]['position'].values())
				distance_list = list(map(lambda x, y: x - y, goal_position, distance_list))
				if np.linalg.norm(np.array(distance_list)) < distance:
					distance = np.linalg.norm(np.array(distance_list))
					nearest_name = objects[object_index]['name']
					target_index = object_index
		if target_index == -1:
			logging.error('Object {} does not exist in current scene'.format(object_name))
		object_poses[target_index]['position'] = {'x': goal_position[0], 'y': goal_position[1], 'z': goal_position[2]}
		object_poses[target_index]['rotation'] = {'x': goal_rotation[0], 'y': goal_rotation[1], 'z': goal_rotation[2]}

		self._event = self._controller.step(action='SetObjectPoses', objectPoses=object_poses)

		return

	# Assume goal is {'position': position, 'rotation': rotation} where position and rotation are dict or list
	def Move_toward(self, goal, rotation_care=True):
		self.Update_event()
		agent_position = self.get_agent_position()
		agent_rotation = self.get_agent_rotation()
		agent_position = list(agent_position.values())
		agent_rotation = list(agent_rotation.values())
		goal_position = goal['position']
		goal_rotation = goal['rotation']
		if isinstance(goal_position, dict):
			goal_position = list(goal_position.values())
			goal_rotation = list(goal_position.values())
		heading_angle = np.arctan2((agent_position[0] - goal_position[0]) / (agent_position[2] - goal_position[2])) * 180 / np.pi
		heading_angle_list = agent_rotation
		heading_angle_list[1] = heading_angle
		position_error = list(map(lambda x, y: np.abs(x - y), goal_position,  agent_position))
		rotation_error = list(map(lambda x, y: np.abs(x - y), heading_angle_list,  agent_rotation))

		rotation_error_corrected = max(rotation_error)
		while rotation_error_corrected > 360:
			rotation_error_corrected -= 360
		while rotation_error_corrected < -360:
			rotation_error_corrected += 360

		if position_error > self._grid_size:
			logging.error('Moving step {} greater than grid size {}'.format(position_error, self._grid_size))

		rotate_steps = int(rotation_error_corrected / self._rotation_step)
		for _ in range(rotate_steps):
			self.Unit_rotate(self._rotation_step * rotation_error_corrected / np.abs(rotation_error_corrected))
			self._Save_RGB()
		self.Unit_rotate((self._rotation_step - rotate_steps * self._rotation_step) *
						rotation_error_corrected / np.abs(rotation_error_corrected))
		self._Save_RGB()

		self.Unit_move()
		self._Save_RGB()

		if not rotation_care:
			return

		self.Update_event()
		agent_rotation = self.get_agent_rotation()
		agent_rotation = list(agent_rotation.values())
		rotation_error = list(map(lambda x, y: np.abs(x - y), goal_rotation,  agent_rotation))
		rotation_error_corrected = max(rotation_error)
		while rotation_error_corrected > 360:
			rotation_error_corrected -= 360
		while rotation_error_corrected < -360:
			rotation_error_corrected += 360
		rotate_steps = int(rotation_error_corrected / self._rotation_step)
		for _ in range(rotate_steps):
			self.Unit_rotate(self._rotation_step * rotation_error_corrected / np.abs(rotation_error_corrected))
			self._Save_RGB()
		self.Unit_rotate((self._rotation_step - rotate_steps * self._rotation_step) *
						rotation_error_corrected / np.abs(rotation_error_corrected))
		self._Save_RGB()
		
		return


if __name__ == '__main__':
	controller = Controller(scene='FloorPlan28', agentControllerType='physics')
	event = controller.step('Pass')
	test = event.metadata['objects']
	floor_id = None
	for i in range(len(test)):
		print(i)
		if test[i]['objectType'] == 'CounterTop':
			floor_id = test[i]['objectId']
			# print('test[i][Receptacle]: ', test[i])
			print('floor_id: ', floor_id)
			event = controller.step('GetSpawnCoordinatesAboveReceptacle', objectId=floor_id, anywhere=True)
			test_position = event.metadata['actionReturn']
			print('test_position', test_position)
			break