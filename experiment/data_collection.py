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
parser.add_argument("--sleep_time", type=float, default=0.05,  help="Sleep time between two actions")
parser.add_argument("--save_directory", type=str, default='./data',  help="Data saving directory")
parser.add_argument("--log_level", type=int, default=5,  help="Level of showing log 1-5 where 5 is most detailed")

args = parser.parse_args()


if args.scene_num == 0:
	args.scene_num = random.randint(1, 30)
scene_setting = {1: '', 2: '2', 3: '3', 4: '4'}

log_setting = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO, 5: logging.DEBUG}

logging.basicConfig(level=log_setting[args.log_level])


class Dumb_Navigetion():
	def __init__(self, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory, debug=False):
		self._map = {}
		self._point_list = []
		self._grid_size = grid_size
		self._point_num = 0
		self._Agent_action = Agent_action(scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory, debug=debug)
		self._starting_point = self._Agent_action.Get_agent_position()
		self._coordinate_dict = self._Agent_action.Get_reachable_coordinate()
		self._map_searched = [False] * len(self._coordinate_dict)
		self._debug = debug
		if self._debug:
			self._map_searched = [True] * len(self._coordinate_dict)
		self._build_map()

	def Open_close_label_text(self):
		return self._Agent_action.Open_close_label_text()

	def Get_agent_position(self):
		return self._Agent_action.Get_agent_position()

	def Get_agent_rotation(self):
		return self._Agent_action.Get_agent_rotation()

	def _build_map(self):
		self._point_list.append(list(self._starting_point.values()))
		self._map[self._point_num] = []
		self._map_searched[self._point_num] = True
		self._point_num += 1
		for point_adding in self._coordinate_dict:
			if self._starting_point == point_adding:
				continue
			self._point_list.append(list(point_adding.values()))
			self._point_num += 1
			self._map[self._point_num - 1] = []
			
			for point_added_index in range(self._point_num - 1):
				point_added = self._point_list[point_added_index]
				distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, point_added, self._point_list[self._point_num - 1]))))
				
				if distance < self._grid_size + 0.03 * self._grid_size:
					# if self._debug:
					# 	print('distance:', distance)
					# 	print('point_added:', point_added)
					# 	print('self._point_list[self._point_num - 1]:', self._point_list[self._point_num - 1])
					# 	print('--------')
					self._map[self._point_num - 1].append(point_added_index)
					self._map[point_added_index].append(self._point_num - 1)
		return

		# Assume goal_position is dict
	def Dumb_navigate(self, goal_position):
		graph = Graph()
		nav_starting_point = self._Agent_action.Get_agent_position()
		nav_starting_point = list(nav_starting_point.values())
		for point in self._point_list:
			if np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, nav_starting_point)))) < 0.25 * self._grid_size:
				nav_starting_point_index = self._point_list.index(point)
				break
		# nav_starting_point_index = self._point_list.index(nav_starting_point)

		if isinstance(goal_position, dict):
			goal_point = list(goal_position.values())

		goal_point_index = None
		for point in self._point_list:
			if np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, goal_point)))) < 0.25 * self._grid_size:
				goal_point_index = self._point_list.index(point)
				break
		if goal_point_index is None or nav_starting_point_index is None:
			logging.error('No matching point in map')
			return

		connected_point_index = self._map[goal_point_index]
		nearest_reachable_index = None
		goal_in_existing_map = False
		if self._map_searched[goal_point_index]:
			nearest_reachable_index = goal_point_index
			goal_in_existing_map = True
		else:
			for index in connected_point_index:
				if self._map_searched[index]:
					nearest_reachable_index = index
					break
			if nearest_reachable_index is None:
				logging.error('Can not reach the point by existing map')
				return

		for index in range(len(self._map)):
			for connected_index in range(len(self._map[index])):
				if self._map_searched[self._map[index][connected_index]]:
					graph.add_edge(index, self._map[index][connected_index], 1)
		result = find_path(graph, nav_starting_point_index, nearest_reachable_index)

		path = result.nodes

		for mid_point_index in range(1, len(path)):
			mid_point_pose = {'position': [], 'rotation': []}
			mid_point_pose['position'] = copy.deepcopy(self._point_list[path[mid_point_index]])
			mid_point_pose['rotation'] = [0, 0, 0]
			self._Agent_action.Move_toward(mid_point_pose, rotation_care=False)

		if self._debug:
			print('not moving by path-----------')
			print('self._point_list[goal_point_index]: ', self._point_list[goal_point_index])
		if not goal_in_existing_map:
			self._Agent_action.Move_toward({'position': copy.deepcopy(self._point_list[goal_point_index]), 'rotation': [0, 0, 0]}, rotation_care=False)
			self._map_searched[goal_point_index] = True
		if self._debug:
				time.sleep(1)
				print('--------------------------------------------------------')
		return


class Agent_action():
	def __init__(self, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory, debug=False):
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
		self._debug = debug
		self._action_label_text_file = None
		self._action_type = {'MOVE_FORWARD': 1, 'STAY_IDLE' :2, 'TURN_RIGHT' :3, 'TURN_LEFT': 4}

	def Update_event(self):
		self._event = self._controller.step('Pass')

	def Open_close_label_text(self):
		if self._action_label_text_file is None:
			# if not os.path.exists(self._save_directory):
			# 	os.makedirs(self._save_directory)
			self._action_label_text_file = open(self._save_directory + '/action.txt', 'w')
		else:
			self._action_label_text_file.close()

	def _Save_RGB_label(self, action):
		self.Update_event()
		RGB_file_name = str(time.time() - self._start_time)
		frame = self._event.frame
		img = Image.fromarray(frame, 'RGB')
		if not os.path.exists(self._save_directory + '/images'):
			os.makedirs(self._save_directory + '/images')
		if not os.path.exists(self._save_directory + '/images' + '/FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num)):
			os.makedirs(self._save_directory + '/images' + '/FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num))
		img.save(self._save_directory + '/images' + '/FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num) +
				'/' + RGB_file_name + '.png')

		if self._action_label_text_file is None:
			logging.error('Action label file is not opened')
			return
		if action:
			self._action_label_text_file.write('/images/FloorPlan' + scene_setting[self._scene_type] + str(self._scene_num) +
				'/' + RGB_file_name + '.png' + ' ' + str(action) + '\n')


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
		return self._event.metadata['objects']

	def Unit_move(self):
		self._event = self._controller.step(action='MoveAhead')
		return 'MOVE_FORWARD'

	def Unit_rotate(self, degree):
		if np.abs(degree) < 2:
			return None
		degree_corrected = degree
		while degree_corrected > 180:
			degree_corrected -= 360
		while degree_corrected < -180:
			degree_corrected += 360
		if degree > 0:
			self._event = self._controller.step(action='RotateRight', degrees=np.abs(degree_corrected))
			return 'TURN_RIGHT'
		else:
			self._event = self._controller.step(action='RotateLeft', degrees=np.abs(degree_corrected))
			return 'TURN_LEFT'

	def Set_object_pose(self, object_name, original_position, pose):
		objects = self.Get_object()
		object_poses = copy.deepcopy(objects)
		object_name_exact = []
		nearest_name = None
		distance = np.inf

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
		agent_position = self.Get_agent_position()
		agent_rotation = self.Get_agent_rotation()

		agent_position = list(agent_position.values())
		agent_rotation = list(agent_rotation.values())

		goal_position = goal['position']
		goal_rotation = goal['rotation']

		if isinstance(goal_position, dict):
			goal_position = list(goal_position.values())
			goal_rotation = list(goal_rotation.values())
		heading_angle = np.arctan2((goal_position[0] - agent_position[0]), (goal_position[2] - agent_position[2])) * 180 / np.pi
		heading_angle_list = copy.deepcopy(agent_rotation)
		heading_angle_list[1] = heading_angle

		position_error = list(map(lambda x, y: np.abs(x - y), goal_position,  agent_position))
		rotation_error = list(map(lambda x, y: x - y, heading_angle_list,  agent_rotation))
		rotation_error_abs = list(map(lambda x: np.abs(x), rotation_error))

		rotation_error_corrected = rotation_error[rotation_error_abs.index(max(rotation_error_abs))]
		while rotation_error_corrected > 180:
			rotation_error_corrected -= 360
		while rotation_error_corrected < -180:
			rotation_error_corrected += 360

		if np.linalg.norm(np.array(position_error)) > self._grid_size * 1.10:
			logging.error('Moving step {} greater than grid size {}'.format(position_error, self._grid_size))
			return
		elif np.linalg.norm(np.array(position_error)) < self._grid_size * 0.10:
			logging.info('Moving distance {} too small'.format(position_error))
			return

		rotate_steps = int(np.abs(rotation_error_corrected / self._rotation_step))

		for _ in range(rotate_steps):
			if self._debug:
				time.sleep(self._sleep_time)
			action = self.Unit_rotate(self._rotation_step * np.sign(rotation_error_corrected))
			if action:
				self._Save_RGB_label(self._action_type[action])
		action = self.Unit_rotate((rotation_error_corrected - rotate_steps * self._rotation_step * np.sign(rotation_error_corrected)))
		if action:
			self._Save_RGB_label(self._action_type[action])

		if self._debug:
				time.sleep(self._sleep_time)
		action = self.Unit_move()
		if action:
			self._Save_RGB_label(self._action_type[action])

		if not rotation_care:
			return

		self.Update_event()
		agent_rotation = self.Get_agent_rotation()
		agent_rotation = list(agent_rotation.values())
		rotation_error = list(map(lambda x, y: x - y, goal_rotation,  agent_rotation))
		rotation_error_abs = list(map(lambda x: np.abs(x), rotation_error))

		rotation_error_corrected = rotation_error[rotation_error_abs.index(max(rotation_error_abs))]
		while rotation_error_corrected > 180:
			rotation_error_corrected -= 360
		while rotation_error_corrected < -180:
			rotation_error_corrected += 360
		rotate_steps = int(np.abs(rotation_error_corrected / self._rotation_step))

		if self._debug:
			print('heading_angle_list: ', heading_angle_list)
			print('agent_rotation: ', agent_rotation)
			print('rotation_error: ', rotation_error)
			print('rotation_error_corrected: ', rotation_error_corrected)
			print('rotate_steps: ', rotate_steps)

		for _ in range(rotate_steps):
			if self._debug:
				time.sleep(self._sleep_time)
			action = self.Unit_rotate(self._rotation_step * np.sign(rotation_error_corrected))
			if action:
				self._Save_RGB_label(self._action_type[action])
		action = self.Unit_rotate((rotation_error_corrected - rotate_steps * self._rotation_step * np.sign(rotation_error_corrected)))
		if action:
			self._Save_RGB_label(self._action_type[action])
		
		return


if __name__ == '__main__':
	Dumb_Navigetion = Dumb_Navigetion(args.scene_type, args.scene_num, args.grid_size,
		args.rotation_step, args.sleep_time, args.save_directory, debug=True)
	Dumb_Navigetion.Open_close_label_text()
	position = Dumb_Navigetion.Get_agent_position()
	ori_position = copy.deepcopy(position)
	reach = Dumb_Navigetion._Agent_action.Get_reachable_coordinate()
	# print(reach)
	position = reach[random.randint(int(len(reach) / 3), len(reach))]
	Dumb_Navigetion.Dumb_navigate(position)
	Dumb_Navigetion.Open_close_label_text()
	time.sleep(2)