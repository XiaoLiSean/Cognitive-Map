from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from dijkstar import Graph, find_path
from distutils.util import strtobool
import numpy as np
from PIL import Image
import time
import copy
import argparse
import random
import logging
import os
import sys
sys.path.append('../Network')
from action_network import Action_network

parser = argparse.ArgumentParser()
parser.add_argument("--scene_type", type=int, default=1,  help="Choose scene type for simulation, 1 for Kitchens, 2 for Living rooms, 3 for Bedrooms, 4 for Bathrooms")
parser.add_argument("--scene_num", type=int, default=0,  help="Choose scene num for simulation, from 1 - 30")
parser.add_argument("--grid_size", type=float, default=0.25,  help="Grid size of AI2THOR simulation")
parser.add_argument("--rotation_step", type=float, default=10,  help="Rotation step of AI2THOR simulation")
parser.add_argument("--sleep_time", type=float, default=0.005,  help="Sleep time between two actions")
parser.add_argument("--save_directory", type=str, default='./data',  help="Data saving directory")
parser.add_argument("--overwrite_data", type=lambda x: bool(strtobool(x)), default=False, help="overwrite the existing data or not")
parser.add_argument("--log_level", type=int, default=2,  help="Level of showing log 1-5 where 5 is most detailed")
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False,  help="Output debug info if True")
parser.add_argument("--test_data", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting test dataset")
parser.add_argument("--AI2THOR", type=lambda x: bool(strtobool(x)), default=False, help="True for RobotTHOR false for ITHOR")


args = parser.parse_args()
print(args)

if args.scene_num == 0:
	args.scene_num = random.randint(1, 30)
scene_setting = {1: 0, 2: 200, 3: 300, 4: 400}

log_setting = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO, 5: logging.DEBUG}

logging.basicConfig(level=log_setting[args.log_level])


class Dumb_Navigetion():
	def __init__(self, AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory, overwrite_data, for_test_data=False, debug=False):
		self._map = {}
		self._point_list = []
		self._grid_size = grid_size
		self._point_num = 0
		self._sleep_time = sleep_time
		self._Agent_action = Agent_action(AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time,
			save_directory, overwrite_data, for_test_data, debug=debug)
		self._starting_point = self._Agent_action.Get_agent_position()
		self._coordinate_dict = self._Agent_action.Get_reachable_coordinate()
		self._map_searched = [False] * len(self._coordinate_dict)
		self._for_test_data = for_test_data
		self._debug = debug
		# if self._debug:
		# 	self._map_searched = [True] * len(self._coordinate_dict)
		self._build_map()
		# self._action_network = Action_network()
		self._rotate_degree_for_train = 30
		self._SPTM_like_method_try = 1800
		self._rand_step_num = 5
		self._total_action_num = 4

	def Wrap_to_degree(self, degree):
		degree_wrap = copy.deepcopy(degree)
		while degree_wrap > 360:
			degree_wrap -= 360
		while degree_wrap < 0:
			degree_wrap += 360
		return degree_wrap

	def degree_right_or_left(self, degree, compared_degree):
		smaller_one = min(degree, compared_degree)
		bigger_one = max(degree, compared_degree)
		if bigger_one - smaller_one > 180:
			if degree == smaller_one:
				return 'Right'
			if degree == bigger_one:
				return 'Left'
		else:
			if degree == smaller_one:
				return 'Left'
			if degree == bigger_one:
				return 'Right'

	def forward_or_backward(self, pre_points, moving_point):
		first_moving_vec = list(map(lambda x, y: x - y, self._point_list[pre_points[1]], self._point_list[pre_points[0]]))
		first_moving_vec_abs = list(np.abs(first_moving_vec))
		changed_coord = first_moving_vec_abs.index(max(first_moving_vec_abs))
		moving_vec = list(map(lambda x, y: x - y, self._point_list[moving_point], self._point_list[pre_points[0]]))
		if first_moving_vec[changed_coord] > 0:
			if moving_vec[changed_coord] > first_moving_vec[changed_coord] + 0.02:
				return True
			if moving_vec[changed_coord] <= first_moving_vec[changed_coord] + 0.02:
				return False
		else:
			if moving_vec[changed_coord] < first_moving_vec[changed_coord] - 0.02:
				return True
			if moving_vec[changed_coord] >= first_moving_vec[changed_coord] - 0.02:
				return False

	def Random_SPTM_like_method(self):
		for i in range(self._SPTM_like_method_try):
			pre_rand_rot_num = random.randint(0, 6)
			for _ in range(pre_rand_rot_num):
				self._Agent_action.Unit_rotate(random.choice([-1, 1]) * self._rotate_degree_for_train)
			# print(i)
			rand_point = random.randint(0, self._point_num - 1)
			pre_point = copy.deepcopy(rand_point)
			self._Agent_action.Teleport_agent(self._point_list[rand_point])
			rand_action_step = random.randint(1, self._rand_step_num)
			# rand_action_step = 3
			first_rand_action = None
			first_traj = None
			init_orientation = self.Get_agent_rotation()['y']

			for action_num in range(rand_action_step):
				action = random.randint(0, self._total_action_num - 1)
				# if action_num == 0:
				# 	action = 2
				# if action_num == 1:
				# 	action = 3
				if self._debug:
					print('action: ', action)
					print('action_num: ', action_num)
				if first_rand_action is None:
					first_rand_action = action
				if self._debug:
					print('first_rand_action first: ', first_rand_action)
				if action == 0:
					success, moving_index = self.Move_navigation_specialized(starting_point_index=pre_point)
					if first_rand_action == 3 and not self.forward_or_backward(first_traj, moving_index):
						if self._debug:
							# print('init move: ', list(map(lambda x, y: x - y, self._point_list[first_traj[1]], self._point_list[first_traj[0]])))
							# print('now move: ', list(map(lambda x, y: x - y, self._point_list[first_traj[1]], self._point_list[moving_index])))
							print('can not move forward')
						continue
					if first_rand_action == 0 and action_num == 0:
						first_traj = [rand_point, moving_index]
					if success:
						action_taken = self._Agent_action.Teleport_agent(self._point_list[moving_index], useful=True)
						pre_point = copy.deepcopy(moving_index)
						if action_taken and not self._debug:
							self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_taken])
					elif action_num == 0:
						first_rand_action = None
				if action == 1:
					if first_rand_action == 2:
						current_orientation = self.Get_agent_rotation()['y']
						if self.degree_right_or_left(self.Wrap_to_degree(current_orientation + self._rotate_degree_for_train - 1),
							self.Wrap_to_degree(init_orientation - self._rotate_degree_for_train)) == 'Right':
							if self._debug:
								print('self.Wrap_to_degree(current_orientation + self._rotate_degree_for_train): ', self.Wrap_to_degree(current_orientation + self._rotate_degree_for_train))
								print('self.Wrap_to_degree(init_orientation - self._rotate_degree_for_train): ', self.Wrap_to_degree(init_orientation - self._rotate_degree_for_train))
								print('can not turn right')
							continue
					action_taken = self._Agent_action.Unit_rotate(self._rotate_degree_for_train)
					if action_taken and not self._debug:
						self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_taken])
				if action == 2:
					if first_rand_action == 1:
						current_orientation = self.Get_agent_rotation()['y']
						if self.degree_right_or_left(self.Wrap_to_degree(current_orientation - self._rotate_degree_for_train + 1),
							self.Wrap_to_degree(init_orientation + self._rotate_degree_for_train)) == 'Left':
							if self._debug:
								print('can not turn left')
							continue
					action_taken = self._Agent_action.Unit_rotate(-self._rotate_degree_for_train)
					if action_taken and not self._debug:
						self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_taken])
				if action == 3:
					success, moving_index = self.Move_navigation_specialized(starting_point_index=pre_point, direction='backward')
					if first_rand_action == 0 and not self.forward_or_backward(first_traj, moving_index):
						if self._debug:
							# print('init move: ', list(map(lambda x, y: x - y, self._point_list[first_traj[1]], self._point_list[first_traj[0]])))
							# print('now move: ', list(map(lambda x, y: x - y, self._point_list[first_traj[1]], self._point_list[moving_index])))
							print('can not move backward')
						continue
					if first_rand_action == 3 and action_num == 0:
						first_traj = [rand_point, moving_index]
					if success:
						action_taken = self._Agent_action.Teleport_agent(self._point_list[moving_index], useful=True)
						pre_point = copy.deepcopy(moving_index)
						if action_taken and not self._debug:
							self._Agent_action._Save_RGB_label(self._Agent_action._action_type['MOVE_BACKWARD'])
					elif action_num == 0:
						first_rand_action = None
				if action_num == rand_action_step - 1:
					first_rand_action = None
				# if self._debug:
				# 	time.sleep(0.5)
				if self._debug:
					print('first_rand_action second: ', first_rand_action)
			if self._debug:
				print('---------------------------------------')
		return

	def Navigate_by_ActionNet(self, image_goal, goal_pose):
		goal_position = goal_pose['position']
		goal_rotation = goal_pose['rotation']
		if isinstance(goal_position, dict):
			goal_position = list(goal_position.values())
		if isinstance(goal_rotation, dict):
			goal_rotation = list(goal_rotation.values())
		current_position = list(self._Agent_action.Get_agent_position().values())
		current_rotation = list(self._Agent_action.Get_agent_rotation().values())
		distance_search_min = 1000
		nearest_index = -1
		for i, point in enumerate(self._point_list):
			distance_search = np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, current_position))))
			if distance_search < distance_search_min:
				distance_search_min = distance_search
				nearest_index = i
		if distance_search_min > 0.5 * self._grid_size:
			logging.error('Can not find starting point in point list')
			return
		current_point_index = nearest_index
		distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, goal_position, current_position))))
		rotation_difference = np.abs(goal_rotation[1] - current_rotation[1])
		print('goal_position: ', goal_position)
		print('goal_rotation: ', goal_rotation)
		print('current_position: ', current_position)
		print('current_rotation: ', current_rotation)
		while distance > 0.5 * self._grid_size or rotation_difference > 10:
			# Dumb_Navigetion
			image_current = Dumb_Navigetion._Agent_action.Get_frame()
			action_predict = self._action_network.predict(image_current=image_current, image_goal=image_goal)
			print('action_predict: ', action_predict.item())
			if self._debug:
				print(action_predict)
			if action_predict == 0:
				# self._Agent_action.Unit_move()
				success, moving_index = self.Move_navigation_specialized(current_point_index)
				if not moving_index == -1:
					current_point_index = moving_index
			elif action_predict == 1:
				self._Agent_action.Unit_rotate(degree=15)
			elif action_predict == 2:
				self._Agent_action.Unit_rotate(degree=-15)

			current_position = list(self._Agent_action.Get_agent_position().values())
			current_rotation = list(self._Agent_action.Get_agent_rotation().values())

			distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, goal_position, current_position))))
			rotation_difference = np.abs(goal_rotation[1] - current_rotation[1])
			print('goal_position: ', goal_position)
			print('current_position: ', current_position)
			print('goal_rotation: ', goal_rotation)
			print('current_rotation: ', current_rotation)
			print('-----------------------------------------------')
			time.sleep(1)
		return

	def Get_orientation_two_points(self, starting_point_index, goal_point_index):
		starting_point_position = self._point_list[starting_point_index]
		goal_point_position = self._point_list[goal_point_index]
		error_vector = list(map(lambda x, y: x - y, goal_point_position, starting_point_position))
		error_orientation = np.arctan2(error_vector[0], error_vector[2]) * 180 / np.pi
		if error_orientation < 0:
			error_orientation += 360
		return error_orientation

	def Move_navigation_specialized(self, starting_point_index, direction='forward'):
		current_orientation = self.Get_agent_rotation()['y']
		current_position = self._point_list[starting_point_index]
		if direction.lower() == 'forward':
			moving_direction = {0: [0, 0, 1], 1: [1, 0, 0], 2: [0, 0, -1], 3: [-1, 0, 0]}
		if direction.lower() == 'backward':
			moving_direction = {0: [0, 0, -1], 1: [-1, 0, 0], 2: [0, 0, 1], 3: [1, 0, 0]}
		moving_step = {}
		for key in list(moving_direction.keys()):
			moving_step[key] = map(lambda x: x * self._grid_size, moving_direction[key])

		heading_direction_index = int(np.floor((current_orientation - 45) / 90) + 1)
		if current_orientation < 45 or current_orientation > 315:
			heading_direction_index = 0
		moving_point = list(map(lambda x, y: x + y, current_position, moving_step[heading_direction_index]))
		connected_point_index = self._map[starting_point_index]
		nearest_point_index = -1
		distance_min = 1000
		if len(connected_point_index) > 0:
			for index in connected_point_index:
				distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, self._point_list[index], moving_point))))
				if distance < distance_min:
					distance_min = distance
					nearest_point_index = index
		else:
			logging.warning('Nowhere to go forward for no connected point')
			return (False, -1)
		if distance_min > 0.5 * self._grid_size:
			logging.warning('Nowhere to go forward')
			return (False, -1)
		# self._Agent_action.Teleport_agent(position=self._point_list[nearest_point_index], useful=True)
		return (True, nearest_point_index)

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
					if distance > self._grid_size:
						print(distance)
		return

	def Fast_traverse_map(self, goal_points_num):
		self._map_searched = [True] * len(self._coordinate_dict)
		goal_points = []
		polar_coordinate = []
		current_position = list(self._starting_point.values())
		if self._debug:
			print('current_position: ', current_position)
		for point in self._point_list:
			polar_coordinate.append([np.arctan2(point[0] - current_position[0],	point[2] - current_position[2]) / np.pi * 180,
				np.sqrt((point[0] - current_position[0])**2 + (point[2] - current_position[2])**2)])
		# print(polar_coordinate)
		polar_coordinate_copy = copy.deepcopy(polar_coordinate)
		polar_coordinate_copy = sorted(polar_coordinate_copy, key=lambda x: x[0])
		# print(polar_coordinate_copy)
		max_dis = [-1] * goal_points_num
		default_polar = [-181, -1]
		goal_points_polar_coordinate = [default_polar] * goal_points_num
		degree_division = 360.0 / goal_points_num
		for polar_point in polar_coordinate_copy:
			degree_division_num = int((polar_point[0] + 180) / degree_division - 0.0001)
			# print(degree_division_num)
			if polar_point[1] > max_dis[degree_division_num]:
				max_dis[degree_division_num] = polar_point[1]
				goal_points_polar_coordinate[degree_division_num] = polar_point
		if self._debug:
			print('goal_points_polar_coordinate: ', goal_points_polar_coordinate)
		for goal_point_polar in goal_points_polar_coordinate:
			if not goal_point_polar == default_polar:
				goal_points.append(polar_coordinate.index(goal_point_polar))
		if self._debug:
			print('goal_points', goal_points)
		for goal_point_index in goal_points:
			if self._debug:
				print('goal_point_index: ', goal_point_index)
				print('self._Agent_action.Get_agent_position(): ', self._Agent_action.Get_agent_position())
			self.Dumb_navigate(goal_index=goal_point_index)
			# self._Agent_action.Teleport_agent(self._point_list[goal_point_index])
			self.Dumb_navigate(goal_position=self._starting_point)
			# self._Agent_action.Teleport_agent(current_position)

	def Traverse_neighbor_map(self):
		for point_index in range(self._point_num):
			self._Agent_action.Teleport_agent(position=self._point_list[point_index], useful=False)
			for _ in range(int(360 / self._rotate_degree_for_train)):
				if self._debug:
					time.sleep(self._sleep_time)
					print('sleep')
				action = self._Agent_action.Unit_rotate(self._rotate_degree_for_train)
				if action and not self._debug:
					self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action])
			for _ in range(int(360 / self._rotate_degree_for_train)):
				if self._debug:
					time.sleep(self._sleep_time)
					print('sleep')
				action = self._Agent_action.Unit_rotate(-self._rotate_degree_for_train)
				if action and not self._debug:
					self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action])
		for point_index in list(self._map.keys()):
			# print('self._point_list[point_index]: ', self._point_list[point_index])
			for connected_point_index in self._map[point_index]:
				# time.sleep(1)
				self._Agent_action.Teleport_agent(position=self._point_list[point_index], useful=False)
				goal_orientation = self.Get_orientation_two_points(point_index, connected_point_index)
				current_orientation = self.Get_agent_rotation()['y']
				orientation_error = goal_orientation - current_orientation
				# print('goal_orientation: ', goal_orientation)
				# print('current_orientation: ', current_orientation)
				# while orientation_error < 0:
				# 	orientation_error += 360
				# print('orientation_error: ', orientation_error)
				self._Agent_action.Unit_rotate(orientation_error)
				# print('self.Get_agent_rotation()[]: ', self.Get_agent_rotation()['y'])
				# time.sleep(1)
				# print('self._point_list[connected_point_index]: ', self._point_list[connected_point_index])
				action = self._Agent_action.Teleport_agent(position=self._point_list[connected_point_index], useful=True)
				# time.sleep(1)
				if action and not self._debug:
					self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action])
				# print('----------------------------')
		return

	def Random_traverse_map(self, pair_num):
		self._map_searched = [True] * len(self._coordinate_dict)
		points = []
		for i in range(pair_num):
			points.append([random.randint(1, self._point_num - 1), random.randint(1, self._point_num - 1)])
		for point_pair in points:
			self._Agent_action.Teleport_agent(self._point_list[point_pair[0]])
			if not self.Dumb_navigate(self._point_list[point_pair[1]], point_pair[1]):
				logging.error('Fail traversing the map')
				break
		return

	def Dumb_traverse_map(self, complete=False):
		searching_queue = []
		searched_map = list(filter(lambda x: x == True, self._map_searched))
		searched_num_start = len(searched_map)
		while False in self._map_searched:
			if not complete:
				if searched_num_start > int(len(self._map_searched) / 5):
					break
			current_position_index = self._point_list.index(list(self._Agent_action.Get_agent_position().values()))
			if current_position_index in searching_queue:
				searching_queue.remove(current_position_index)
			for connected_point_index in self._map[current_position_index]:
				if connected_point_index not in searching_queue and not self._map_searched[connected_point_index]:
					searching_queue.insert(0, connected_point_index)
			if self._debug:
				print('searching_queue: ', searching_queue)
				print('self._map connected: ', self._map[searching_queue[0]])
				# print('self._map_searched: ', )
				for point_temp in self._map[searching_queue[0]]:
					print('searched or not: ', point_temp, self._map_searched[point_temp])
			if len(searching_queue) == 0:
				break
			if not self.Dumb_navigate(self._point_list[searching_queue[0]], searching_queue[0]):
				logging.error('Fail traversing the map')
				break
			searched_num_start += 1
		return

		# Assume goal_position is dict
	def Dumb_navigate(self, goal_position=None, goal_index=None):
		graph = Graph()
		nav_starting_point = self._Agent_action.Get_agent_position()
		nav_starting_point = list(nav_starting_point.values())
		nav_starting_point_index = None
		for point in self._point_list:
			if np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, nav_starting_point)))) < 0.25 * self._grid_size:
				nav_starting_point_index = self._point_list.index(point)
				break
		
		# nav_starting_point_index = self._point_list.index(nav_starting_point)

		if isinstance(goal_position, dict):
			goal_point = list(goal_position.values())

		goal_point_index = None
		if goal_index is None:
			for point in self._point_list:
				if np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, goal_point)))) < 0.25 * self._grid_size:
					goal_point_index = self._point_list.index(point)
					break
		else:
			goal_point_index = goal_index
		if self._debug:
			print('goal_index: ', goal_index)
			print('goal_point_index: ', goal_point_index)
		if goal_point_index is None:
			logging.error('No matching goal point in map')
			return False
		if nav_starting_point_index is None:
			logging.error('No matching starting point in map')
			print('nav_starting_point: ', nav_starting_point)
			print('self._point_list.index(point): ', self._point_list.index(nav_starting_point))
			return False
		connected_point_index = self._map[goal_point_index]
		# if self._debug:
		# 	print('connected_point_index: ', connected_point_index)
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
				return False

		for index in range(len(self._map)):
			for connected_index in range(len(self._map[index])):
				if self._map_searched[self._map[index][connected_index]]:
					graph.add_edge(index, self._map[index][connected_index], 1)
		result = find_path(graph, nav_starting_point_index, nearest_reachable_index)

		path = result.nodes
		pre_pose = None
		pre_real_pose = None
		pre_pre_real_pose = None
		pre_pre_pose = None
		for mid_point_index in range(1, len(path)):
			mid_point_pose = {'position': [], 'rotation': []}
			mid_point_pose['position'] = copy.deepcopy(self._point_list[path[mid_point_index]])
			mid_point_pose['rotation'] = [0, 0, 0]

			if not self._Agent_action.Move_toward(mid_point_pose, rotation_care=False):
				print('path: ', path)
				print('pre_pose: ', pre_pose)
				print('pre_pre_pose: ', pre_pre_pose)
				print('mid_point_pose: ', mid_point_pose)
				print('mid_point_index: ', path[mid_point_index])
				print('pre_real_pose: ', pre_real_pose)
				print('pre_pre_real_pose: ', pre_pre_real_pose)
				if pre_real_pose == pre_pre_real_pose:
					mid_point_pose = {'position': [], 'rotation': []}
					mid_point_pose['position'] = list(pre_real_pose.values())
					mid_point_pose['rotation'] = [0, 0, 0]
					if not self._Agent_action.Move_toward(pre_pose, rotation_care=False, debug=True):
						print('still can not move')
					print('self._Agent_action.Get_agent_position(): ', self._Agent_action.Get_agent_position())
				return False
			pre_pre_pose = copy.deepcopy(pre_pose)
			pre_pose = copy.deepcopy(mid_point_pose)
			pre_pre_real_pose = copy.deepcopy(pre_real_pose)
			pre_real_pose = self._Agent_action.Get_agent_position()
			

		if self._debug:
			print('not moving by path-----------')
			print('self._point_list[goal_point_index]: ', self._point_list[goal_point_index])
		if not goal_in_existing_map:
			if not self._Agent_action.Move_toward({'position': copy.deepcopy(self._point_list[goal_point_index]), 'rotation': [0, 0, 0]}, rotation_care=False):
				return False
			self._map_searched[goal_point_index] = True
		if self._debug:
				time.sleep(self._sleep_time)
				print('--------------------------------------------------------')
		return True


class Agent_action():
	def __init__(self, AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory, overwrite_data, for_test_data=False, debug=False):
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._grid_size = grid_size
		self._rotation_step = rotation_step
		self._sleep_time = sleep_time
		self._AI2THOR = AI2THOR
		self._for_test_data = for_test_data
		if self._AI2THOR:
			if not self._for_test_data:
				self._scene_name = 'FloorPlan_Train' + str(self._scene_type) + '_' + str(self._scene_num)
			else:
				self._scene_name = 'FloorPlan_Val' + str(self._scene_type) + '_' + str(self._scene_num)
			print(self._scene_name)
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, fieldOfView=120, agentMode='bot')
		else:
			self._scene_name = 'FloorPlan' + str(scene_setting[self._scene_type] + self._scene_num)
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, fieldOfView=120)
		# self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, fieldOfView=120)
		self._save_directory = save_directory
		
		self._overwrite_data = overwrite_data
		self._event = self._controller.step('Pass')
		self._start_time = time.time()
		
		self._debug = debug
		self._action_label_text_file = None
		self._action_type = {'INVALID_ACTION': -1, 'MOVE_FORWARD': 0, 'TURN_RIGHT': 1, 'TURN_LEFT': 2, 'MOVE_BACKWARD': 3}
		self._pre_image_name = None

	def Update_event(self):
		self._event = self._controller.step('Pass')

	def Open_close_label_text(self):
		if self._for_test_data:
			self._save_directory = self._save_directory + '/test_dataset'
		if self._action_label_text_file is None:
			if not os.path.exists(self._save_directory):
				os.makedirs(self._save_directory)
			if  self._overwrite_data:
				self._action_label_text_file = open(self._save_directory + '/action.txt', 'w')
			else:
				self._action_label_text_file = open(self._save_directory + '/action.txt', 'a')
				self._action_label_text_file.seek(0, 2)
		else:
			self._action_label_text_file.close()

	def _Save_RGB_label(self, action):

		self.Update_event()
		RGB_file_name = str(time.time() - self._start_time)
		# print('action: ', action, RGB_file_name)
		frame = self._event.frame
		img = Image.fromarray(frame, 'RGB')
		if not os.path.exists(self._save_directory + '/images'):
			os.makedirs(self._save_directory + '/images')
		if not os.path.exists(self._save_directory + '/images/' + self._scene_name):
			os.makedirs(self._save_directory + '/images/' + self._scene_name)
		img.save(self._save_directory + '/images/' + self._scene_name + '/' + RGB_file_name + '.png')

		if self._action_label_text_file is None:
			logging.error('Action label file is not opened')
			return
		if isinstance(action, int):
			self._action_label_text_file.write('/images/' + self._scene_name +
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

	def Get_frame(self):
		self.Update_event()
		return self._event.frame

	def Teleport_agent(self, position, useful=False):
		self.Update_event()
		if isinstance(position, dict):
			position_list = list(position.values())
		else:
			position_list = copy.deepcopy(position)
		self._event = self._controller.step(action='Teleport', x=position_list[0], y=position_list[1], z=position_list[2])
		if not self._debug and not useful:
			self._Save_RGB_label(self._action_type['INVALID_ACTION'])
		return 'MOVE_FORWARD'
		
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
		if degree_corrected > 0:
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
	def Move_toward(self, goal, rotation_care=True, debug=False):
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

		if np.linalg.norm(np.array(position_error)) > self._grid_size * 1.20:
			logging.error('{} Moving step {} greater than grid size {}'.format(agent_position, goal_position, self._grid_size))
			return False
		elif np.linalg.norm(np.array(position_error)) < self._grid_size * 0.10:
			logging.info('Moving distance {} too small'.format(position_error))
			return False

		rotate_steps = int(np.abs(rotation_error_corrected / self._rotation_step))

		for _ in range(rotate_steps):
			if self._debug:
				time.sleep(self._sleep_time)
			action = self.Unit_rotate(self._rotation_step * np.sign(rotation_error_corrected))
			if action and not self._debug:
				self._Save_RGB_label(self._action_type[action])
		action = self.Unit_rotate((rotation_error_corrected - rotate_steps * self._rotation_step * np.sign(rotation_error_corrected)))
		if action and not self._debug:
			self._Save_RGB_label(self._action_type[action])

		if self._debug:
				time.sleep(self._sleep_time)
		if debug:
			print('debug agent_position: ', self.Get_agent_position())
		# action = self.Unit_move()
		action = self.Teleport_agent(goal_position, useful=True)
		# print(action)
		if debug:
			print('debug agent_position: ', self.Get_agent_position())
		if action and not self._debug:
			# print('action')
			self._Save_RGB_label(self._action_type[action])

		if not rotation_care:
			return True

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
				print('sleep')
			action = self.Unit_rotate(self._rotation_step * np.sign(rotation_error_corrected))
			if action and not self._debug:
				self._Save_RGB_label(self._action_type[action])
		action = self.Unit_rotate((rotation_error_corrected - rotate_steps * self._rotation_step * np.sign(rotation_error_corrected)))
		if action and not self._debug:
			self._Save_RGB_label(self._action_type[action])

		return True


if __name__ == '__main__':
	# controller = Controller(scene='FloorPlan_Train1_1', agentMode='bot')
	# time.sleep(10)
	# exit()
	Dumb_Navigetion = Dumb_Navigetion(args.AI2THOR, args.scene_type, args.scene_num, args.grid_size,
		args.rotation_step, args.sleep_time, args.save_directory, overwrite_data=args.overwrite_data, for_test_data=args.test_data, debug=args.debug)
	# action_network = Action_network()
	# print(Dumb_Navigetion.degree_right_or_left(Dumb_Navigetion.Wrap_to_degree(350 - 30), 20))

	reach = Dumb_Navigetion._Agent_action.Get_reachable_coordinate()
	print('reach: ', len(reach))
	# print(Dumb_Navigetion._Agent_action.Teleport_agent(Dumb_Navigetion._point_list[0]))
	# exit()
	# frame = Dumb_Navigetion._Agent_action._event.frame
	# current_frame = copy.deepcopy(frame)
	# Dumb_Navigetion._Agent_action.Unit_rotate(30)
	# print('Get_agent_position(): ', Dumb_Navigetion.Get_agent_position())
	# Dumb_Navigetion._Agent_action.Unit_move()
	# print('---------------move--------------')
	# print('Get_agent_position(): ', Dumb_Navigetion.Get_agent_position())
	# print('Get_agent_rotation: ', Dumb_Navigetion.Get_agent_rotation())
	# Dumb_Navigetion._Agent_action.Unit_rotate(83)
	# print('-----------------rotate--------------------')
	# Dumb_Navigetion._Agent_action.Unit_move()
	# print('---------------move--------------')
	# print('Get_agent_position(): ', Dumb_Navigetion.Get_agent_position())
	# print('Get_agent_rotation: ', Dumb_Navigetion.Get_agent_rotation())
	# Dumb_Navigetion._Agent_action.Unit_rotate(90)
	# print('-----------------rotate--------------------')
	# Dumb_Navigetion._Agent_action.Unit_move()
	# print('---------------move--------------')
	# print('Get_agent_position(): ', Dumb_Navigetion.Get_agent_position())
	# print('Get_agent_rotation: ', Dumb_Navigetion.Get_agent_rotation())
	# Dumb_Navigetion._Agent_action.Unit_rotate(90)
	# print('-----------------rotate--------------------')
	# Dumb_Navigetion._Agent_action.Unit_move()
	# print('---------------move--------------')
	# print('Get_agent_position(): ', Dumb_Navigetion.Get_agent_position())
	# print('Get_agent_rotation: ', Dumb_Navigetion.Get_agent_rotation())
	# # Dumb_Navigetion._Agent_action.Unit_rotate(90)
	# # exit()
	# Dumb_Navigetion._Agent_action.Unit_rotate(15)
	# print('Get_agent_rotation: ', Dumb_Navigetion.Get_agent_rotation())
	# # time.sleep(2)
	# # _, index = Dumb_Navigetion.Move_navigation_specialized(0)
	# # print(index)
	# # time.sleep(2)
	# # _, index = Dumb_Navigetion.Move_navigation_specialized(index)
	# # time.sleep(2)
	# exit()
	# Dumb_Navigetion._Agent_action.Teleport_agent(position=reach[random.randint(1, len(reach))], useful=True)
	# Dumb_Navigetion._Agent_action.Unit_rotate(180)
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_rotate(30)

	# position = list(Dumb_Navigetion.Get_agent_position().values())
	# time.sleep(1)
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # time.sleep(1)
	# Dumb_Navigetion._Agent_action.Unit_move()
	# # time.sleep(1)
	# Dumb_Navigetion._Agent_action.Unit_rotate(60)
	# time.sleep(1)
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_move()
	
	# # Dumb_Navigetion._Agent_action.Unit_rotate(90)
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_rotate(60)
	# goal_pose = {'position': Dumb_Navigetion.Get_agent_position(), 'rotation': Dumb_Navigetion.Get_agent_rotation()}
	# Dumb_Navigetion._Agent_action.Update_event()
	# frame = Dumb_Navigetion._Agent_action._event.frame
	# time.sleep(1)
	# Dumb_Navigetion._Agent_action.Teleport_agent(position=position, useful=True)
	# Dumb_Navigetion._Agent_action.Unit_rotate(-60)

	
	# Dumb_Navigetion.Navigate_by_ActionNet(frame, goal_pose)
	# time.sleep(100)
	# exit()


	# print(Action_network(image_current=current_frame, image_goal=next_frame))
	# time.sleep(2)
	# Dumb_Navigetion._Agent_action.Teleport_agent(reach[100])
	# time.sleep(2)

	# Dumb_Navigetion._Agent_action.Update_event()
	# objects = Dumb_Navigetion._Agent_action._event.metadata['objects']
	# print(len(objects))
	# print(objects[0].keys())
	# Fridge_id = None
	# for object in objects:
	# 	if 'Fridge' in object['name']:
	# 		Fridge_id = object['objectId']
	# 		# print(object)
	# 	# pass
	# event = Dumb_Navigetion._Agent_action._controller.step('OpenObject', objectId=Fridge_id)
	# time.sleep(1)
	# event = Dumb_Navigetion._Agent_action._controller.step('CloseObject', objectId=Fridge_id)
	# time.sleep(5)
	# exit()
	# print(Dumb_Navigetion._starting_point)
	# devision = 36
	# for _ in range(devision):
	# 	Dumb_Navigetion._Agent_action.Unit_rotate(360 / devision)
	# 	Dumb_Navigetion._Agent_action.Unit_move()
	# 	print('Dumb_Navigetion._Agent_action.Get_agent_position()', Dumb_Navigetion._Agent_action.Get_agent_position())
	# 	# print()
	# 	if Dumb_Navigetion._Agent_action.Get_agent_position() == Dumb_Navigetion._starting_point:
	# 		print('did not move')
	# 	Dumb_Navigetion._Agent_action.Teleport_agent(Dumb_Navigetion._starting_point)
	# 	pass
	
	Dumb_Navigetion.Open_close_label_text()
	# Dumb_Navigetion.Traverse_neighbor_map()
	# Dumb_Navigetion.Dumb_traverse_map()
	Dumb_Navigetion.Random_SPTM_like_method()
	# Dumb_Navigetion.Random_traverse_map(pair_num=7)
	# Dumb_Navigetion.Fast_traverse_map(goal_points_num=24)
	# position = Dumb_Navigetion.Get_agent_position()
	# ori_position = copy.deepcopy(position)
	# reach = Dumb_Navigetion._Agent_action.Get_reachable_coordinate()
	Dumb_Navigetion.Open_close_label_text()
	# position = Dumb_Navigetion.Get_agent_position()

	# time.sleep(2)
