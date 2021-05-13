from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from dijkstar import Graph, find_path
from distutils.util import strtobool
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import time
import copy
import argparse
import random
import logging
import os
import sys
import csv
from node_generation import *
sys.path.append('../Network')
sys.path.append('..')
from action_network import Action_network
from Network.retrieval_network.retrieval_network import Retrieval_network
from experiment_config import *
from lib.params import *

SIM_WINDOW_HEIGHT = 700
SIM_WINDOW_WIDTH = 900

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
parser.add_argument("--special", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting special long range dataset")
parser.add_argument("--AI2THOR", type=lambda x: bool(strtobool(x)), default=False, help="True for RobotTHOR false for ITHOR")


args = parser.parse_args()
# print(args)

if args.scene_num == 0:
	args.scene_num = random.randint(1, 30)
scene_setting = {1: 0, 2: 200, 3: 300, 4: 400}

log_setting = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO, 5: logging.DEBUG}

logging.basicConfig(level=log_setting[args.log_level])


class Dumb_Navigetion():
	def __init__(self, scene_type, scene_num, save_directory, overwrite_data=False, AI2THOR=False, grid_size=0.25, rotation_step=90, sleep_time=0.005, for_test_data=False, debug=False, special=False, more_special=False):
		self._map = {}
		self._point_list = []
		self._grid_size = grid_size
		self._point_num = 0
		self._sleep_time = sleep_time
		self._Agent_action = Agent_action(AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time,
			save_directory, overwrite_data, for_test_data, debug=debug, special=special, more_special=more_special)
		self._starting_point = self._Agent_action.Get_agent_position()
		self._coordinate_dict = self._Agent_action.Get_reachable_coordinate()
		self._map_searched = [False] * len(self._coordinate_dict)
		self._for_test_data = for_test_data
		self._debug = debug

		self._point_x = []
		self._point_y = []
		self._build_map()
		self._action_network = Action_network()

		self._rotate_degree_for_train = 90
		self._SPTM_like_method_try = 2000
		self._Navigation_test_num = 5
		self._Navigation_max_try = 18
		self._rand_step_num = 8
		self._total_action_num = 6

		self._node_list = None
		self._nodes_pair = None
		self._connected_subnodes = None
		self._neighbor_nodes_dis = None
		self._neighbor_nodes_facing = None
		self._neighbor_nodes_facing_degree = None
		self._neighbor_nodes_coor_difference = None
		self._success_list = None

		self._agent_current_pos_index = None

		self._goal_num = 0
		self._current_num = 0
		self._step_poses = []

		self._train_data_distance = []
		self._train_data_orientation = []

		self.Set_localization_network()

	def Reset_scene(self, scene_type, scene_num):
		self._Agent_action.Reset_scene(scene_type=scene_type, scene_num=scene_num)

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

	def follow_init_move(self, pre_points, moving_point):
		first_moving_vec = list(map(lambda x, y: x - y, self._point_list[pre_points[1]], self._point_list[pre_points[0]]))
		first_moving_vec_abs = list(np.abs(first_moving_vec))
		changed_coord = first_moving_vec_abs.index(max(first_moving_vec_abs))
		moving_vec = list(map(lambda x, y: x - y, self._point_list[moving_point], self._point_list[pre_points[0]]))
		if first_moving_vec[changed_coord] > 0:
			if moving_vec[changed_coord] >= first_moving_vec[changed_coord]: # + 0.02:
				return True
			if moving_vec[changed_coord] < first_moving_vec[changed_coord]: # + 0.02:
				return False
		else:
			if moving_vec[changed_coord] <= first_moving_vec[changed_coord]: # - 0.02:
				return True
			if moving_vec[changed_coord] > first_moving_vec[changed_coord]: # - 0.02:
				return False

	def Rotate_to_degree(self, goal_degree):
		current_orientation = self.Get_agent_rotation()['y']
		orientation_error = goal_degree - current_orientation
		self._Agent_action.Unit_rotate(orientation_error)

	def Long_range_data_specified(self):
		orientations = [0, 90, 180, 270]
		action_name = {0: 'MOVE_FORWARD', 1: 'TURN_RIGHT', 2: 'TURN_LEFT', 3: 'MOVE_BACKWARD', 4: 'MOVE_RIGHT', 5: 'MOVE_LEFT'}
		move_direction = {0: 'forward', 3: 'backward'}
		explored_pos = []
		for _ in range(int(self._point_num / 8)):
			pos_index = random.randint(0, self._point_num - 1)
			while pos_index in explored_pos:
				pos_index = random.randint(0, self._point_num - 1)
			explored_pos.append(pos_index)
			# print('pos_index / self._point_num: ', len(explored_pos), self._point_num)
			for orientation in orientations:
				# print('orientation: ', orientation)
				self.Rotate_to_degree(goal_degree=orientation)
				for action_index, direction in move_direction.items():
					starting_point_index = copy.deepcopy(pos_index)
					goal_positions = {}
					for action_step in range(int(2 / self._grid_size)):
						# print('action_step: ', action_step)
						action_success, starting_point_index = self.Move_navigation_specialized(starting_point_index=starting_point_index, direction=direction, do_move=False)
						if action_success and action_step >= 4:
							goal_positions[action_step] = starting_point_index
						elif not action_success:
							break
					# print('goal_positions: ', goal_positions)
					for goal_index, goal_position_index in goal_positions.items():
						self._Agent_action.Teleport_agent(self._point_list[pos_index])
						action_taken = self._Agent_action.Teleport_agent(self._point_list[goal_position_index], useful=True)
						self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_name[action_index]])
						self._train_data_distance.append((goal_index + 1) * self._grid_size)
		return

	def Long_range_data_more_specified(self):
		orientations = [0, 90, 180, 270]
		action_name = {0: 'MOVE_FORWARD', 1: 'TURN_RIGHT', 2: 'TURN_LEFT', 3: 'MOVE_BACKWARD', 4: 'MOVE_RIGHT', 5: 'MOVE_LEFT'}
		move_direction = {0: 'forward', 3: 'backward'}
		side_move_directions = {4: 'right', 5: 'left'}
		explored_pos = []
		side_move_steps = 4
		for _ in range(int(self._point_num / 12)):
			pos_index = random.randint(0, self._point_num - 1)
			while pos_index in explored_pos:
				pos_index = random.randint(0, self._point_num - 1)
			explored_pos.append(pos_index)
			# print('pos_index / self._point_num: ', len(explored_pos), self._point_num)
			for orientation in orientations:
				# print('orientation: ', orientation)
				self.Rotate_to_degree(goal_degree=orientation)
				for action_index, direction in move_direction.items():

					starting_point_index = copy.deepcopy(pos_index)
					goal_positions = {}

					for action_step in range(int(2 / self._grid_size) - 1):
						# print('action_step: ', action_step)
						action_success, starting_point_index = self.Move_navigation_specialized(starting_point_index=starting_point_index, direction=direction, do_move=False)
						if action_success and action_step >= 3:
							for side_move_direction_index, side_move_direction in side_move_directions.items():

								side_move_starting_pt_index = copy.deepcopy(starting_point_index)
								real_side_move_steps = min(side_move_steps, action_step + 1)

								for side_move_step in range(real_side_move_steps):

									side_move_success, side_move_starting_pt_index = self.Move_navigation_specialized(starting_point_index=side_move_starting_pt_index, direction=side_move_direction, do_move=False)
									# print('side_move_step: ', side_move_step)
									if side_move_success:

										distance = np.sqrt(((action_step + 1) * self._grid_size) ** 2 + ((side_move_step + 1) * self._grid_size) ** 2)

										if not distance in list(goal_positions.keys()):
											goal_positions[distance] = [side_move_starting_pt_index]
										else:
											goal_positions[distance].append(side_move_starting_pt_index)
										self._train_data_distance.append(distance)

									else:
										break
						elif not action_success:
							break
					# print('goal_positions: ', goal_positions)
					for goal_index, goal_position_indexes in goal_positions.items():
						for goal_position_index in goal_position_indexes:
							# print('pos_index', pos_index, goal_position_index)
							self._Agent_action.Teleport_agent(self._point_list[pos_index])
							# time.sleep(0.5)
							action_taken = self._Agent_action.Teleport_agent(self._point_list[goal_position_index], useful=True)
							# time.sleep(0.5)
							self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_name[action_index]])

		return

	def Random_SPTM_like_method(self):
		for i in range(self._SPTM_like_method_try):
			coor_move = [0, 0]
			orientation_init = [i for i in range(12)]
			orientation_init.extend([0, 3, 6, 9])
			# pre_rand_rot_num = random.randint(0, 12)
			pre_rand_rot_num = random.choice(orientation_init)
			current_orientation = self.Get_agent_rotation()['y']
			orientation_error = pre_rand_rot_num * 30 - current_orientation
			self._Agent_action.Unit_rotate(orientation_error)
			# pre_rand_rot_num = random.randint(0, 6)
			# for _ in range(pre_rand_rot_num):
			# 	self._Agent_action.Unit_rotate(random.choice([-1, 1]) * 30)
			rand_point = random.randint(0, self._point_num - 1)
			pre_point = copy.deepcopy(rand_point)
			self._Agent_action.Teleport_agent(self._point_list[rand_point])
			rand_action_step = random.randint(1, self._rand_step_num)
			# rand_action_step = 1
			first_rand_action = None
			first_traj = None
			init_orientation = self.Get_agent_rotation()['y']
			init_action_just_assigned = False
			move_action_list = [0, 3, 4, 5]
			opposite_action = {0: 3, 3: 0, 4: 5, 5: 4}
			action_name = {0: 'MOVE_FORWARD', 1: 'TURN_RIGHT', 2: 'TURN_LEFT', 3: 'MOVE_BACKWARD', 4: 'MOVE_RIGHT', 5: 'MOVE_LEFT'}
			move_direction = {0: 'forward', 3: 'backward', 4: 'right', 5: 'left'}
			move_coor = {0: [0.25, 0], 3: [-0.25, 0], 4: [0, -0.25], 5: [0, 0.25]}
			for action_num in range(rand_action_step):

				action = random.randint(0, self._total_action_num - 1)
				action_prob = [0.20, 0.10, 0.10, 0.20, 0.20, 0.20]
				action = np.random.choice([0, 1, 2, 3, 4, 5], p=action_prob)
				# action = 0
				# while action == 4:
				# 	action = np.random.choice([0, 1, 2, 3, 4, 5], p=action_prob)
				if first_rand_action in move_action_list:
					prob = [0.225 for action_index in range(len(move_action_list))]
					prob[move_action_list.index(first_rand_action)] = 0.40
					prob[move_action_list.index(opposite_action[first_rand_action])] = 0.15
					action = np.random.choice(move_action_list, p=prob)
					# action = 0
					# if action == 4 or action == 3:
					# 	continue
					action = random.choice(move_action_list)
				if self._debug:
					print('action: ', action)
					print('action_num: ', action_num)
				init_action_just_assigned = False
				if first_rand_action is None:
					first_rand_action = action
					init_action_just_assigned = True
				if self._debug:
					print('first_rand_action: ', first_rand_action)
				if action in move_action_list:

					success, moving_index = self.Move_navigation_specialized(starting_point_index=pre_point, direction=move_direction[action])
					if first_rand_action == opposite_action[action] and not self.follow_init_move(first_traj, moving_index):
						if self._debug:
							print('can not move ', action_name[action])
						continue
					if first_rand_action == action and init_action_just_assigned:
						first_traj = [rand_point, moving_index]
					if success:
						action_taken = self._Agent_action.Teleport_agent(self._point_list[moving_index], useful=True)
						coor_move = list(map(lambda x, y: x + y, coor_move, move_coor[action]))
						pre_point = copy.deepcopy(moving_index)

					elif init_action_just_assigned:
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
					# if action_taken and not self._debug:
					if action_taken:
						self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_name[first_rand_action]])
						break
				if action == 2:
					if first_rand_action == 1:
						current_orientation = self.Get_agent_rotation()['y']
						if self.degree_right_or_left(self.Wrap_to_degree(current_orientation - self._rotate_degree_for_train + 1),
							self.Wrap_to_degree(init_orientation + self._rotate_degree_for_train)) == 'Left':
							if self._debug:
								print('can not turn left')
							continue
					action_taken = self._Agent_action.Unit_rotate(-self._rotate_degree_for_train)
					# if action_taken and not self._debug:
					if action_taken:
						self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_name[first_rand_action]])
						break
				# if self._debug:
				# 	time.sleep(0.5)
				if self._debug:
					print('first_rand_action second: ', first_rand_action)
				if action_num == rand_action_step - 1 and not first_rand_action is None:
				# if action_num == rand_action_step - 1:
					start_point = self._point_list[rand_point]
					current_point = list(self.Get_agent_position().values())
					position_error = [current_point[0] - start_point[0], current_point[2] - start_point[2]]
					# goal_facing = np.arctan2(position_error[0], position_error[1]) * 180 / np.pi + 180
					# current_facing = init_orientation
					# print('goal_facing: ', goal_facing)
					# print('current_facing: ', current_facing)
					# print('position_error: ', position_error)
					# facing_error = current_facing - goal_facing
					facing_error = np.arctan2(coor_move[1], coor_move[0]) * 180 / np.pi - pre_rand_rot_num * 30
					if facing_error < 0:
						facing_error = 180 - facing_error
					# print('coor_move: ', coor_move)
					# print('facing_error: ', facing_error)
					self._train_data_distance.append(np.linalg.norm(position_error))
					self._train_data_orientation.append(facing_error)
					# print('distance: ', np.linalg.norm(position_error))
					self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_name[first_rand_action]])
			if self._debug:
				print('---------------------------------------')
				# time.sleep(0.5)
		return

	# Make a random action, all_actions false for only making forward/backward/move_right/move_left move. Weight is probability of actions
	def Random_move_w_weight(self, all_actions=True, weight=None):

		if all_actions:
			if weight is None:
				action_index = np.random.choice(list(range(0, self._total_action_num)))
			else:
				action_index = np.random.choice(list(range(0, self._total_action_num)), p=weight)
		else:
			if weight is None:
				action_index = np.random.choice(list(TRANSLATION_ACTION_TYPE.values()))
			else:
				action_index = np.random.choice(list(TRANSLATION_ACTION_TYPE.values()), p=weight)

		success, self._agent_current_pos_index = self.Move_navigation_specialized(self._agent_current_pos_index, direction=TRANSLATION_DIRECTION[action_index], do_move=True)
		return success, self._agent_current_pos_index

	def Random_action(self, starting_point_index, rotation_degree):
		current_point_index = starting_point_index
		rand_action_index = random.randint(0, len(self._Agent_action._action_type) - 1)
		rand_action_index = random.choice([0, 3, 4, 5])
		# print('rand action: ', rand_action_index)
		if rand_action_index == 0:
			success, moving_index = self.Move_navigation_specialized(starting_point_index, do_move=True)
			if success:
				current_point_index = moving_index
		elif rand_action_index == 1:
			self._Agent_action.Unit_rotate(degree=rotation_degree)
		elif rand_action_index == 2:
			self._Agent_action.Unit_rotate(degree=-rotation_degree)
		if rand_action_index == 3:
			success, moving_index = self.Move_navigation_specialized(starting_point_index, direction='backward', do_move=True)
			if success:
				current_point_index = moving_index
		if rand_action_index == 4:
			success, moving_index = self.Move_navigation_specialized(starting_point_index, direction='right', do_move=True)
			if success:
				current_point_index = moving_index
		if rand_action_index == 5:
			success, moving_index = self.Move_navigation_specialized(starting_point_index, direction='left', do_move=True)
			if success:
				current_point_index = moving_index
		return current_point_index

	def Set_nodes(self, node_list):
		self._node_list = node_list

	def Set_nodes_pair(self, nodes_pair):
		self._nodes_pair = nodes_pair

	def Set_connected_subnodes(self, connected_subnodes):
		self._connected_subnodes = connected_subnodes

		# Assume path is given as list of node index
	# def Navigate_by_path(self, path):

	# 	pah_node_index = []
	# 	max_path_value = max(path)
	# 	if max_path_value > len(self._node_list) - 1:
	# 		for node in path:
	# 			pah_node_index.append(self._node_list.index(node))
	# 	else:
	# 		pah_node_index = path

	# 	pass

	def Node_navigation_test(self, node_pair_list, subnodes=None):
		success_case = 0
		orientations = [0, 90, 180, 270]
		positive = 0
		negative = 0
		reach_node = []
		reach_node_positive = []
		reach_node_negative = []
		num_test = 0
		test_num_test = 0
		self._neighbor_nodes_dis = []
		self._neighbor_nodes_facing = [[] for i in range(len(node_pair_list))]
		self._neighbor_nodes_facing_degree = [[] for i in range(len(node_pair_list))]
		self._neighbor_nodes_coor_difference = []
		self._success_list = [[] for i in range(len(node_pair_list))]

		for i in range(len(node_pair_list)):
			# if i > 0:
			# 	break
			neighbor_node_delta_position = list(map(lambda x, y: x - y, self._point_list[node_pair_list[i][0]], self._point_list[node_pair_list[i][1]]))
			# print('self._point_list[node_pair_list[i][0]]: ', self._point_list[node_pair_list[i][0]])
			# print('self._point_list[node_pair_list[i][1]]: ', self._point_list[node_pair_list[i][1]])
			self._neighbor_nodes_dis.append(np.linalg.norm(neighbor_node_delta_position))
			self._neighbor_nodes_coor_difference.append(neighbor_node_delta_position)
			# self._neighbor_nodes_dis.append()

			num_test += 2 * len(subnodes[i])

			node_pair = node_pair_list[i]
			reverse_node_pair = copy.deepcopy(node_pair)
			reverse_node_pair.reverse()
			node_pair_bidir = [node_pair, reverse_node_pair]
			# print('node_pair_bidir: ', node_pair_bidir)
			for direction in range(len(node_pair_bidir)):
				# print('node_pair_bidir[direction]: ',  node_pair_bidir[direction])
				# if not direction == 0:
				# 	break
				goal_error = list(map(lambda x, y: x - y, self._point_list[node_pair_bidir[direction][0]], self._point_list[node_pair_bidir[direction][1]]))

				goal_facing = np.arctan2(goal_error[0], goal_error[2]) * 180 / np.pi

				# print('goal_facing: ', goal_facing)
				if goal_facing < 0:
					goal_facing = 360 + goal_facing
				# print('goal_facing: ', goal_facing)
				# self._neighbor_nodes_facing[i]

				current_orientation = self.Get_agent_rotation()['y']
				orientation_error = 0 - current_orientation
				self._Agent_action.Unit_rotate(orientation_error)
				# print('Dumb_Navigetion.Get_agent_rotation(): ', Dumb_Navigetion.Get_agent_rotation())
				# time.sleep(2)

				for orientation_index in range(len(orientations)):

					# if not orientation_index == 1:
					# 	continue
					# print('orientation_index:', orientation_index)
					# print('orientations[orientation_index]: ', orientations[orientation_index])
					if not subnodes is None:
						if not orientation_index in subnodes[i]:
							continue
					current_facing = orientations[orientation_index]

					facing_error = goal_facing - current_facing
					# print('facing_error: ', facing_error)
					self._neighbor_nodes_facing_degree[i].append(facing_error)
					# if not 50 < facing_error <= 130:
					# if not -130 <= facing_error < -50 or np.linalg.norm(neighbor_node_delta_position) > 0.5:
					# 	continue
					# print('goal_error: ', goal_error)
					# print('current_facing: ', current_facing)
					# print('facing_error: ', facing_error)
					test_num_test += 1

					while facing_error > 180:
						facing_error -= 360
					while facing_error < -180:
						facing_error += 360
					if -50 <= facing_error <= 50:
						self._neighbor_nodes_facing[i].append('front')
						# print('front')
					elif 130 < facing_error <= 180 or -180 <= facing_error < -130:
						self._neighbor_nodes_facing[i].append('back')
						# print('back')
					elif 50 < facing_error <= 130:
						self._neighbor_nodes_facing[i].append('left')
					elif -130 <= facing_error < -50:
						self._neighbor_nodes_facing[i].append('right')
						# print('side')
						# continue

					# print('goal_facing: ', goal_facing)
					# print('current_facing: ', current_facing)
					# print('goal: ', self._point_list[node_pair_bidir[direction][0]])
					# print('start: ', self._point_list[node_pair_bidir[direction][1]])
					# print('goal_error: ', goal_error)
					start = [self._point_list[node_pair_bidir[direction][1]][2], self._point_list[node_pair_bidir[direction][1]][0]]
					goal = [self._point_list[node_pair_bidir[direction][0]][2], self._point_list[node_pair_bidir[direction][0]][0]]

					# print('self._neighbor_nodes_facing[i]: ', self._neighbor_nodes_facing[i][])
					current_orientation = self.Get_agent_rotation()['y']
					# print('rotation_pre: ', self.Get_agent_rotation())
					orientation_error = orientations[orientation_index] - current_orientation
					self._Agent_action.Unit_rotate(orientation_error)

					# time.sleep(2)
					# print('Teleport_agent to goal')
					self._Agent_action.Teleport_agent(self._point_list[node_pair_bidir[direction][0]], useful=True)

					# self.Plot_map(start=start, goal=goal, orientation=orientation_index)

					# time.sleep(2)
					goal_pose = {'position': self.Get_agent_position(), 'rotation': self.Get_agent_rotation()}
					self._Agent_action.Update_event()
					goal_frame = self._Agent_action.Get_frame()
					# print('Teleport_agent to start')
					# print('--------------------------------')

					self._Agent_action.Teleport_agent(self._point_list[node_pair_bidir[direction][1]], useful=True)

					# self.Plot_map(start=start, goal=goal, orientation=orientation_index)
					# time.sleep(0.5)
					# self._goal_num += 1
					# self._current_num += 1
					if self.Navigate_by_ActionNet(image_goal=goal_frame, goal_pose=goal_pose, max_steps=self._Navigation_max_try):
						success_case += 1
						self._success_list[i].append(1)
						reach_node.append(i)
						if direction == 0:
							positive += 1
							reach_node_positive.append(i)
						elif direction == 1:
							negative += 1
							reach_node_negative.append(i)
					else:
						self._success_list[i].append(0)

		reach_node = list(set(reach_node))
		reach_node_positive = list(set(reach_node_positive))
		reach_node_negative = list(set(reach_node_negative))
		print('success_case: ', success_case)
		print('len(node_pair_list): ', len(node_pair_list) * 2 * 4)
		print('num_test: ', num_test)
		print('test_num_test: ', test_num_test)
		print('positive: ', positive)
		print('negative: ', negative)
		print('reach_node_positive: ', reach_node_positive)
		print('reach_node_negative: ', reach_node_negative)
		print('reach_node: ', len(reach_node))
		print('reach_node: ', reach_node)
		print('len(node_pair_list): ', len(node_pair_list))

		return

	def Write_result_csv(self):
		result = open('result.csv','a')
		result_writer = csv.writer(result)
		for i in range(len(self._success_list)):
			for j in range(len(self._success_list[i])):
				result_writer.writerow([str(self._neighbor_nodes_dis[i]), str(self._neighbor_nodes_facing[i][j]), str(self._success_list[i][j]),
					str(self._neighbor_nodes_facing_degree[i][j])])
		return

	def Write_dis_csv(self):
		distance = open('distance.csv','a')
		distance_writer = csv.writer(distance)
		for i in range(len(self._train_data_distance)):
				distance_writer.writerow([self._train_data_distance[i], self._train_data_orientation[i]])
		return

	def Navigation_test(self):
		move_direction = {0: 'forward', 3: 'backward', 4: 'right', 5: 'left'}
		move_action = [0, 3, 4, 5]
		success_case = 0
		for i in range(self._Navigation_test_num):

			pre_rand_rot_num = random.randint(0, 6)
			for _ in range(pre_rand_rot_num):
				self._Agent_action.Unit_rotate(random.choice([-1, 1]) * self._rotate_degree_for_train)
			rand_point = random.randint(0, self._point_num - 1)
			current_orientation = self.Get_agent_rotation()['y']
			self._Agent_action.Teleport_agent(self._point_list[rand_point], useful=True)
			current_point_index = rand_point
			goal_pose = {}
			rotation_change = 0
			# time.sleep(1)
			for _ in range(self._rand_step_num - 2):
			# for _ in range(1):
			 	action = random.randint(0, len(self._Agent_action._action_type) - 2)
			 	action = random.choice([0, 0, 0, 3, 4, 5])
			 	action = 5
			 	action = random.choice([0, 3, 5, 5])

			 	# action = 0
			 	# print('action: ', action)
			 	# time.sleep(0.5)
			 	if action in [1, 2]:
			 		continue
			 	# action = random.choice([1, 2])
			 	if action in move_action:
			 		success, moving_index = self.Move_navigation_specialized(current_point_index, direction=move_direction[action], do_move=True)
			 		if success:
			 			current_point_index = moving_index
			 			print('action: ', action)
			 	elif action == 1:
			 		self._Agent_action.Unit_rotate(degree=90)
			 		rotation_change += 90
			 	elif action == 2:
			 		self._Agent_action.Unit_rotate(degree=-90)
			 		rotation_change -= 90
			 	goal_pose = {'position': Dumb_Navigetion.Get_agent_position(), 'rotation': Dumb_Navigetion.Get_agent_rotation()}
			 	goal_frame = Dumb_Navigetion._Agent_action.Get_frame()
			# time.sleep(1)

			self._Agent_action.Teleport_agent(self._point_list[rand_point], useful=True)

			# time.sleep(0.5)
			# print('rotation_change: ', rotation_change)
			# print('position change: ', list(map(lambda x, y: x - y, list(Dumb_Navigetion.Get_agent_position().values()), list(goal_pose['position'].values()))))
			position_change = list(map(lambda x, y: y - x, list(Dumb_Navigetion.Get_agent_position().values()), list(goal_pose['position'].values())))
			goal_orientation = np.arctan2(position_change[0], position_change[2]) / np.pi * 180
			if goal_orientation < 0:
				goal_orientation = 360 + goal_orientation
			print('position_change: ', position_change)
			print('goal_orientation: ', goal_orientation)
			print('current_orientation: ', current_orientation)
			print('-------------------------------')
			self._train_data_distance.append(np.linalg.norm(position_change))
			facing_error = goal_orientation - current_orientation
			# if facing_error < 0:
			# 			facing_error = 180 - facing_error
			self._train_data_orientation.append(facing_error)
			self._Agent_action.Unit_rotate(degree=-rotation_change)
			if self.Navigate_by_ActionNet(image_goal=goal_frame, goal_pose=goal_pose, max_steps=self._Navigation_max_try):
				success_case += 1
		print('success_case: ', success_case)
		print('self._Navigation_test_num: ', self._Navigation_test_num)

		return

	def Set_localization_network(self):
		checkpoint = './Network/retrieval_network/checkpoints/image_siamese_nondynamics_best_fit.pkl'
		# checkpoint = CHECKPOINTS_PREFIX + 'image_siamese_nondynamics_best_fit.pkl'
		self._model = Retrieval_network()


	def Navigation_stop(self, image_goal, image_current, goal_pose=None, hardcode=False):

		position_current = list(self.Get_agent_position().values())
		rotation_current = list(self.Get_agent_rotation().values())

		if hardcode is True:
			if goal_pose is None:
				return False
			goal_position = goal_pose['position']
			goal_rotation = goal_pose['rotation']

			if isinstance(goal_position, dict):
				goal_position = list(goal_position.values())
			if isinstance(goal_rotation, dict):
				goal_rotation = list(goal_rotation.values())

			distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, goal_position, position_current))))
			rotation_difference = np.abs(goal_rotation[1] - rotation_current[1])

			if distance > 0.5 * self._grid_size or rotation_difference > 10:
				return False
			else:
				return True

		Image_goal = Image.fromarray(image_goal)
		Image_current = Image.fromarray(image_current)

		self._current_num += 1
		self._step_poses.append({'position': position_current, 'rotation': rotation_current})
		# Image_goal.save('./images/' + str(self._goal_num) + '_0.jpg')


		localized = self._model.is_localized_static(Image_goal, Image_current)

		Image_current.save('./images/' + str(self._current_num) + '_' + str(position_current[0])+ '_' + str(position_current[2]) +
			 '_' +str(int(rotation_current[1])) + '_' + str(localized) + '.jpg')

		return localized

	def Self_localize(self):

		current_position = list(self.Get_agent_position().values())
		distance_search_min = 1000
		nearest_index = -1

		for i, point in enumerate(self._point_list):

			distance_search = np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, current_position))))
			if distance_search < distance_search_min:
				distance_search_min = distance_search
				nearest_index = i

		if distance_search_min > 0.5 * self._grid_size:
			logging.error('Can not find starting point in point list')
			return False
		return nearest_index

	def Node_localize(self, node_position):

		if isinstance(node_position, dict):
			node_position_list = list(node_position.values())
		else:
			node_position_list = copy.deepcopy(node_position)
		distance_search_min = 1000
		nearest_index = -1

		for i, point in enumerate(self._point_list):

			distance_search = np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, node_position_list))))
			if distance_search < distance_search_min:
				distance_search_min = distance_search
				nearest_index = i

		if distance_search_min > 0.5 * self._grid_size:
			logging.error('Can not find starting point in point list')
			return False
		return nearest_index

	def Navigate_by_ActionNet(self, image_goal, goal_pose, max_steps):
		goal_position = goal_pose['position']
		goal_rotation = goal_pose['rotation']
		if isinstance(goal_position, dict):
			goal_position = list(goal_position.values())
		if isinstance(goal_rotation, dict):
			goal_rotation = list(goal_rotation.values())
		pre_action = None
		loop_action = [3, 2, 1, 0, 5, 4]
		move_direction = {0: 'forward', 3: 'backward', 4: 'right', 5: 'left'}
		rot_direction = {1: 90, 2: -90}
		move_action = [0, 3, 4, 5]
		rotation_action = [1, 2]

		nearest_index = self.Self_localize()
		if nearest_index is False:
			return False
		else:
			self._agent_current_pos_index = nearest_index

		# print('goal_position: ', goal_position)
		# print('current_position: ', current_position)
		# print('goal_rotation: ', goal_rotation)
		# print('current_rotation: ', current_rotation)
		step = 0
		self._Agent_action.Update_event()
		current_frame = self._Agent_action.Get_frame()

		while not self.Navigation_stop(image_goal=image_goal, image_current=current_frame, goal_pose=goal_pose, hardcode=True):

			self._current_num += 1
			position_current = list(self.Get_agent_position().values())
			rotation_current = list(self.Get_agent_rotation().values())
			Image_current = Image.fromarray(current_frame)

			Image_current.save('./images/' + str(self._current_num) + '_' + str(position_current[0])+ '_' + str(position_current[2]) +
			 	'_' +str(int(rotation_current[1])) + '_' + str(False) + '.jpg')
			step += 1
			if step >= max_steps:
				return False
			image_current = self._Agent_action.Get_frame()
			action_predict = self._action_network.predict(image_current=image_current, image_goal=image_goal)

			if loop_action[action_predict] == pre_action:

				self._agent_current_pos_index = self.Random_action(starting_point_index=self._agent_current_pos_index, rotation_degree=30)
				pre_action = None
				continue

			if self._debug:
				print('action_predict: ', action_predict)

			if action_predict in move_action:
				success, moving_index = self.Move_navigation_specialized(self._agent_current_pos_index, direction=move_direction[action_predict.item()], do_move=True)
				if success:
					self._agent_current_pos_index = moving_index
					pre_action = action_predict.item()
				else:
					self._agent_current_pos_index = self.Random_action(starting_point_index=self._agent_current_pos_index, rotation_degree=30)
					# print('can not move forward')
					pre_action = None

			elif action_predict in rotation_action:
				self._Agent_action.Unit_rotate(degree=rot_direction[action_predict.item()])
				pre_action = action_predict.item()

			current_frame = self._Agent_action.Get_frame()
			# print('goal_position: ', goal_position)
			# print('current_position: ', current_position)
			# print('goal_rotation: ', goal_rotation)
			# print('current_rotation: ', current_rotation)
			# print('-----------------------------------------------')
			# time.sleep(0.5)
		return True

	def Get_orientation_two_points(self, starting_point_index, goal_point_index):
		starting_point_position = self._point_list[starting_point_index]
		goal_point_position = self._point_list[goal_point_index]
		error_vector = list(map(lambda x, y: x - y, goal_point_position, starting_point_position))
		error_orientation = np.arctan2(error_vector[0], error_vector[2]) * 180 / np.pi
		if error_orientation < 0:
			error_orientation += 360
		return error_orientation

	def Move_navigation_specialized(self, starting_point_index, direction='forward', do_move=False):
		current_orientation = self.Get_agent_rotation()['y']
		current_position = self._point_list[starting_point_index]
		if direction.lower() == 'forward':
			moving_direction = {0: [0, 0, 1], 1: [1, 0, 0], 2: [0, 0, -1], 3: [-1, 0, 0]}
		if direction.lower() == 'backward':
			moving_direction = {0: [0, 0, -1], 1: [-1, 0, 0], 2: [0, 0, 1], 3: [1, 0, 0]}
		# if direction.lower() == 'left':
		if direction.lower() == 'right':
			moving_direction = {0: [1, 0, 0], 1: [0, 0, -1], 2: [-1, 0, 0], 3: [0, 0, 1]}
		# if direction.lower() == 'right':
		if direction.lower() == 'left':
			moving_direction = {0: [-1, 0, 0], 1: [0, 0, 1], 2: [1, 0, 0], 3: [0, 0, -1]}
		moving_step = {}
		for key in list(moving_direction.keys()):
			moving_step[key] = map(lambda x: x * self._grid_size, moving_direction[key])

		heading_direction_index = int(np.floor((current_orientation - 45) / 90) + 1)
		if current_orientation < 45 or current_orientation > 315:
			heading_direction_index = 0
		moving_point = list(map(lambda x, y: x + y, current_position, moving_step[heading_direction_index]))
		connected_point_index = self._map[starting_point_index]
		# if direction.lower() == 'left':
		# 	print('connected_point_index: ', connected_point_index)
		# 	print('moving_point: ', moving_point)
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

		# if direction.lower() == 'left':
		# 	print('distance_min: ', distance_min)
		# 	print('self._point_list[nearest_point_index]: ', self._point_list[nearest_point_index])

		if distance_min > 0.5 * self._grid_size:
			logging.error('Nowhere to go forward')
			return (False, -1)
		if do_move:
			self._Agent_action.Teleport_agent(position=self._point_list[nearest_point_index], useful=True)
		return (True, nearest_point_index)

	def Open_close_label_text(self):
		return self._Agent_action.Open_close_label_text()

	def Get_agent_position(self):
		return self._Agent_action.Get_agent_position()

	def Get_agent_rotation(self):
		return self._Agent_action.Get_agent_rotation()

	def Plot_map(self, start, goal, orientation):
		fig, ax = plt.subplots()

		sight_line_point = [[[0.25, 0.25], [0.25, -0.25]], [[0.25, 0.25], [-0.25, 0.25]],
		[[-0.25, 0.25], [-0.25, -0.25]], [[-0.25, -0.25], [0.25, -0.25]]]

		plt.scatter(self._point_x, self._point_y, color='#1f77b4')
		plt.scatter([start[0]], [start[1]], color='#DEB887')
		# plt.scatter([goal[0]], [goal[1]], color='#00FFFF')

		for point in sight_line_point[orientation]:
			x = [start[0], start[0] + point[0]]
			y = [start[1], start[1] + point[1]]
			plt.plot(x, y, color='#A52A2A')
		for point in sight_line_point[orientation]:
			x = [goal[0], goal[0] + point[0]]
			y = [goal[1], goal[1] + point[1]]
			plt.plot(x, y, color='#A52A2A')
		plt.axis('equal')
		plt.show()


	def _build_map(self):
		self._point_list.append(list(self._starting_point.values()))
		self._map[self._point_num] = []
		self._map_searched[self._point_num] = True
		self._point_num += 1
		for point_adding in self._coordinate_dict:
			if self._starting_point == point_adding:
				continue
			self._point_list.append(list(point_adding.values()))
			self._point_x.append(point_adding['z'])
			self._point_y.append(point_adding['x'])
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
		went_points = []
		for _ in range(int(self._point_num / 20)):
			# print(went_points)
			point_index = random.randint(0, self._point_num - 1)
			while point_index in went_points:
				point_index = random.randint(0, self._point_num - 1)
			went_points.append(point_index)
			pre_rand_rot_num = random.randint(0, 6)
			for _ in range(pre_rand_rot_num):
				self._Agent_action.Unit_rotate(random.choice([-1, 1]) * self._rotate_degree_for_train)
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
					# print('sleep')
				action = self._Agent_action.Unit_rotate(-self._rotate_degree_for_train)
				if action and not self._debug:
					self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action])
		# for point_index in list(self._map.keys()):
		for point_index in went_points:
			# print(point_index)
			# print('self._point_list[point_index]: ', self._point_list[point_index])
			goal_orientation = [0, 90, 180, 270]

			actions = {'forward': 'MOVE_FORWARD', 'backward': 'MOVE_BACKWARD', 'right': 'MOVE_RIGHT', 'left': 'MOVE_LEFT'}
			for facing in goal_orientation:
				current_orientation = self.Get_agent_rotation()['y']
				orientation_error = facing - current_orientation
				self._Agent_action.Unit_rotate(orientation_error)

				for action_type, action_name in actions.items():
					# time.sleep(0.5)
					self._Agent_action.Teleport_agent(position=self._point_list[point_index], useful=False)
					# time.sleep(0.5)
					success, moving_index = self.Move_navigation_specialized(point_index, direction=action_type, do_move=True)
					if success:
						self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action_name])

			# for connected_point_index in self._map[point_index]:
			# 	# time.sleep(1)
			# 	self._Agent_action.Teleport_agent(position=self._point_list[point_index], useful=False)
			# 	goal_orientation = self.Get_orientation_two_points(point_index, connected_point_index)
			# 	current_orientation = self.Get_agent_rotation()['y']
			# 	orientation_error = goal_orientation - current_orientation

			# 	# while orientation_error < 0:
			# 	# 	orientation_error += 360
			# 	# print('orientation_error: ', orientation_error)
			# 	self._Agent_action.Unit_rotate(orientation_error)
			# 	# print('self.Get_agent_rotation()[]: ', self.Get_agent_rotation()['y'])
			# 	# time.sleep(1)

			# 	action = self._Agent_action.Teleport_agent(position=self._point_list[connected_point_index], useful=True)
			# 	# time.sleep(1)
			# 	if action and not self._debug:
			# 		self._Agent_action._Save_RGB_label(self._Agent_action._action_type[action])
			# 	# print('----------------------------')
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
	def __init__(self, AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory, overwrite_data=False, for_test_data=False, debug=False, special=False, more_special=False):
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._grid_size = grid_size
		self._rotation_step = rotation_step
		self._sleep_time = sleep_time
		self._AI2THOR = AI2THOR
		self._for_test_data = for_test_data
		self._special = special
		self._more_special = more_special
		self._scene_name = self.Get_scene_name()
		if self._AI2THOR:
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, fieldOfView=120, agentMode='bot')
		else:
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, fieldOfView=120)

		self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)
		if self._special:
			self._scene_name += '_special'
		elif self._more_special:
			self._scene_name += '_right'
		# self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, fieldOfView=120)
		self._save_directory = save_directory

		self._overwrite_data = overwrite_data
		self._event = self._controller.step('Pass')
		# self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)
		self._start_time = time.time()

		self._debug = debug
		self._action_label_text_file = None
		self._action_type = {'INVALID_ACTION': -1, 'MOVE_FORWARD': 0, 'TURN_RIGHT': 1, 'TURN_LEFT': 2, 'MOVE_BACKWARD': 3, 'MOVE_RIGHT': 4, 'MOVE_LEFT': 5}
		self._pre_image_name = None

	def Reset_scene(self, scene_type, scene_num):
		self._scene_name = self.Get_scene_name(scene_type=scene_type, scene_num=scene_num)
		self._controller.reset(scene=self._scene_name)

	def Get_scene_name(self, scene_type=None, scene_num=None):

		if not scene_type is None and not scene_num is None:
			self._scene_type = scene_type
			self._scene_num = scene_num

		if self._AI2THOR:
			if not self._for_test_data:
				scene_name = 'FloorPlan_Train' + str(self._scene_type) + '_' + str(self._scene_num)
			else:
				scene_name = 'FloorPlan_Val' + str(self._scene_type) + '_' + str(self._scene_num)
			print('scene_name: ', self._scene_name)
		else:
			scene_name = 'FloorPlan' + str(scene_setting[self._scene_type] + self._scene_num)
		return scene_name

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
		frame = self.Get_frame()
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
		# if not self._debug and not useful:
		if not useful:
			self._Save_RGB_label(self._action_type['INVALID_ACTION'])
		return 'MOVE_FORWARD'

	def Unit_move(self):
		self._event = self._controller.step(action='MoveAhead')
		return 'MOVE_FORWARD'

	def Unit_move_back(self):
		self._event = self._controller.step(action='MoveBack')
		return 'MOVE_FORWARD'

	def Unit_move_left(self):
		self._event = self._controller.step(action='MoveLeft')
		return 'MOVE_Left'

	def Unit_move_right(self):
		self._event = self._controller.step(action='MoveRight')
		return 'MOVE_Right'

	def Rotate_to_degree(self, goal_degree):
		current_orientation = self.Get_agent_rotation()['y']
		orientation_error = goal_degree - current_orientation
		self.Unit_rotate(orientation_error)

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

	# checkpoint = '../Network/retrieval_network/checkpoints/image_siamese_nondynamics_best_fit.pkl'
	# print('checkpoint: ', checkpoint)
	# model = SiameseNetImage()
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# print("Model testing on: ", device)
	# model.to(device)
	# model.load_state_dict(torch.load(checkpoint))
	# model.eval()
 #    # Generate images for
	# image1 = Image.open('./images/0_0.jpg')
	# image2 = Image.open('./images/5_1.jpg')

	# # plt.imshow(array)
	# print(is_localized_static(model, device, image1, image2))


	# # # controller = Controller(scene='FloorPlan_Train1_1', agentMode='bot')
	# # # time.sleep(10)
	# exit()

	Dumb_Navigetion = Dumb_Navigetion(AI2THOR=args.AI2THOR, scene_type=args.scene_type, scene_num=args.scene_num, grid_size=args.grid_size,
		rotation_step=args.rotation_step, sleep_time=args.sleep_time, save_directory=args.save_directory, overwrite_data=args.overwrite_data,
		for_test_data=args.test_data, debug=args.debug, special=args.special, more_special=False)

	# Dumb_Navigetion.Set_localization_network()


	# Dumb_Navigetion.Open_close_label_text()
	# # Dumb_Navigetion.Traverse_neighbor_map()
	# # Dumb_Navigetion.Dumb_traverse_map()
	# # Dumb_Navigetion.Long_range_data_specified()
	# # Dumb_Navigetion.Long_range_data_more_specified()
	# Dumb_Navigetion.Random_SPTM_like_method()
	# # Dumb_Navigetion.Random_traverse_map(pair_num=7)
	# # Dumb_Navigetion.Fast_traverse_map(goal_points_num=24)
	# # position = Dumb_Navigetion.Get_agent_position()
	# # ori_position = copy.deepcopy(position)
	# # reach = Dumb_Navigetion._Agent_action.Get_reachable_coordinate()
	# Dumb_Navigetion.Open_close_label_text()
	# Dumb_Navigetion.Write_dis_csv()
	# exit()
	# for i in range(10):
	# 	rand_point = random.randint(0, Dumb_Navigetion._point_num - 1)
	# 	rand_ori = random.randint(0, 12)
	# 	Dumb_Navigetion._Agent_action.Unit_rotate(random.choice([-15, 15]) * rand_ori)
	# 	current_orientation = Dumb_Navigetion.Get_agent_rotation()['y']
	# 	print('current_orientation: ', current_orientation)
	# 	print('position: ', Dumb_Navigetion._point_list[rand_point])
	# 	Dumb_Navigetion._Agent_action.Teleport_agent(Dumb_Navigetion._point_list[rand_point], useful=True)
	# 	success, moving_index = Dumb_Navigetion.Move_navigation_specialized(rand_point, direction='left', do_move=True)
	# 	# time.sleep(1)
	# 	Dumb_Navigetion._Agent_action.Teleport_agent(Dumb_Navigetion._point_list[moving_index], useful=True)
	# 	print('position move: ', Dumb_Navigetion._point_list[moving_index])
	# 	print('---------------------------------')
	# exit()

	# Dumb_Navigetion.Navigation_test()
	# Dumb_Navigetion.Write_dis_csv()
	# exit()


	node_generator = Node_generator(controller=Dumb_Navigetion._Agent_action._controller, node_radius=1)
	# node_list = NODES[Dumb_Navigetion._Agent_action.Get_scene_name()]

	# node_generator.Get_node_from_position(node_list)
	# node_generator.Get_connected_orientaton_by_overlap_scene()
	# node_pair_list = node_generator.Get_neighbor_nodes()
	# subnodes = node_generator.Get_connected_subnodes()
	# node_pair = [[1, 8], [1, 16], [1, 28], [102, 107], [8, 16], [8, 28], [139, 140], [206, 242], [206, 180], [206, 221], [16, 28], [241, 221], [242, 180], [180, 221], [139, 206], [140, 206],
	# [241, 242], [242, 221], [64, 28], [104, 16], [104, 28], [64, 1], [64, 8], [64, 16], [1, 104], [8, 104]]
	# subnodes = [[0, 1, 2, 3], [0, 2, 3], [0, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 2], [0, 1, 2, 3], [0, 1], [0, 1, 2, 3],
	# [0, 3], [0, 2], [0, 1], [1], [1], [1], [1], [2, 3], [2, 3], [2, 3], [3], [3], [3], [3], [3]]
	Dumb_Navigetion.Node_navigation_test(node_pair_list=node_generator.Get_neighbor_nodes(), subnodes=node_generator.Get_connected_subnodes())
	# Dumb_Navigetion.Node_navigation_test(node_pair_list=node_pair, subnodes=subnodes)
	Dumb_Navigetion.Write_result_csv()


	# # Dumb_Navigetion.Node_navigation_test(node_pair_list=test_pairs)
	# # test_controller.step(action='RotateLeft', degrees=np.abs(30))
	# action_network = Action_network()

	# time.sleep(10)


	# print(Dumb_Navigetion.degree_right_or_left(Dumb_Navigetion.Wrap_to_degree(350 - 30), 20))
	# init = time.time()
	# for _ in range(1000000):
	# 	for _ in range(100):
	# 		pass
	# print(time.time() - init)
	exit()
	reach = Dumb_Navigetion._Agent_action.Get_reachable_coordinate()
	# print('reach: ', len(reach))

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



	# # Dumb_Navigetion._Agent_action.Unit_rotate(-30)
	# # time.sleep(2)
	# # Dumb_Navigetion._Agent_action.Unit_rotate(-30)
	# # Dumb_Navigetion._Agent_action.Unit_rotate(-30)
	# print('rotation: ', Dumb_Navigetion.Get_agent_rotation())
	# print('position: ', Dumb_Navigetion.Get_agent_position())
	# Dumb_Navigetion._Agent_action.Unit_rotate(-90)
	# Dumb_Navigetion._Agent_action.Unit_move()
	# print('rotation: ', Dumb_Navigetion.Get_agent_rotation())
	# print('position: ', Dumb_Navigetion.Get_agent_position())
	# time.sleep(2)
	# # Dumb_Navigetion._Agent_action.Unit_move_left()
	# # time.sleep(2)
	# # Dumb_Navigetion._Agent_action.Unit_move_right()
	# # time.sleep(2)
	# # print('position: ', Dumb_Navigetion.Get_agent_position())
	# exit()
	# time.sleep(1)
	rand_point = random.randint(1, len(reach) - 1)
	rand_point = 1
	print('rand_point: ', rand_point)
	Dumb_Navigetion._Agent_action.Teleport_agent(position=reach[rand_point], useful=True)
	position_temp = reach[132]
	# position_temp = copy.deepcopy(reach[20])
	position_temp_0 = copy.deepcopy(reach[20])
	# position_temp['x'] = 5.5
	# position_temp_0['x'] = 5.5
	# position_temp_0['z'] = -3.75
	print(position_temp)
	print(position_temp_0)

	# time.sleep(100)

	# Dumb_Navigetion._Agent_action.Unit_rotate(180)
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_move()
	# # Dumb_Navigetion._Agent_action.Unit_rotate(30)

	position = list(Dumb_Navigetion.Get_agent_position().values())

	# time.sleep(1)
	# Dumb_Navigetion._Agent_action.Unit_move_back()
	# time.sleep(1)
	# Dumb_Navigetion._Agent_action.Unit_move_back()
	# time.sleep(1)

	Dumb_Navigetion._Agent_action.Unit_rotate(90)
	# Dumb_Navigetion._Agent_action.Unit_move()
	# Dumb_Navigetion._Agent_action.Unit_rotate(-90)
	# time.sleep(1)
	# Dumb_Navigetion._Agent_action.Unit_rotate(30)
	# time.sleep(1)
	# Dumb_Navigetion._Agent_action.Unit_rotate(-30)
	# time.sleep(1)
	# Dumb_Navigetion._Agent_action.Unit_rotate(-30)
	# Dumb_Navigetion._Agent_action.Unit_move()


	# time.sleep(5)
	Dumb_Navigetion._Agent_action.Unit_move()
	time.sleep(1)
	Dumb_Navigetion._Agent_action.Unit_move()
	time.sleep(1)
	Dumb_Navigetion._Agent_action.Unit_move()
	time.sleep(1)
	Dumb_Navigetion._Agent_action.Unit_rotate(90)
	time.sleep(1)
	Dumb_Navigetion._Agent_action.Unit_move()
	# Dumb_Navigetion._Agent_action.Unit_move()
	time.sleep(1)
	Dumb_Navigetion._Agent_action.Unit_rotate(-90)
	time.sleep(1)
	Dumb_Navigetion._Agent_action.Unit_move()

	# Dumb_Navigetion._Agent_action.Unit_rotate(90)
	# Dumb_Navigetion._Agent_action.Unit_move()
	# Dumb_Navigetion._Agent_action.Unit_rotate(60)
	# Dumb_Navigetion._Agent_action.Unit_rotate(180)
	# Dumb_Navigetion._Agent_action.Teleport_agent(position=position_temp, useful=True)
	# time.sleep(3)


	goal_pose = {'position': Dumb_Navigetion.Get_agent_position(), 'rotation': Dumb_Navigetion.Get_agent_rotation()}
	# Dumb_Navigetion._Agent_action.Update_event()
	frame = Dumb_Navigetion._Agent_action.Get_frame()
	time.sleep(3)
	Dumb_Navigetion._Agent_action.Teleport_agent(position=position, useful=True)
	time.sleep(5)
	# Dumb_Navigetion._Agent_action.Teleport_agent(position=position_temp_0, useful=True)
	# # Dumb_Navigetion._Agent_action.Unit_rotate(30)


	Dumb_Navigetion.Navigate_by_ActionNet(frame, goal_pose, max_steps=100)
	time.sleep(100)
	exit()



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


	# position = Dumb_Navigetion.Get_agent_position()

	# time.sleep(2)
