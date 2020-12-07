from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib import cm
from dijkstar import Graph, find_path
from distutils.util import strtobool
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import time, copy, random
import logging, argparse, os, sys, csv

from experiment.node_generation import *
from experiment.experiment_config import *
from Network.action_network.action_network import Action_network
from Network.retrieval_network.retrieval_network import Retrieval_network
from lib.params import *

# SIM_WINDOW_HEIGHT = 700
# SIM_WINDOW_WIDTH = 900

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


class Robot():
	def __init__(self, scene_type, scene_num, save_directory, overwrite_data=False, AI2THOR=False, grid_size=0.25, rotation_step=90, sleep_time=0.005, use_test_scene=False, debug=False, server=None, comfirmed=None):

		self._grid_size = grid_size
		self._sleep_time = sleep_time
		self._AI2THOR_controller = AI2THOR_controller(AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time,
													  save_directory, overwrite_data, use_test_scene, debug=debug)
		self._use_test_scene = use_test_scene
		self._debug = debug

		self._total_action_num = 6
		self._Navigation_max_try = 18

		self.multithread_node = dict(server=server, comfirmed=comfirmed)

		self.Set_navigation_network()
		self.Set_localization_network()

	def Reset_scene(self, scene_type, scene_num):
		self._AI2THOR_controller.Reset_scene(scene_type=scene_type, scene_num=scene_num)

	def Set_navigation_network(self, network=None):
		if not network is None:
			self._action_network = network
			return
		else:
			self._action_network = Action_network()

	def Set_localization_network(self, network=None):
		if not network is None:
			self._localization_network = network
			return
		else:
			self._localization_network = Retrieval_network()

	def Navigation_stop(self, image_goal, image_current, goal_pose=None, hardcode=False):

		position_current = list(self.Get_robot_position().values())
		rotation_current = list(self.Get_robot_rotation().values())

		if hardcode is True:
			if goal_pose is None:
				return False
			goal_position = self._AI2THOR_controller.Get_list_form(pos_or_rot=goal_pose['position'])
			goal_rotation = self._AI2THOR_controller.Get_list_form(pos_or_rot=goal_pose['rotation'])

			distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, goal_position, position_current))))
			rotation_difference = np.abs(goal_rotation[1] - rotation_current[1])

			if distance > 0.5 * self._grid_size or rotation_difference > 10:
				return False
			else:
				return True

		Image_goal = Image.fromarray(image_goal)
		Image_current = Image.fromarray(image_current)
		localized = self._localization_network.is_localized_static(Image_goal, Image_current)

		return localized

	def Navigate_by_ActionNet(self, image_goal, goal_pose, max_steps):

		goal_position = self._AI2THOR_controller.Get_list_form(pos_or_rot=goal_pose['position'])
		goal_rotation = self._AI2THOR_controller.Get_list_form(pos_or_rot=goal_pose['rotation'])

		pre_action = None
		loop_action = [3, 2, 1, 0, 5, 4]
		move_direction = {0: 'forward', 3: 'backward', 4: 'right', 5: 'left'}
		rot_direction = {1: 90, 2: -90}
		move_action = [0, 3, 4, 5]
		rotation_action = [1, 2]

		nearest_index = self._AI2THOR_controller.Self_localize()
		if nearest_index is False:
			return False
		else:
			self._AI2THOR_controller._agent_current_pos_index = nearest_index

		step = 0
		self._AI2THOR_controller.Update_event()
		current_frame = self._AI2THOR_controller.Get_frame()

		while not self.Navigation_stop(image_goal=image_goal, image_current=current_frame, goal_pose=goal_pose, hardcode=True):
			# ------------------------------------------------------------------
			# Send information to plotter
			# ------------------------------------------------------------------
			if self.multithread_node['server'] != None:
				cur_pos = self.Get_robot_position()
				cur_rot = self.Get_robot_rotation()
				info = dict(goal_pose=[goal_position[0], goal_position[2], goal_rotation[1]], goal_img=image_goal,
							cur_pose=[cur_pos['x'], cur_pos['z'], cur_rot['y']], cur_img=current_frame, is_reached=False)
				self.multithread_node['server'].send(info)
				while True:
					if self.multithread_node['comfirmed'].value:
						self.multithread_node['comfirmed'].value = 0
						break
			# ------------------------------------------------------------------
			# ------------------------------------------------------------------

			image_current = self._AI2THOR_controller.Get_frame()
			action_predict = self._action_network.predict(image_current=image_current, image_goal=image_goal)

			if loop_action[action_predict] == pre_action:
				_, self._AI2THOR_controller._agent_current_pos_index = self._AI2THOR_controller.Random_move_w_weight()
				pre_action = None
				continue

			if self._debug:
				print('action_predict: ', action_predict)

			if action_predict in move_action:
				success, moving_index = self._AI2THOR_controller.Move_navigation_specialized(self._AI2THOR_controller._agent_current_pos_index, direction=move_direction[action_predict.item()], move=True)
				if success:
					self._AI2THOR_controller._agent_current_pos_index = moving_index
					pre_action = action_predict.item()
				else:
					_, self._AI2THOR_controller._agent_current_pos_index = self._AI2THOR_controller.Random_move_w_weight()
					pre_action = None

			elif action_predict in rotation_action:
				self._AI2THOR_controller.Unit_rotate(degree=rot_direction[action_predict.item()])
				pre_action = action_predict.item()

			current_frame = self._AI2THOR_controller.Get_frame()

			step += 1
			if step >= max_steps:
				return False

		# ----------------------------------------------------------------------
		# Send information to plotter
		# ----------------------------------------------------------------------
		if self.multithread_node['server'] != None:
			cur_pos = self.Get_robot_position()
			cur_rot = self.Get_robot_rotation()
			info = dict(goal_pose=[goal_position[0], goal_position[2], goal_rotation[1]], goal_img=image_goal,
						cur_pose=[cur_pos['x'], cur_pos['z'], cur_rot['y']], cur_img=current_frame, is_reached=True)
			self.multithread_node['server'].send(info)
			while True:
				if self.multithread_node['comfirmed'].value:
					self.multithread_node['comfirmed'].value = 0
					break
		# ----------------------------------------------------------------------
		# ----------------------------------------------------------------------
		return True

	def Open_close_label_text(self):
		return self._AI2THOR_controller.Open_close_label_text()

	def Get_robot_position(self):
		return self._AI2THOR_controller.Get_agent_position()

	def Get_robot_rotation(self):
		return self._AI2THOR_controller.Get_agent_rotation()


class AI2THOR_controller():
	def __init__(self, AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time, save_directory, overwrite_data=False, use_test_scene=False, debug=False):
		self._scene_type = scene_type
		self._scene_num = scene_num
		self._grid_size = grid_size
		self._rotation_step = rotation_step
		self._sleep_time = sleep_time
		self._AI2THOR = AI2THOR
		self._use_test_scene = use_test_scene
		self._scene_name = self.Get_scene_name()

		if self._AI2THOR:
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE, fieldOfView=FIELD_OF_VIEW, agentMode='bot')
		else:
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE, fieldOfView=FIELD_OF_VIEW)

		self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)
		self._save_directory = save_directory

		self._overwrite_data = overwrite_data
		self._event = self._controller.step('Pass')
		self._start_time = time.time()

		self._debug = debug
		self._action_label_text_file = None
		self._action_type = {'INVALID_ACTION': -1, 'MOVE_FORWARD': 0, 'TURN_RIGHT': 1, 'TURN_LEFT': 2, 'MOVE_BACKWARD': 3, 'MOVE_RIGHT': 4, 'MOVE_LEFT': 5}
		self._point_list = []
		self._agent_current_pos_index = None
		self._get_reachable_list()

	def store_toggled_map(self):
		self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size)
		self._event = self._controller.step('Pass')
		self._controller.step({"action": "ToggleMapView"})
		self._controller.step({"action": "Initialize", "makeAgentsVisible": False,})
		map = Image.fromarray(self.Get_frame(), 'RGB')
		map.save('./icon/' + self._scene_name + '.png')
		self._controller.stop()

	def Wrap_to_degree(self, degree):
		degree_wrap = copy.deepcopy(degree)
		while degree_wrap > 360:
			degree_wrap -= 360
		while degree_wrap < 0:
			degree_wrap += 360
		return degree_wrap

	def _get_reachable_list(self):
		reachable_dict = self.Get_reachable_coordinate()
		for reachable_dict_one in reachable_dict:
			self._point_list.append(list(reachable_dict_one.values()))

	def Reset_scene(self, scene_type, scene_num):
		self._scene_name = self.Get_scene_name(scene_type=scene_type, scene_num=scene_num)
		self._controller.reset(scene=self._scene_name)

	def get_scene_info(self):
		scene_name = self.Get_scene_name()
		scene_bbox = self.get_floor_bbox()
		grid_size = self._grid_size
		reachable_points = self.Get_reachable_coordinate()
		objs = self._event.metadata['objects']
		return (scene_name, scene_bbox, grid_size, reachable_points, objs)

	def get_floor_bbox(self):
		self.Update_event()
		floor = [obj for obj in self._event.metadata['objects'] if obj['objectType'] == 'Floor'][0]
		data = floor['axisAlignedBoundingBox']
		center_x = data['center']['x']
		center_z = data['center']['z']
		size_x = data['size']['x']
		size_z = data['size']['z']

		bbox_x = [center_x-size_x*0.5, center_x+size_x*0.5, center_x+size_x*0.5, center_x-size_x*0.5, center_x-size_x*0.5]
		bbox_z = [center_z+size_z*0.5, center_z+size_z*0.5, center_z-size_z*0.5, center_z-size_z*0.5, center_z+size_z*0.5]

		return (bbox_x, bbox_z)

	def Get_scene_name(self, scene_type=None, scene_num=None):

		if not scene_type is None and not scene_num is None:
			self._scene_type = scene_type
			self._scene_num = scene_num

		if self._AI2THOR:
			if not self._use_test_scene:
				scene_name = 'FloorPlan_Train' + str(self._scene_type) + '_' + str(self._scene_num)
			else:
				scene_name = 'FloorPlan_Val' + str(self._scene_type) + '_' + str(self._scene_num)
			print('scene_name: ', self._scene_name)
		else:
			scene_name = 'FloorPlan' + str(scene_setting[self._scene_type] + self._scene_num)
		return scene_name

		# Make a random action, all_actions false for only making forward/backward/move_right/move_left move. Weight is probability of actions
	def Random_move_w_weight(self, all_actions=False, weight=None):

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

		success, self._agent_current_pos_index = self.Move_navigation_specialized(self._agent_current_pos_index, direction=TRANSLATION_DIRECTION[action_index], move=True)
		return success, self._agent_current_pos_index

	def Move_navigation_specialized(self, starting_point_index, direction='forward', move=False):
		current_orientation = self.Get_agent_rotation()['y']
		current_position = self._point_list[starting_point_index]

		if direction.lower() == 'forward':
			moving_direction = {0: [0, 0, 1], 1: [1, 0, 0], 2: [0, 0, -1], 3: [-1, 0, 0]}
		if direction.lower() == 'backward':
			moving_direction = {0: [0, 0, -1], 1: [-1, 0, 0], 2: [0, 0, 1], 3: [1, 0, 0]}
		if direction.lower() == 'right':
			moving_direction = {0: [1, 0, 0], 1: [0, 0, -1], 2: [-1, 0, 0], 3: [0, 0, 1]}
		if direction.lower() == 'left':
			moving_direction = {0: [-1, 0, 0], 1: [0, 0, 1], 2: [1, 0, 0], 3: [0, 0, -1]}

		moving_step = {}
		for key in list(moving_direction.keys()):
			moving_step[key] = map(lambda x: x * self._grid_size, moving_direction[key])

		heading_direction_index = int(np.floor((current_orientation - 45) / 90) + 1)
		if current_orientation < 45 or current_orientation > 315:
			heading_direction_index = 0
		moving_point = list(map(lambda x, y: x + y, current_position, moving_step[heading_direction_index]))
		closest_point_index = self.Node_localize(node_position=moving_point)

		if closest_point_index is False:
			return (False, -1)
		if move:
			self.Teleport_agent(position=self._point_list[closest_point_index], save_image=False)
		return (True, closest_point_index)

	def Self_localize(self):
		return self.Node_localize(node_position=self.Get_agent_position())

	def Node_localize(self, node_position):

		node_position_list = self.Get_list_form(pos_or_rot=node_position)
		distance_search_min = np.inf
		nearest_index = -1

		for i, point in enumerate(self._point_list):

			distance_search = np.linalg.norm(np.array(list(map(lambda x, y: x - y, point, node_position_list))))
			if distance_search < distance_search_min:
				distance_search_min = distance_search
				nearest_index = i

		if distance_search_min > 0.5 * self._grid_size:
			logging.warning('Can not find starting point in point list')
			return False
		return nearest_index

	def Update_event(self):
		self._event = self._controller.step('Pass')

	def Open_close_label_text(self):
		if self._use_test_scene:
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

	def Get_list_form(self, pos_or_rot):
		if isinstance(pos_or_rot, dict):
			pos_or_rot_list = list(pos_or_rot.values())
		elif isinstance(pos_or_rot, list):
			pos_or_rot_list = copy.deepcopy(pos_or_rot)
		return pos_or_rot_list

	def Rotate_to_degree(self, goal_degree):
		current_orientation = self.Get_agent_rotation()['y']
		orientation_error = goal_degree - current_orientation
		self.Unit_rotate(orientation_error)

	def Teleport_agent(self, position, save_image=False):
		self.Update_event()

		position_list = self.Get_list_form(pos_or_rot=position)
		self._event = self._controller.step(action='Teleport', x=position_list[0], y=position_list[1], z=position_list[2])

		if save_image:
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
