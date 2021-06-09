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
import torch

from experiment.node_generation import *
from experiment.experiment_config import *
from Network.action_network.action_network import Action_network
from Network.navigation_network.navigation_network import Navigation_network
from Network.retrieval_network.retrieval_network import Retrieval_network
from Network.retrieval_network.datasets import get_adj_matrix
from lib.scene_graph_generation import Scene_Graph
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

if args.scene_num == 0:
	args.scene_num = random.randint(1, 30)
scene_setting = {1: 0, 2: 200, 3: 300, 4: 400}

log_setting = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO, 5: logging.DEBUG}

logging.basicConfig(level=log_setting[args.log_level])


class Robot():
	def __init__(self, scene_type, scene_num, netName='rnet', save_directory=None, overwrite_data=False, AI2THOR=False, grid_size=0.25, rotation_step=90, sleep_time=0.005, use_test_scene=False, debug=False, server=None, comfirmed=None):

		self._grid_size = grid_size
		self._sleep_time = sleep_time
		self._AI2THOR_controller = AI2THOR_controller(AI2THOR, scene_type, scene_num, grid_size, rotation_step, sleep_time,
													  save_directory, overwrite_data, use_test_scene, debug=debug)
		self.netName = netName
		self.isImageLocalization = (netName != 'rnet')
		self._use_test_scene = use_test_scene
		self._debug = debug

		self._total_action_num = 6
		self._Navigation_max_try = 2*(ADJACENT_NODES_SHIFT_GRID+FORWARD_GRID)

		self.multithread_node = dict(server=server, comfirmed=comfirmed)
		self.image_goal = None
		self.goal_position = None
		self.goal_rotation = None
		self._navinet_collision_by_obstacle = False # boolean variable used to detect collision
		self._passed_position = []
		# self._hardcode_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 3, 3, 3]
		self._hardcode_actions = [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
		self._action_num = 0

		self.Set_navigation_network()
		self.Set_localization_network(self.isImageLocalization)

	def Reset_scene(self, scene_type, scene_num):
		self._AI2THOR_controller.Reset_scene(scene_type=scene_type, scene_num=scene_num)

	def Set_navigation_network(self, network=None):
		if not network is None:
			self._action_network = network
			return
		else:
			self._action_network = Navigation_network(self.netName, isImageNavigation=self.isImageLocalization)

	def Set_localization_network(self, isImageLocalization, network=None):
		if not network is None:
			self._localization_network = network
			return
		else:
			self._localization_network = Retrieval_network(self.netName, isImageLocalization=isImageLocalization)

	def HardCodeLocalization(self, goal_pose=None):
		position_current = list(self.Get_robot_position().values())
		rotation_current = list(self.Get_robot_orientation().values())

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
	# Used to pre-process the features [scene graph, image] for network prediction (localization and navigation)
	def feature_preprocess(self, info):
		if self.isImageLocalization:
			feature = Image.fromarray(info[0])
		else:
			SG = info[1]
			feature = [Image.fromarray(info[0]), get_adj_matrix(SG['on']),
					   get_adj_matrix(SG['in']), get_adj_matrix(SG['proximity']),
					   np.asarray(SG['fractional_bboxs'], dtype=np.float32),
					   np.asarray(SG['vec'].todense(), dtype=np.float32)]

		return copy.deepcopy(feature)

	def Navigation_stop(self, feature_goal, feature_current, goal_pose=None, hardcode=False):

		position_current = list(self.Get_robot_position().values())
		rotation_current = list(self.Get_robot_orientation().values())

		self._localization_network.is_localized(self.feature_preprocess(feature_current),  self.feature_preprocess(feature_goal))

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

		localized = self._localization_network.is_localized(self.feature_preprocess(feature_current),  self.feature_preprocess(feature_goal))

		return localized

	# --------------------------------------------------------------------------
	# Send information to plotter
	# --------------------------------------------------------------------------
	def send_msg_to_client(self, is_reached=False):
		if self.multithread_node['server'] != None:
			current_frame = self._AI2THOR_controller.Get_frame()
			cur_pos = self.Get_robot_position()
			cur_rot = self.Get_robot_orientation()
			info = dict(goal_pose=[self.goal_position[0], self.goal_position[2], self.goal_rotation[1]], goal_img=self.image_goal,
						cur_pose=[cur_pos['x'], cur_pos['z'], cur_rot['y']], cur_img=current_frame, is_reached=is_reached)
			self.multithread_node['server'].send(info)
			while True:
				if self.multithread_node['comfirmed'].value:
					self.multithread_node['comfirmed'].value = 0
					break
	# --------------------------------------------------------------------------

	def Navigate_by_ActionNet(self, image_goal, goal_pose, goal_scene_graph, max_steps, rotation_degree=None):

		is_collision_by_obstacle = False # used to determine if a navigation sequence has collision
		self._navinet_collision_by_obstacle = False

		goal_position = self._AI2THOR_controller.Get_list_form(pos_or_rot=goal_pose['position'])
		goal_rotation = self._AI2THOR_controller.Get_list_form(pos_or_rot=goal_pose['rotation'])

		pre_action = None
		loop_action = [3, 2, 1, 0, 5, 4]
		move_direction = {0: 'forward', 3: 'backward', 4: 'right', 5: 'left'}
		rot_direction = {1: 90, 2: -90}
		move_action = [0, 3, 4, 5]
		rotation_action = [1, 2]

		fail_type = 'navigation'

		self._AI2THOR_controller.Self_localize()

		step = 0
		self._AI2THOR_controller.Update_event()
		current_frame = self._AI2THOR_controller.Get_frame()
		current_scene_graph = self._AI2THOR_controller.Get_SceneGraph()
		feature_current = (current_frame, current_scene_graph)
		feature_goal = (image_goal, goal_scene_graph)

		if not rotation_degree is None:
			rotation_move = False

		while not self.Navigation_stop(feature_goal=feature_goal, feature_current=feature_current, goal_pose=goal_pose, hardcode=False):
			# ------------------------------------------------------------------
			# Send information to plotter
			# ------------------------------------------------------------------
			self.image_goal = image_goal
			self.goal_position = goal_position
			self.goal_rotation = goal_rotation
			self.send_msg_to_client(is_reached=False)
			# ------------------------------------------------------------------
			# ------------------------------------------------------------------

			position_current = list(self.Get_robot_position().values())
			rotation_current = list(self.Get_robot_orientation().values())

			self._passed_position.append(position_current)
			# print('self._passed_position： ', self._passed_position)

			distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, goal_position, position_current))))
			rotation_difference = np.abs(goal_rotation[1] - rotation_current[1])

			if distance < 0.5 * self._grid_size and rotation_difference < 10:
				fail_type = 'localization'

			action_predict = self._action_network.action_prediction(self.feature_preprocess(feature_current),  self.feature_preprocess(feature_goal))

			if not rotation_degree is None and not rotation_move:
				if rotation_degree > 0:
					action_predict[0] = 1
				else:
					action_predict[0] = 2
				rotation_move = True


			if loop_action[action_predict] == pre_action:
				_, self._AI2THOR_controller._agent_current_pos_index = self._AI2THOR_controller.Random_move_w_weight()
				pre_action = None
				continue

			if self._debug:
				print('action_predict: ', action_predict)

			# if self._action_num < len(self._hardcode_actions):
			# 	action_predict = torch.Tensor([self._hardcode_actions[self._action_num]])
			# 	self._action_num += 1

			if action_predict in move_action:
				success, moving_index = self._AI2THOR_controller.Move_navigation_specialized(self._AI2THOR_controller._agent_current_pos_index, direction=move_direction[action_predict.item()], move=True)

				position_current = list(self.Get_robot_position().values())
				# is_collision_by_obstacle = self._AI2THOR_controller._event.metadata['lastActionSuccess']
				# print('not success: ', not success)
				is_collision_by_obstacle = not success
				if is_collision_by_obstacle:
					self._navinet_collision_by_obstacle = True

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
			current_scene_graph = self._AI2THOR_controller.Get_SceneGraph()
			feature_current = (current_frame, current_scene_graph)

			step += 1
			if step >= max_steps:
				return (False, fail_type)

		# ----------------------------------------------------------------------
		# Send information to plotter
		# ----------------------------------------------------------------------
		self.image_goal = image_goal
		self.goal_position = goal_position
		self.goal_rotation = goal_rotation
		self.send_msg_to_client(is_reached=True)
		# ----------------------------------------------------------------------
		# ----------------------------------------------------------------------

		position_current = list(self.Get_robot_position().values())
		rotation_current = list(self.Get_robot_orientation().values())

		distance = np.linalg.norm(np.array(list(map(lambda x, y: x - y, goal_position, position_current))))
		rotation_difference = np.abs(goal_rotation[1] - rotation_current[1])

		if distance > 0.5 * self._grid_size or rotation_difference > 10:
			fail_type = 'localization'
			return (False, fail_type)

		return True

	def Open_close_label_text(self):
		return self._AI2THOR_controller.Open_close_label_text()

	def Get_robot_position(self):
		return self._AI2THOR_controller.Get_agent_position()

	def Get_robot_orientation(self):
		return self._AI2THOR_controller.Get_agent_orientation()


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

		# This two variable is used for localization in dynamics env
		self._CurrentSceneGraph = Scene_Graph()
		self._CurrentImg = []

		if self._AI2THOR:
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE, fieldOfView=FIELD_OF_VIEW, renderObjectImage=True, agentMode='bot')
		else:
			self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE, fieldOfView=FIELD_OF_VIEW, renderObjectImage=True)

		# self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)
		self._save_directory = save_directory

		self._overwrite_data = overwrite_data
		self._event = self._controller.step('Pass')
		self._start_time = time.time()

		self._debug = debug
		self._action_label_text_file = None
		self._action_type = {'INVALID_ACTION': -1, 'MOVE_FORWARD': 0, 'TURN_RIGHT': 1, 'TURN_LEFT': 2, 'MOVE_BACKWARD': 3, 'MOVE_RIGHT': 4, 'MOVE_LEFT': 5}
		self._point_list = []
		self._agent_current_pos_index = None
		self._agent_current_orientation = None

		self._get_reachable_list()
		self.Self_localize()

	def Get_agent_current_pos_index(self):
		return self._agent_current_pos_index

	def Get_agent_current_orientation(self):
		return self._agent_current_orientation

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
			# print('scene_name: ', self._scene_name)
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

		current_position = self._point_list[starting_point_index]

		if direction.lower() == 'forward':
			moving_direction = {0: [0, 0, 1], 90: [1, 0, 0], 180: [0, 0, -1], 270: [-1, 0, 0]}
		if direction.lower() == 'backward':
			moving_direction = {0: [0, 0, -1], 90: [-1, 0, 0], 180: [0, 0, 1], 270: [1, 0, 0]}
		if direction.lower() == 'right':
			moving_direction = {0: [1, 0, 0], 90: [0, 0, -1], 180: [-1, 0, 0], 270: [0, 0, 1]}
		if direction.lower() == 'left':
			moving_direction = {0: [-1, 0, 0], 90: [0, 0, 1], 180: [1, 0, 0], 270: [0, 0, -1]}

		moving_step = {}
		for key in list(moving_direction.keys()):
			moving_step[key] = map(lambda x: x * self._grid_size, moving_direction[key])

		heading_direction = self.Get_current_direction()
		moving_point = list(map(lambda x, y: x + y, current_position, moving_step[heading_direction]))
		closest_point_index = self.Node_localize(node_position=moving_point)

		if closest_point_index is False:
			# return (False, -1)
			return (False, starting_point_index)
		if move:
			self.Teleport_agent(position=self._point_list[closest_point_index], save_image=False)
		return (True, closest_point_index)

	def Get_current_direction(self):

		current_orientation = self.Get_agent_orientation()['y']
		heading_direction_index = int(np.floor((current_orientation - 45) / 90) + 1)
		if current_orientation < 45 or current_orientation > 315:
			heading_direction_index = 0
		return heading_direction_index * 90

	def Self_localize(self, orientation_only=False, direction_correction=False):
		self._agent_current_orientation = self.Get_current_direction()
		if direction_correction:
			self.Unit_rotate(degree=self._agent_current_orientation-self.Get_agent_orientation()['y'])
		if orientation_only:
			return
		self._agent_current_pos_index = self.Node_localize(node_position=self.Get_agent_position())

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
		objs = self._event.metadata['objects']
		self._CurrentImg = self._event.frame
		Img = Image.fromarray(self._CurrentImg, 'RGB')
		self._CurrentSceneGraph.reset()
		objs = self._CurrentSceneGraph.visibleFilter_by_2Dbbox(objs, self._event.instance_detections2D)
		self._CurrentSceneGraph.update_from_data(objs, image_size=Img.size[0])

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

	def Get_agent_orientation(self):
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
		return self._CurrentImg

	def Get_SceneGraph(self):
		self.Update_event()
		return self._CurrentSceneGraph.get_SG_as_dict()

	def Get_list_form(self, pos_or_rot):
		if isinstance(pos_or_rot, dict):
			pos_or_rot_list = list(pos_or_rot.values())
		elif isinstance(pos_or_rot, list):
			pos_or_rot_list = copy.deepcopy(pos_or_rot)
		return pos_or_rot_list

	def Rotate_to_degree(self, goal_degree):
		current_orientation = self.Get_agent_orientation()['y']
		orientation_error = goal_degree - current_orientation
		self.Unit_rotate(orientation_error)

	def Teleport_agent(self, position, position_localize=False, save_image=False):
		self.Update_event()

		position_list = self.Get_list_form(pos_or_rot=position)
		self._event = self._controller.step(action='Teleport', x=position_list[0], y=position_list[1], z=position_list[2])

		if position_localize:
			self.Self_localize()

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
		current_orientation = self.Get_agent_orientation()['y']
		orientation_error = goal_degree - current_orientation
		self.Unit_rotate(orientation_error)

	def Unit_rotate(self, degree):
		if np.abs(degree) < 2:
			return
		degree_corrected = copy.deepcopy(degree)
		while degree_corrected > 180:
			degree_corrected -= 360
		while degree_corrected < -180:
			degree_corrected += 360
		if degree_corrected > 0:
			self._event = self._controller.step(action='RotateRight', degrees=np.abs(degree_corrected))
			self.Self_localize(orientation_only=True)
			return 'TURN_RIGHT'
		else:
			self._event = self._controller.step(action='RotateLeft', degrees=np.abs(degree_corrected))
			self.Self_localize(orientation_only=True)
			return 'TURN_LEFT'
