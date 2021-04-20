# Module for iTHOR env set up and simple navigation
from ai2thor.controller import Controller
from termcolor import colored
from PIL import Image
from progress.bar import Bar
from math import floor, ceil
from scipy.sparse import lil_matrix
from matplotlib.patches import Circle, Rectangle
from lib.params import SIM_WINDOW_HEIGHT, SIM_WINDOW_WIDTH, VISBILITY_DISTANCE, FIELD_OF_VIEW, DOOR_NODE, SUB_NODES_NUM
from lib.scene_graph_generation import Scene_Graph
from lib.object_dynamics import shuffle_scene_layout
import matplotlib.pyplot as plt
import numpy as np
import time, copy, sys, random, os

# Class for agent and nodes in simulation env
class Robot():
    def __init__(self, scene_type='Kitchen', scene_num=1, grid_size=0.25,
    			 node_radius=VISBILITY_DISTANCE, default_resol=True, ToggleMapView=False,
    			 applyActionNoise=False, renderObjectImage=True):
        self._scene_type = scene_type
        self._scene_num = scene_num
        self._grid_size = grid_size
        self._node_radius = node_radiusxlabel
        # Used to collect data points
        self._SG = Scene_Graph()
        self._img = []
        self._bbox = []

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

        self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE,
        							  fieldOfView=FIELD_OF_VIEW, applyActionNoise=applyActionNoise, renderObjectImage=renderObjectImage)

        if not default_resol:
        	self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)  # Change simulation window size

        if ToggleMapView:   # Top view of the map to see the objets layout. issue: no SG can be enerated
        	self._controller.step({"action": "ToggleMapView"})
        	self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)  # Change simulation window size

        self._event = self._controller.step('Pass')

    # --------------------------------------------------------------------------
    # Update data [img, bbox, SG] of current frame
    def update_event(self):
        self._event = self._controller.step('Pass')
        objs = self._event.metadata['objects']
        frame = self._event.frame
        self._img = Image.fromarray(frame, 'RGB')
        self._SG.reset()
        objs = self._SG.visibleFilter_by_2Dbbox(objs, self._event.instance_detections2D)
        self._SG.update_from_data(objs, image_size=self._img.size[0])
        is_empty = (len(objs) <= 0)
        return is_empty

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
                self._controller.step(action='Teleport', position=dict(x=node[0], y=self.get_agent_position()['y'], z=node[1]), rotation=dict(x=0.0, y=subnode, z=0.0))
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
    Following functions is used to collect data with and without dynamcis
    for localization/retrieval network
    '''
	# --------------------------------------------------------------------------

    def collect_data_point(self, FILE_PATH, nominal_i, nominal_j, array_map, nominal_theta,  ignore_empty=False):
        is_empty = self.update_event()
        if ignore_empty and is_empty:
            return array_map
        array_map[nominal_i, nominal_j] += 1
        file_name = str(nominal_theta) + '_' + str(nominal_i) + '_' + str(nominal_j) + '_' + str(array_map[nominal_i, nominal_j])
        sg_data = self._SG.get_SG_as_dict()
        # self._SG.visualize_data_point(self._img) # used to visualize the date point
        self._img.save(FILE_PATH + '/' + file_name + '.png')
        np.save(FILE_PATH + '/' + file_name + '.npy', sg_data)

        return array_map

    def get_scene_bbox(self):
        self.update_event()
        data = self._event.metadata['sceneBounds']
        center_x = data['center']['x']
        center_z = data['center']['z']
        size_x = data['size']['x']
        size_z = data['size']['z']
        bbox = [center_x-size_x*0.5, center_z-size_z*0.5, center_x+size_x*0.5, center_z+size_z*0.5]

        return bbox

    def generate_index_in_array_map(self, coordinates):
        bbox = self.get_scene_bbox()
        i = round((coordinates[0] - bbox[0]) / self._grid_size)
        j = round((coordinates[1] - bbox[1]) / self._grid_size)

        return i, j

    def get_array_map(self):
        self.update_event()
        data = self._event.metadata['sceneBounds']
        size_x = data['size']['x']
        size_z = data['size']['z']
        rows = round(size_x / self._grid_size)
        cols = round(size_z / self._grid_size)
        array_map = lil_matrix((rows, cols), dtype=np.int)

        return array_map

    def get_noisy_target_pose(self, x, z, theta):
        sigma_x = self._grid_size / 12
        sigma_z = self._grid_size / 12
        sigma_theta = 360 / (30*SUB_NODES_NUM)
        noise_x = np.clip(np.random.normal(0.0, sigma_x), a_min = -1*sigma_x, a_max = 1*sigma_x)
        noise_z = np.clip(np.random.normal(0.0, sigma_z), a_min = -1*sigma_z, a_max = 1*sigma_z)
        noise_theta = np.clip(np.random.normal(0.0, sigma_theta), a_min = -1*sigma_theta, a_max = 1*sigma_theta)
        rotation = dict(x=0.0, y=theta+noise_theta, z=0.0)

        return x+noise_x, z+noise_z, rotation

    def sparsify_reachable_points(self, reachable_points, reserved_fraction=0.5):
        random.shuffle(reachable_points)
        cufoff_idx = round(len(reachable_points)*reserved_fraction)
        new_reachable_points = copy.deepcopy(reachable_points[0:cufoff_idx])
        return new_reachable_points


    def coordnates_patroling(self, saving_data=False, file_path=None, dynamics_rounds=5, pertubation_round=3):
        rotations = [0.0, 90.0, 180.0, 270.0]
        array_maps = {'0.0':self.get_array_map(), '90.0':self.get_array_map(), '180.0':self.get_array_map(), '270.0':self.get_array_map()}

        # define file path
        file_path = file_path + '/' + self._scene_name
        if not os.path.isdir(file_path):
        	os.mkdir(file_path)
        else:
        	# case the dataset is already exists
        	# comment out to allow incrementally update the dataset
        	print('Skip: ', self._scene_name)
        	return

        print('----'*10)
        for round in range(dynamics_rounds):
        	# change obj layout for the 2-end rounds
            if round != 0:
                # change object layout
                shuffle_scene_layout(self._controller)
                self.update_event()

        	# random get fractional points as node in train and validation scene
            reachable_points = self.get_reachable_coordinate()
            points = copy.deepcopy(self.sparsify_reachable_points(reachable_points))

            # ------------------------------------------------------------------
            # Start collecting data
            bar = Bar('{}: {}/{} round'.format(self._scene_name, round+1, dynamics_rounds), max=len(points)*len(rotations)*pertubation_round)
        	# store image and SG
            for p in points:
                nominal_x = p['x']
                nominal_z = p['z']
                nominal_i, nominal_j = self.generate_index_in_array_map([nominal_x, nominal_z])
                for nominal_theta in rotations:
                    for _ in range(pertubation_round):
                        pose_x, pose_z, pose_r = self.get_noisy_target_pose(nominal_x, nominal_z, nominal_theta)
                        event = self._controller.step(action='Teleport', position=dict(x=pose_x, y=p['y'], z=pose_z), rotation=pose_r)

                        if not event.metadata['lastActionSuccess']:
                            continue

                        if saving_data:
                            array_maps[str(nominal_theta)] = self.collect_data_point(file_path, nominal_i, nominal_j, array_maps[str(nominal_theta)], nominal_theta, ignore_empty=True)
                            bar.next()
                        else:
                            self.update_event()
            bar.finish()

        np.save(file_path + '/' + 'array_maps.npy', array_maps)
