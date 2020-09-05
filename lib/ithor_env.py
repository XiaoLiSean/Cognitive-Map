 # Module for iTHOR env set up and simple navigation
from ai2thor.controller import Controller
from termcolor import colored
from dijkstar import Graph, find_path
# from lib.scene_graph_generation import *
from lib.params import SIM_WINDOW_HEIGHT, SIM_WINDOW_WIDTH, VISBILITY_DISTANCE, FIELD_OF_VIEW
import matplotlib.pyplot as plt
import numpy as np
import time, copy, sys

class Agent_Sim():
    def __init__(self, scene_type, scene_num, grid_size=0.25, rotation_step=10, sleep_time=0.05, ToggleMapView=False):
        self._scene_type = scene_type
        self._scene_num = scene_num
        self._grid_size = grid_size
        self._rotation_step = rotation_step
        self._sleep_time = sleep_time

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
        self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE, fieldOfView=FIELD_OF_VIEW)
        self._controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)  # Change simulation window size

        if ToggleMapView:   # Top view of the map to see the objets layout. issue: no SG can be enerated
            self._controller.step({"action": "ToggleMapView"})

        self._event = self._controller.step('Pass')
        self._start_time = time.time()
        self._action_type = {'MOVE_FORWARD': 1, 'STAY_IDLE' :2, 'TURN_RIGHT' :3, 'TURN_LEFT': 4}

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

    def get_object(self):
        self.update_event()
        return self._event.metadata['objects']

    def unit_move(self):
        self._event = self._controller.step(action='MoveAhead')
        return 'MOVE_FORWARD'

    def unit_rotate(self, degree):
        if np.abs(degree) < 2:
            print(colored('INFO: ','blue') + 'Robot rotate for {} degree which is less than 2 deg'.format(degree))
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

    # Assume goal is {'position': position, 'rotation': rotation} where position and rotation are dict or list
    def move_towards(self, goal):
        self.update_event()
        agent_position = self.get_agent_position()
        agent_rotation = self.get_agent_rotation()

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
            sys.stderr.write(colored('ERROR: ','red')
                             + 'Moving step {} greater than grid size {}'.format(position_error, self._grid_size))
            sys.exit(1)
        elif np.linalg.norm(np.array(position_error)) < self._grid_size * 0.10:
            sys.stderr.write(colored('ERROR: ','red')
                             + 'Moving distance {} too small'.format(position_error))
            sys.exit(1)

        rotate_steps = int(np.abs(rotation_error_corrected / self._rotation_step))

        for _ in range(rotate_steps):
            time.sleep(self._sleep_time)
            action = self.unit_rotate(self._rotation_step * np.sign(rotation_error_corrected))

        action = self.unit_rotate((rotation_error_corrected - rotate_steps * self._rotation_step * np.sign(rotation_error_corrected)))

        time.sleep(self._sleep_time)
        action = self.unit_move()

class Dumb_Navigetor():
    def __init__(self, agent_sim):
        self._map = {}
        self._point_list = []
        self._grid_size = agent_sim._grid_size
        self._point_num = 0
        self._agent_sim = agent_sim
        self._starting_point = self._agent_sim.get_agent_position()
        self._coordinate_dict = self._agent_sim.get_reachable_coordinate()
        self._map_searched = [True] * len(self._coordinate_dict)
        self._build_map()

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
                    self._map[self._point_num - 1].append(point_added_index)
                    self._map[point_added_index].append(self._point_num - 1)
        return

    # Assume goal_position is dict
    def dumb_navigate(self, goal_position, server=None, comfirmed=None):
        print(colored('Dumb navigate to: {}','cyan').format(goal_position))
        # server and comfirm is not none --> this function is used as a server node
        graph = Graph()
        nav_starting_point = self._agent_sim.get_agent_position()
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
            sys.stderr.write(colored('ERROR: ','red') + 'No matching point in map' + '\n')
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
                sys.stderr.write(colored('ERROR: ','red') + 'Can not reach the point by existing map' + '\n')
                return

        for index in range(len(self._map)):
            for connected_index in range(len(self._map[index])):
                if self._map_searched[self._map[index][connected_index]]:
                    graph.add_edge(index, self._map[index][connected_index], 1)

        result = find_path(graph, nav_starting_point_index, nearest_reachable_index)

        path = result.nodes

        for mid_point_index in range(1, len(path)):
            # This navigator serve as a server node if server is not None
            if server is not None:
                objs = [obj for obj in self._agent_sim._event.metadata['objects'] if obj['visible']]
                print(self._agent_sim._event.metadata['agent'])
                server.send(objs)
                print(colored('Server: ','cyan') + 'Sent Data from navigator at mid_point_index {}'.format(mid_point_index))
                while True:  # Waiting for client to confirm
                    if comfirmed.value:
                        break
                comfirmed.value = 0  # Turn off the switch

            # Action
            mid_point_pose = {'position': [], 'rotation': []}
            mid_point_pose['position'] = copy.deepcopy(self._point_list[path[mid_point_index]])
            mid_point_pose['rotation'] = [0, 0, 0]
            self._agent_sim.move_towards(mid_point_pose)

        # Terminate the service by sending 'END'
        if server is not None:
            server.send('END')
            print(colored('Server: ','cyan') + 'END')

        if not goal_in_existing_map:
            self._agent_sim.move_towards({'position': copy.deepcopy(self._point_list[goal_point_index]), 'rotation': [0, 0, 0]})
            self._map_searched[goal_point_index] = True

        return
