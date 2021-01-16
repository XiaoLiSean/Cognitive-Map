# Module for iTHOR env set up and simple navigation
from ai2thor.controller import Controller
from termcolor import colored
from PIL import Image
from math import floor, ceil
from matplotlib.patches import Circle, Rectangle, Wedge
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


        self._scene_name = 'FloorPlan_Train4_5'

        self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, visibilityDistance=VISBILITY_DISTANCE, fieldOfView=FIELD_OF_VIEW, applyActionNoise=applyActionNoise)

        self._event = self._controller.step('Pass')


    def update_event(self):
        self._event = self._controller.step('Pass')
        self._SG.reset()
        self._SG.update_from_data(self._event.metadata['objects'])

    def get_agent_position(self):
        self.update_event()
        return self._event.metadata['agent']['position']

    def get_agent_rotation(self):
        self.update_event()
        return self._event.metadata['agent']['rotation']

    def get_reachable_coordinate(self):
        self._event = self._controller.step(action='GetReachablePositions')
        return self._event.metadata['actionReturn']

    # --------------------------------------------------------------------------
    '''
    Following functions is used to visualize map of a certain scene for
    manually topological map construction
    '''
	# --------------------------------------------------------------------------
    def get_scene_bbox(self):
        data = self._event.metadata['sceneBounds']
        center_x = data['center']['x']
        center_z = data['center']['z']
        size_x = data['size']['x']
        size_z = data['size']['z']

        bbox_x = [center_x-size_x*0.5, center_x+size_x*0.5, center_x+size_x*0.5, center_x-size_x*0.5, center_x-size_x*0.5]
        bbox_z = [center_z+size_z*0.5, center_z+size_z*0.5, center_z-size_z*0.5, center_z-size_z*0.5, center_z+size_z*0.5]

        return (bbox_x, bbox_z)

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

        return edges

    # --------------------------------------------------------------------------
    '''
    Following functions is used to visualize map of a certain scene for
    navigation demo
    '''
    # --------------------------------------------------------------------------
    def process_nodes(self, ax, nodes):
        nodes_x = []
        nodes_y = []
        points = nodes
        for idx, p in enumerate(points):
            if idx == 0 or idx == 20:
                circ = Circle(xy = (p[0], p[1]), radius=0.2*self._node_radius, alpha=0.3, color='red')
            else:
                circ = Circle(xy = (p[0], p[1]), radius=0.2*self._node_radius, alpha=0.3, color='green')
            ax.add_patch(circ)
            nodes_x.append(p[0])
            nodes_y.append(p[1])
            ax.text(p[0], p[1], str(idx))

        return (nodes_x, nodes_y)

    def navi_demo(self):
        self.update_event()
        # Plot reachable points
        points = self.get_reachable_coordinate()
        X = [p['x'] for p in points]
        Z = [p['z'] for p in points]

        fig, ax = plt.subplots()

        # Plot rectangle bounding the entire scene
        scene_bbox = self.get_scene_bbox()
        plt.plot(scene_bbox[0], scene_bbox[1], '-', color='orangered', linewidth=4)
        # Overlay map image
        ax.imshow(plt.imread('Demo_IMG/map.png'), extent=[scene_bbox[0][0], scene_bbox[0][1], scene_bbox[1][3], scene_bbox[1][4]])

        plt.plot(X, Z, 'o', color='lightskyblue',
                 markersize=5, linewidth=4,
                 markerfacecolor='white',
                 markeredgecolor='lightskyblue',
                 markeredgewidth=2)

        # # Plot objects 2D boxs
        # for obj in self._event.metadata['objects']:
        #     size = obj['axisAlignedBoundingBox']['size']
        #     center = obj['axisAlignedBoundingBox']['center']
        #     rect = Rectangle(xy = (center['x'] - size['x']*0.5, center['z'] - size['z']*0.5), width=size['x'], height=size['z'], fill=True, alpha=0.3, color='darkgray', hatch='//')
        #     ax.add_patch(rect)

		# Setup plot parameters
        plt.xticks(np.arange(floor(min(scene_bbox[0])/self._grid_size), ceil(max(scene_bbox[0])/self._grid_size)+1, 1) * self._grid_size, rotation=90)
        plt.yticks(np.arange(floor(min(scene_bbox[1])/self._grid_size), ceil(max(scene_bbox[1])/self._grid_size)+1, 1) * self._grid_size)
        plt.xlabel("x coordnates, [m]")
        plt.ylabel("z coordnates, [m]")
        plt.title("{}: Node radius {} [m]".format(self._scene_name, str(self._node_radius)))
        plt.xlim(min(scene_bbox[0])-self._grid_size, max(scene_bbox[0])+self._grid_size)
        plt.ylim(min(scene_bbox[1])-self._grid_size, max(scene_bbox[1])+self._grid_size)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.grid(True)

        nodes_data = [[2.00, -1.50], [2.00, -2.50], [2.00, -3.50], [3.00, -2.50], [3.00, -3.50], [3.00, -4.50],
                      [4.00, -1.50], [4.00, -2.50], [4.00, -4.50], [5.00,-4.50], [6.00, -4.50], [6.00, -2.50],
                      [6.00, -3.25], [7.00, -3.25], [7.00, -2.50], [8.00, -3.25], [8.00, -4.25], [8.00, -2.50],
                      [8.00, -1.50], [9.00, -2.50], [9.00, -1.50]]


        nodes = self.process_nodes(ax, nodes_data)
        plt.plot(nodes[0], nodes[1], 'o', color="None",
                 markersize=5, linewidth=4,
                 markerfacecolor='red',
                 markeredgecolor="None",
                 markeredgewidth=2)

        self.add_edges(nodes_data, ax=ax)

        # ----------------------------------------------------------------------
        # This part is live plot
        # ----------------------------------------------------------------------
        img_num = 116
        imgs_name = ['']*img_num
        pose = [0.0]*img_num
        is_subgoals = [False]*img_num
        subgoals = []
        for filename in os.listdir('Demo_IMG/images/'):
            info = filename.split('_')
            imgs_name[int(info[0])-1] = 'Demo_IMG/images/' + filename
            pose[int(info[0])-1] = [float(info[1]), float(info[2]), float(info[3])]
            if int(info[0]) != 1:
                is_subgoals[int(info[0])-1] = (info[4] == 'True.jpg')

        subgoals_idx = [i for i,value in enumerate(is_subgoals) if value]

        goal_idx = 0

        plt.pause(5)
        for i in range(img_num):
            if i >= 1:
                rob.remove()
                del rob
                current.remove()
                del current

            rob = Image.open('Demo_IMG/robot.png')
            scale = 2*self._grid_size
            rob = ax.imshow(rob.rotate(-pose[i][2]), extent=[pose[i][0] - scale, pose[i][0] + scale, pose[i][1] - scale, pose[i][1] + scale])

            current = Wedge((pose[i][0], pose[i][1]), 1.2*self._grid_size, - pose[i][2] + 90 - 60, - pose[i][2] + 90 + 60, width=self._grid_size, color='red')
            ax.add_patch(current)

            # idx = subgoals_idx[goal_idx]
            #
            # if idx == i:
            #     goal_idx +=1
            #     reached = Wedge((pose[i][0], pose[i][1]), 1.2*self._grid_size, - pose[i][2] + 90 - 60, - pose[i][2] + 90 + 60, width=self._grid_size, color='green')
            #     ax.add_patch(reached)
            #     idx = subgoals_idx[goal_idx]
            #
            # subgoal = Wedge((pose[idx][0], pose[idx][1]), 1.2*self._grid_size, - pose[idx][2] + 90 - 60, - pose[idx][2] + 90 + 60, width=self._grid_size, color='yellow')
            # ax.add_patch(subgoal)

            plt.show(block=False)
            plt.pause(0.05)

        plt.show()

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    robot = Agent_Sim()
    robot._controller.reset('FloorPlan_Train4_5')
    robot.navi_demo()
