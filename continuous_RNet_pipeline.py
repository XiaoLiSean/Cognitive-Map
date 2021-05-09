'''
Retrieval Network Testing in Continuous World, Written by Xiao
For robot localization in a dynamic environment.
'''
# Import params and similarity from lib module
import torch
import argparse, os, copy, pickle, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from progress.bar import Bar
from termcolor import colored
from ai2thor.controller import Controller
from lib.scene_graph_generation import Scene_Graph
from lib.object_dynamics import shuffle_scene_layout
from lib.params import VISBILITY_DISTANCE, SCENE_TYPES, SCENE_NUM_PER_TYPE, NODES, FIELD_OF_VIEW
from Network.retrieval_network.retrieval_network import Retrieval_network
from Network.retrieval_network.datasets import get_adj_matrix
from Network.retrieval_network.params import TRAIN_FRACTION, VAL_FRACTION, DATA_DIR, CHECKPOINTS_DIR
from os.path import dirname, abspath

XZ_RANGE = (-4*0.25, 4*0.25) # range of 2*2 m^2 field
THETA_RANGE = (-180, 180) # rotation range
DXDZ = 0.05 # traverse movement of minimum step 0.01 m
DTHETA = 5 # rotation movement of minimum step 1 deg


'''
New Defined agent operated in a finer grid world (0.005 m)
'''
class Robot():
    def __init__(self, scene_type='Kitchen', scene_num=1, grid_size=DXDZ, applyActionNoise=False, renderObjectImage=True):

        self._scene_type = scene_type
        self._scene_num = scene_num
        self._grid_size = grid_size
        # Used to collect data points
        self._SG = Scene_Graph()
        self._img = []
        self._bbox = []
        self.RNet = Retrieval_network()

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

        self._controller = Controller(scene=self._scene_name, snapToGrid=False, gridSize=self._grid_size,
                                      applyActionNoise=applyActionNoise, renderObjectImage=renderObjectImage)

        self._event = self._controller.step('Pass')
        self.global_y = self._event.metadata['agent']['position']['y']

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

    # --------------------------------------------------------------------------
    # get current info as tuple_list
    def get_current_info(self):
        self.update_event()
        current_info = copy.deepcopy([self._img, get_adj_matrix(self._SG._R_on), get_adj_matrix(self._SG._R_in),
                                      get_adj_matrix(self._SG._R_proximity), np.asarray(self._SG._fractional_bboxs, dtype=np.float32),
                                      np.asarray(self._SG._obj_vec.todense(), dtype=np.float32)])
        return current_info

    # --------------------------------------------------------------------------
    # define scene Bounds
    def in_scene_bbox(self, x, z):
        data = self._event.metadata['sceneBounds']

        center_x = data['center']['x']
        center_z = data['center']['z']
        size_x = data['size']['x']
        size_z = data['size']['z']
        is_in = (x > center_x-size_x*0.5) and (x < center_x+size_x*0.5) and (z > center_z-size_z*0.5) and (z < center_z+size_z*0.5)

        return is_in

    # --------------------------------------------------------------------------
    # Update testing staticstics near a node
    def updateStatistics(self, staticstics):
        goal_infos = []
        nominal_theta = 180
        for node in NODES[self._scene_name]:
            nominal_x = node[0]
            nominal_z = node[1]
            #global_y = self._event.metadata['agent']['position']['y']
            event = self._controller.step(action='Teleport', position=dict(x=nominal_x, y=self.global_y, z=nominal_z), rotation=dict(x=0.0, y=nominal_theta, z=0.0))
            if not event.metadata['lastActionSuccess']:
                goal_infos.append([])
            else:
                goal_infos.append(self.get_current_info())

        shuffle_scene_layout(self._controller)

        for idx, node in enumerate(NODES[self._scene_name]):
            goal_info = goal_infos[idx]
            if len(goal_info) == 0:
                continue
            nominal_x = node[0]
            nominal_z = node[1]
            for dx in np.arange(XZ_RANGE[0], XZ_RANGE[1]+DXDZ, DXDZ):
                idx_x = int((dx-XZ_RANGE[0])/DXDZ)
                for dz in np.arange(XZ_RANGE[0], XZ_RANGE[1]+DXDZ, DXDZ):
                    idx_z = int((dz-XZ_RANGE[0])/DXDZ)
                    event = self._controller.step(action='Teleport', position=dict(x=nominal_x+dx, y=self.global_y, z=nominal_z+dz), rotation=dict(x=0.0, y=nominal_theta, z=0.0))
                    if not event.metadata['lastActionSuccess']:
                        continue
                    for dtheta in np.arange(THETA_RANGE[0], THETA_RANGE[1]+DTHETA, DTHETA):
                        idx_t = int((dtheta-THETA_RANGE[0])/DTHETA)
                        event = self._controller.step(action='Teleport', position=dict(x=nominal_x+dx, y=self.global_y, z=nominal_z+dz), rotation=dict(x=0.0, y=nominal_theta+dtheta, z=0.0))
                        if not event.metadata['lastActionSuccess']:
                            continue
                        current_info = self.get_current_info()
                        similarity = self.RNet.get_similarity(current_info, copy.deepcopy(goal_info))
                        staticstics['n'][idx_x, idx_z, idx_t] += 1
                        staticstics['sum'][idx_x, idx_z, idx_t] += similarity
                        staticstics['sq_sum'][idx_x, idx_z, idx_t] += similarity**2

            return # one node a scene

# ------------------------------------------------------------------------------
# ---------------------------Thresholding and Heatmap---------------------------
# ------------------------------------------------------------------------------
def degradeTo2D(staticstics):
    size = staticstics['n'].shape
    x0 = int(size[0]*0.5)
    z0 = int(size[1]*0.5)
    t0 = int(size[2]*0.5)
    degraded = []
    for ti in range(size[2]):
        degraded.append(dict(distance=[],n=[],sum=[],sq_sum=[]))
        for xi in range(size[0]):
            for zi in range(size[1]):
                distance = DXDZ*((xi-x0)**2+(zi-z0)**2)**0.5
                if distance in degraded[ti]['distance']:
                    idx = degraded[ti]['distance'].index(distance)
                    degraded[ti]['n'][idx] += staticstics['n'][xi,zi,ti]
                    degraded[ti]['sum'][idx] += staticstics['sum'][xi,zi,ti]
                    degraded[ti]['sq_sum'][idx] += staticstics['sq_sum'][xi,zi,ti]
                else:
                    degraded[ti]['distance'].append(distance)
                    degraded[ti]['n'].append(staticstics['n'][xi,zi,ti])
                    degraded[ti]['sum'].append(staticstics['sum'][xi,zi,ti])
                    degraded[ti]['sq_sum'].append(staticstics['sq_sum'][xi,zi,ti])

    for ti in range(size[2]):
        num = np.asarray(copy.deepcopy(degraded[ti]['n']))
        sum = np.asarray(degraded[ti]['sum'])
        num[sum == 0] = 1.0
        mean = np.true_divide(sum, num)
        sq_sum = np.asarray(degraded[ti]['sq_sum'])
        std = np.true_divide(sq_sum, num) - np.true_divide(np.power(sum, 2), np.power(num, 2))
        sigma = np.power(std, 0.5)
        degraded[ti].update(mean=mean)
        degraded[ti].update(sigma=sigma)

    return degraded

def plot_heatmap():
    staticstics = np.load(CHECKPOINTS_DIR + 'continues_heatmap_staticstics.npy', allow_pickle=True).item()
    size = staticstics['n'].shape
    staticstics = degradeTo2D(staticstics)

    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    theta = []
    distance = []
    mean = []
    sigma = []

    for ti in range(size[2]):
        theta_i = (ti -int(size[2]*0.5))*DTHETA
        distance_i = staticstics[ti]['distance']
        theta.append(theta_i*np.ones_like(distance_i))
        distance.append(distance_i)
        mean.append(staticstics[ti]['mean'])
        sigma.append(staticstics[ti]['sigma'])

    distance = np.asarray(distance)
    theta = np.asarray(theta)
    mean = np.asarray(mean)
    sigma = np.asarray(sigma)

    surfmean = ax1.plot_surface(theta*np.ones_like(distance), distance, mean, cmap=cm.coolwarm)
    ax1.set_xlabel(r'$\Delta \theta$')
    ax1.set_xlabel(r'$\Delta r$')
    ax1.set_xlabel('mean similarity')
    fig.colorbar(surfmean, ax=ax1, shrink=0.6)
    surfsigma = ax2.plot_surface(theta*np.ones_like(distance), distance, sigma, cmap=cm.coolwarm)
    ax2.set_xlabel(r'$\Delta \theta$')
    ax2.set_xlabel(r'$\Delta r$')
    ax2.set_xlabel('similarity standard deviation')
    fig.colorbar(surfsigma, ax=ax2, shrink=0.6)
    plt.show()


def thresholding():
    # ------------------------------Thresholding Main---------------------------
    test_idx_initial = int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1
    test_idx_end = SCENE_NUM_PER_TYPE + 1
    template = np.zeros((2*int(XZ_RANGE[1]/DXDZ)+1, 2*int(XZ_RANGE[1]/DXDZ)+1, 2*int(THETA_RANGE[1]/DTHETA)+1))
    staticstics = dict(n=copy.deepcopy(template), sum=copy.deepcopy(template), sq_sum=copy.deepcopy(template))
    for scene_type in SCENE_TYPES:
        for scene_num in range(test_idx_initial, test_idx_end):
            robot = Robot(scene_type=scene_type, scene_num=scene_num)
            robot.updateStatistics(staticstics)
            #np.save(CHECKPOINTS_DIR + 'continues_heatmap_staticstics.npy', staticstics)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # plot_heatmap()
    # exit()
    thresholding()
