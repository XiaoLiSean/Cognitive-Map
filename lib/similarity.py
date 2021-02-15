# Module for caculate view and scene graph similarity scores \in (0,1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from termcolor import colored
from matplotlib.collections import PatchCollection
from numpy import matlib as mb
from lib.params import *

# color map and setting for plot
COLOR_REGULAR = 'lightgrey'
COLOR_INTERCEPT = 'red'
MARKERSIZE = 1

# ------------------------------------------------------------------------------
# wrap angle of degree into range of (-pi,pi)
def wrap_to_pi(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

# ------------------------------------------------------------------------------
# Fuction to calculate if dot in Robot's fieldOfView
# pose is a dictionary of {x: ... meter, z: ... meter, theta: ... degree}
# dot is a array of X, Z coordinates [x, z] just like [x,y] in 2D
# return boolean
def in_FoV(pose, dot):
    is_in_FoV = True
    distance = np.linalg.norm(np.array([pose['x'], pose['z']]) - np.array(dot))
    off_angle = np.arctan2(dot[1] - pose['z'], dot[0] - pose['x']) * 180 / np.pi
    if (distance > VISBILITY_DISTANCE) or (abs(wrap_to_pi(90-off_angle-pose['theta'])) > FIELD_OF_VIEW / 2.0):
        is_in_FoV = False
    return is_in_FoV

# ------------------------------------------------------------------------------
# Function calculate view fans similarity (2d FoV horizon in horizontal direction)
# Note: In AI2THOR, z is the forward axis, x is the horizon axis and y is the verticle axis
# pose is a dictionary of {x: ... meter, z: ... meter, theta: ... degree}
def view_similarity(pose1, pose2, visualization_on=False):
    # special case/ hard coding filter
    angle_diff = abs(pose1['theta']-pose2['theta'])
    distance = ((pose1['x']-pose2['x'])**2 + (pose1['z']-pose2['z'])**2)**0.5
    if (angle_diff - 360 / SUB_NODES_NUM) >= 0 or distance >= VISBILITY_DISTANCE:
        return 0

    grid_size = VISBILITY_DISTANCE * 2 / 10**SIMILARITY_GRID_ORDER # distance between two dots
    X = np.arange(-VISBILITY_DISTANCE, VISBILITY_DISTANCE + grid_size, grid_size)
    dots_num_sqrt = len(X)
    X = mb.repmat(X, dots_num_sqrt, 1) + pose1['x'] # this is a dots_num_sqrt^2 dots field for their x coordinates
    Z = np.arange(VISBILITY_DISTANCE, -VISBILITY_DISTANCE - grid_size, -grid_size)
    Z = np.transpose(mb.repmat(Z, dots_num_sqrt, 1)) + pose1['z']# this is a dots_num_sqrt^2 dots field for their z coordinates
    # X, Z form a dots field of size dots_num_sqrt^2 where X, Z are their coordinates
    # used as test field for pose1 against pose2
    num_in_FoV1 = 0
    num_in_intercept = 0
    # Used for visualization
    if visualization_on:
        fig, ax = plt.subplots()
        wedge1 = Wedge((pose1['x'], pose1['z']), VISBILITY_DISTANCE,
                       90-pose1['theta']-FIELD_OF_VIEW/2.0, 90-pose1['theta']+FIELD_OF_VIEW/2.0)
        wedge2 = Wedge((pose2['x'], pose2['z']), VISBILITY_DISTANCE,
                       90-pose2['theta']-FIELD_OF_VIEW/2.0, 90-pose2['theta']+FIELD_OF_VIEW/2.0)
        patches = [wedge1, wedge2]
        p = PatchCollection(patches, alpha=0.2)
        p.set_array(np.array([0.0, 0.5]))
        ax.add_collection(p)
    # main algorithm to estimate the similarity score
    for i in range(dots_num_sqrt):
        for j in range(dots_num_sqrt):
            dot = [X[i,j], Z[i,j]]
            if in_FoV(pose1, dot): # dot in FoV of robot in pose1
                num_in_FoV1 = num_in_FoV1 + 1
                if in_FoV(pose2, dot): # dot also in FoV of robot in pose1
                    num_in_intercept = num_in_intercept + 1
                    if visualization_on:
                        plt.plot(X[i,j], Z[i,j], '.', color=COLOR_INTERCEPT, markersize=MARKERSIZE)
                else:
                    if visualization_on:
                        plt.plot(X[i,j], Z[i,j], '.', color=COLOR_REGULAR, markersize=MARKERSIZE)
    # main algorithm to estimate the similarity score
    if visualization_on:
        plt.axis('equal')
        plt.xlabel('X (Horizon)')
        plt.ylabel('Z (Forward)')
        plt.show()
    # similarity calculation
    return num_in_intercept / num_in_FoV1

# ------------------------------------------------------------------------------
# This function is used to load Glove vector
def load_glove():
    print(colored('INFO: ','blue') + "Loading GloVe vectors")
    FILE = THIRD_PARTY_PATH + '/' + GLOVE_FILE_NAME
    f = open(FILE,'r',encoding='utf-8')
    glove_embedding = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        glove_embedding[word] = wordEmbedding
    print(colored('INFO: ','blue') + "Saving GloVe vectors as glove_embedding.npy")
    np.save(THIRD_PARTY_PATH + '/' + 'glove_embedding.npy', glove_embedding) # Save list as .npy
    print(colored('INFO: ','blue') + "Done")


# ------------------------------------------------------------------------------
# This function is used to pair Glove word to AI2THOR objectTypes
def thor_2_glove():
    THOR_2_GLOVE = {}
    words = glove_embedding.keys()
    for objectType in idx_2_obj_list:
        for word in words:
            if objectType.lower() == word.lower():
                THOR_2_GLOVE.update({objectType: word})
        if objectType not in THOR_2_VEC:
            THOR_2_GLOVE.update({objectType: 'None'})

    # print(THOR_2_GLOVE)
    np.save(THIRD_PARTY_PATH + '/' + 'THOR_2_GLOVE.npy', THOR_2_GLOVE)

# ------------------------------------------------------------------------------
# This function is used to process objectType with GLOVE vectors
def thor_2_vec():
    THOR_2_VEC = {}
    for objectType in idx_2_obj_list:
        vec = glove_embedding[THOR_2_GLOVE[objectType]]
        THOR_2_VEC.update({objectType: vec})
    np.save(THIRD_PARTY_PATH + '/' + 'THOR_2_VEC.npy', THOR_2_VEC)
    # print(len(THOR_2_VEC))
    # print(THOR_2_VEC)
