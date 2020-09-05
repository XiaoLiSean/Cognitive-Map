# Module for caculate view and scene graph similarity scores \in (0,1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from numpy import matlib as mb
from params import VISBILITY_DISTANCE, FIELD_OF_VIEW, SIMILARITY_GRID_ORDER

# color map and setting for plot
COLOR_REGULAR = 'lightgrey'
COLOR_INTERCEPT = 'red'
MARKERSIZE = 1

# wrap angle of degree into range of (-pi,pi)
def wrap_to_pi(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

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

# Function calculate view fans similarity (2d FoV horizon in horizontal direction)
# Note: In AI2THOR, z is the forward axis, x is the horizon axis and y is the verticle axis
# pose is a dictionary of {x: ... meter, z: ... meter, theta: ... degree}
def view_similarity(pose1, pose2, visualization_on=False):
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
        p = PatchCollection(patches, alpha=0.1)
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


if __name__ == '__main__':
    # Test case
    pose1 = {'x': -1.0, 'theta': 270.0, 'z': 1.0}
    pose2 = {'x': -1.0, 'theta': 180.0, 'z': 0.75}
    pose3 = {'x': -1.0, 'theta': 180.0, 'z': 0.5}
    print(view_similarity(pose2, pose3, visualization_on=True))
