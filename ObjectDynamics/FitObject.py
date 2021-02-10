from ai2thor.controller import Controller
import numpy as np
from scipy.spatial.transform import Rotation as R

# constraints are just dimensions for each axis projection
# we want to get a bounding box approximating the object relative to the robot, as well as the robot
# we are assuming that the center of the robot is zero/zero

#First, we want to check and see if, in our default configuration of our object, if we are outside of the dimensions
# of our constraints (AKA: object too tall, too wide, too thick, ect)
# if this is in fact the case, then we want to rotate in an attempt to fix this problem
# I think we want to kind of like, binary search this.
# If our binary rotation fails, then we fail loud and just ... try another passage way

def recenterPoints(robotCenter, boundingBoxCorner):
    points = []
    for p in boundingBoxCorner:
        points.append(p-robotCenter)

    return points

def inDims(event, objIdx, minDims):
    boundingBox = recenterPoints(event.metadata['agent']['position'], event.metadata['objects'][objIdx])
    dimensionsViolated = []
    for p in boundingBox:
        for i in range(0,len(p)):
           if i  not in dimensionsViolated and minDims[i] < p[i]:
               dimensionsViolated.append(i)
    return dimensionsViolated

def calcMag(vals):
    sum = 0
    for v in vals:
        sum += v*v
    return np.sqrt(sum)

def create_transform(event):
    pos = event.metadata['agent']['position']
    rot = event.metadata['agent']['rotation']
    rot_mat = R.from_rotvec([rot['x'], rot['y'], rot['z']])
    out =np.zeros((4,4))
    out[0:3,0:3] = rot_mat.as_matrix()
    out[0,3] = pos['x']
    out[1,3] = pos['y']
    out[2,3] = pos['z']
    out[3,3] = 1
    return out



def fitObject(controller, maxDims):
    event = controller.step("Pass")

    robot_centric = create_transform(event)

    objIdx = None
    for i in range(0, len(event.metadata['objects'])):
        if event.metadata['objects'][i]['isPickedUp']:
            objIdx = i
            break

    bounding_box = event.metadata['objects'][objIdx]['axisAlignedBoundingBox']
    object_dims = [bounding_box['size']['x'], bounding_box['size']['y'],bounding_box['size']['z']]

    # check trivial case

    if (object_dims < maxDims):
        print("No realign")
        return

    # check axis aligned case
    ordered_obj_dims = object_dims.copy()
    ordered_max_dims = maxDims.copy()
    ordered_obj_dims.sort(reverse=True)
    ordered_max_dims.sort(reverse=True)

    if (ordered_obj_dims < ordered_max_dims):
        print("axis realign")
        # do rotation to fit here
        dim_object_ordering = [0,0,0]
        for i in range(0, 3):
            for j in range(0,3):
                if object_dims[i] == ordered_obj_dims[j]:
                    dim_object_ordering[i] = j

        constraint_dim_ordering = [0,0,0]
        for i in range(0, 3):
            for j in range(0,3):
                if maxDims[i] == ordered_max_dims[j]:
                    constraint_dim_ordering[i] = j

        if dim_object_ordering[0] == constraint_dim_ordering[0]:
            controller.step("RotateHandRelative", x=90, raise_for_failure=True)
            return
        elif dim_object_ordering[1] == constraint_dim_ordering[1]:
            controller.step("RotateHandRelative", y=90,raise_for_failure=True)
            return
        elif dim_object_ordering[2] == constraint_dim_ordering[2]:
            controller.step("RotateHandRelative", z=90,raise_for_failure=True)
            return

        print("All offset")

        #if they're all offset
        if dim_object_ordering[-1] == 0 and dim_object_ordering[0] == 1:
            controller.step("RotateHandRelative", x=90)
            controller.step("RotateHandRelative", z=90)
        if dim_object_ordering[-1] == 0 and dim_object_ordering[0] == 2:
            controller.step("RotateHandRelative", x=90)
            controller.step("RotateHandRelative", y=90)
        if dim_object_ordering[-1] == 1 and dim_object_ordering[0] == 0:
            controller.step("RotateHandRelative", y=90)
            controller.step("RotateHandRelative", z=90)
        if dim_object_ordering[-1] == 1 and dim_object_ordering[0] == 2:
            controller.step("RotateHandRelative", y=90)
            controller.step("RotateHandRelative", x=90)
        if dim_object_ordering[-1] == 2 and dim_object_ordering[0] == 0:
            controller.step("RotateHandRelative", z=90)
            controller.step("RotateHandRelative", y=90)
        if dim_object_ordering[-1] == 2 and dim_object_ordering[0] == 1:
            controller.step("RotateHandRelative", z=90)
            controller.step("RotateHandRelative", x=90)


        return


    # check longest distance case
    obj_longest = calcMag(object_dims)
    longest_dims = calcMag(maxDims)
    shortEnd = calcMag(object_dims[1:2])

    if obj_longest+shortEnd < longest_dims:
        print("corner realign")
        xaxis_rot = np.atan2(maxDims[1], maxDims[2])
        yaxis_rot = np.atan2(maxDims[2], maxDims[0])
        zaxis_rot = np.atan2(maxDims[1], maxDims[0])

        controller.step("RotateObjectRelative", x=xaxis_rot, y = yaxis_rot, z=zaxis_rot)

        return

    print("no flags")





