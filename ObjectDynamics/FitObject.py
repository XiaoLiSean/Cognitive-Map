from ai2thor.controller import Controller
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point, Polygon
import shapely.affinity as aff
import matplotlib.pyplot as plt

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
    rot_mat = R.from_euler('xyz',[rot['x'], rot['y'], rot['z']])
    out =np.zeros((4,4))
    out[0:3,0:3] = rot_mat.as_matrix()
    out[0,3] = pos['x']
    out[1,3] = pos['y']
    out[2,3] = pos['z']
    out[3,3] = 1
    return out


def does_fit(object_dims, max_dims):
    # check trivial case

    already_fits = True
    for i in range(0, 3):
        if object_dims[i] > max_dims[i]:
            already_fits = False
    if (already_fits):
        print(f"no axis realign during graph check: {object_dims} vs {max_dims}")
        return True

    # check axis aligned case
    ordered_obj_dims = object_dims.copy()
    ordered_max_dims = max_dims.copy()
    ordered_obj_dims.sort(reverse=True)
    ordered_max_dims.sort(reverse=True)

    if (ordered_obj_dims < ordered_max_dims):
        print(f"axis realign required during graph check: {ordered_obj_dims} vs {ordered_max_dims}")
        return True
    return False

def does_fit_poly(object, max_dims, robot_pos):
    corners = object['objectOrientedBoundingBox']
    object_poly = Polygon(corners)
    constraint_corners = []
    for i in [-1,1]:
        for j in [-1,1]:
            for k in [-1,1]:
                modified = [max_dims[0]*i+robot_pos[0],
                            max_dims[1]*j+robot_pos[1],
                            max_dims[2]*k+robot_pos[2]]
                constraint_corners.append(modified)

    constraint_poly = Polygon(constraint_corners)
    return rotate_to_fit(constraint_corners, corners, robot_pos) is not None

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
    rot_mat = R.from_euler('xyz',[rot['x'], rot['y'], rot['z']])
    out =np.zeros((4,4))
    out[0:3,0:3] = rot_mat.as_matrix()
    out[0,3] = pos['x']
    out[1,3] = pos['y']
    out[2,3] = pos['z']
    out[3,3] = 1
    return out


def does_fit(object_dims, max_dims):
    # check trivial case

    already_fits = True
    for i in range(0, 3):
        if object_dims[i] > max_dims[i]:
            already_fits = False
    if (already_fits):
        print(f"no axis realign during graph check: {object_dims} vs {max_dims}")
        return True

    # check axis aligned case
    ordered_obj_dims = object_dims.copy()
    ordered_max_dims = max_dims.copy()
    ordered_obj_dims.sort(reverse=True)
    ordered_max_dims.sort(reverse=True)

    if (ordered_obj_dims < ordered_max_dims):
        print(f"axis realign required during graph check: {ordered_obj_dims} vs {ordered_max_dims}")
        return True
    return False

def does_fit_poly(object, max_dims, robot_pos):
    corners = object['objectOrientedBoundingBox']
    object_poly = Polygon(corners)
    constraint_corners = []
    for i in [-1,1]:
        for j in [-1,1]:
            for k in [-1,1]:
                modified = [max_dims[0]*i+robot_pos[0],
                            max_dims[1]*j+robot_pos[1],
                            max_dims[2]*k+robot_pos[2]]
                constraint_corners.append(modified)

    constraint_poly = Polygon(constraint_corners)
    return rotate_to_fit(constraint_corners, corners, object['position']) is not None


def fitObject(controller, max_dims):
    event = controller.step("Pass")

    robot_centric = create_transform(event)

    objIdx = None
    for i in range(0, len(event.metadata['objects'])):
        if event.metadata['objects'][i]['isPickedUp']:
            objIdx = i
            break

    bounding_box = event.metadata['objects'][objIdx]['axisAlignedBoundingBox']
    object_dims = [bounding_box['size']['x'], bounding_box['size']['y'],bounding_box['size']['z']]
    object_center_dict = event.metadata['objects'][objIdx]['position']
    object_center = [object_center_dict['x'], object_center_dict['y'], object_center_dict['z']]

    corners = event.metadata['objects'][objIdx]['objectOrientedBoundingBox']['cornerPoints']
    robot_pos = robot_centric[0:3, 3]
    print(f"object corners {corners}")
    object_poly = Polygon(corners)
    constraint_corners = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                modified = [max_dims[0] * i + robot_pos[0],
                            max_dims[1] * j + robot_pos[1],
                            max_dims[2] * k + robot_pos[2]]
                constraint_corners.append(modified)

    constraint_poly = Polygon(constraint_corners)
    print(f"constraint corners {constraint_corners}")

    rot = rotate_to_fit(constraint_poly, object_poly, object_center)
    controller.step("RotateObjectRelative", x=rot[0], y=rot[1], z=rot[2])
    return

def rotate_around(point, center, matrix):
    homogenous = np.ones((4,1))
    homogenous[0:3,0] = point - center
    homogenous_matrix = np.zeros((4,4))
    homogenous_matrix[0:3,0:3] = matrix
    homogenous_matrix[3,3] = 1
    rotated = np.matmul(homogenous_matrix,homogenous)
    return rotated[0:3,0] + center

def is_inside(point, box):
    min_vals = [None, None, None]
    max_vals = [None, None, None]
    for c in box:
        for i in range(0,3):
            if min_vals[i] is None or c[i] < min_vals[i]:
                min_vals[i] = c[i]
            if max_vals[i] is None or c[i] > max_vals[i]:
                max_vals[i]  = c[i]

    for i in range(0,3):
        if point[i] < min_vals[i]:
            return False
        if point[i] > max_vals[i]:
            return False
    return True


if __name__ == "__main__":
    p = np.array([1,0,0])
    c = np.array([0,0,0])
    rot = R.from_euler('xyz',[np.pi,-np.pi/2,0])
    print(f"rotated {p} around {c} by {rot} and got {rotate_around(p,c,rot.as_matrix())}")

    """

    # check trivial case
    already_fits = True
    for i in range(0,3):
        if object_dims[i] > max_dims[i]:
            already_fits = False
    if (already_fits):
        print("No realign:{},{}".format(object_dims,max_dims))
        return True

    # check axis aligned case
    ordered_obj_dims = object_dims.copy()
    ordered_max_dims = max_dims.copy()
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
                if max_dims[i] == ordered_max_dims[j]:
                    constraint_dim_ordering[i] = j

        # check one rotation solutions
        if dim_object_ordering[0] == constraint_dim_ordering[0]:
            controller.step("RotateHandRelative", x=90)
            return True
        elif dim_object_ordering[1] == constraint_dim_ordering[1]:
            controller.step("RotateHandRelative", y=90)
            return True
        elif dim_object_ordering[2] == constraint_dim_ordering[2]:
            controller.step("RotateHandRelative", z=90)
            return True

        print("All offset")

        #if they're all offset, use a two rotation solution
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


        return True

    #since we don't have the generalized checker implemented yet, if we fail to fit in either of these scenarios, then we return false
    return False
    
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
    """






def rotate_to_fit(constraint_box, object_box, object_center):
    rotations = [0, np.pi/2, np.pi, -np.pi/2]
    np_center = np.array(object_center)
    print(f"rotating {object_box} to fit {constraint_box}")
    for x in rotations:
        for y in rotations:
            for z in rotations:
                r = R.from_euler('xyz', [x, y, z])
                mat = r.as_matrix()
                is_outside = False
                for b in object_box:
                    b = np.array(b)
                    rotated = rotate_around(b, np_center, mat)

                    if not is_inside(rotated,constraint_box):
                        is_outside = True
                        if y == 0 and z == 0:
                            print(f"failed on {rotated} by {x} ")

                if not is_outside:
                    return x,y,z

    return None


def fitObject(controller, max_dims):
    event = controller.step("Pass")

    robot_centric = create_transform(event)

    objIdx = None
    for i in range(0, len(event.metadata['objects'])):
        if event.metadata['objects'][i]['isPickedUp']:
            objIdx = i
            break

    bounding_box = event.metadata['objects'][objIdx]['axisAlignedBoundingBox']
    object_dims = [bounding_box['size']['x'], bounding_box['size']['y'],bounding_box['size']['z']]
    object_center_dict = event.metadata['objects'][objIdx]['position']
    object_center = [object_center_dict['x'], object_center_dict['y'], object_center_dict['z']]

    corners = event.metadata['objects'][objIdx]['objectOrientedBoundingBox']['cornerPoints']
    robot_pos = robot_centric[0:3, 3]
    print(f"object corners {corners}")
    object_poly = Polygon(corners)
    constraint_corners = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                modified = [max_dims[0]/2 * i + robot_pos[0],
                            max_dims[1]/2 * j + robot_pos[1],
                            max_dims[2]/2 * k + robot_pos[2]]
                constraint_corners.append(modified)

    constraint_poly = Polygon(constraint_corners)
    print(f"constraint corners {constraint_corners}")

    rot = rotate_to_fit(constraint_corners, corners, object_center)
    controller.step("RotateHandRelative", x=rot[0], y=rot[1], z=rot[2])
    return

def rotate_around(point, center, matrix):
    homogenous = np.ones((4,1))
    homogenous[0:3,0] = point - center
    homogenous_matrix = np.zeros((4,4))
    homogenous_matrix[0:3,0:3] = matrix
    homogenous_matrix[3,3] = 1
    rotated = np.matmul(homogenous_matrix,homogenous)
    return rotated[0:3,0] + center

def is_inside(point, box):
    min_vals = [None, None, None]
    max_vals = [None, None, None]
    for c in box:
        for i in range(0,3):
            if min_vals[i] is None or c[i] < min_vals[i]:
                min_vals[i] = c[i]
            if max_vals[i] is None or c[i] > max_vals[i]:
                max_vals[i]  = c[i]

    for i in range(0,3):
        if point[i] < min_vals[i]:
            return False
        if point[i] > max_vals[i]:
            return False
    return True

test_constraints = np.array([[1.5, 0.525999128818512, 1.25], [1.5, 0.525999128818512, 2.25], [1.5, 1.275999128818512, 1.25], [1.5, 1.275999128818512, 2.25],
                    [2.0, 0.525999128818512, 1.25], [2.0, 0.525999128818512, 2.25], [2.0, 1.275999128818512, 1.25], [2.0, 1.275999128818512, 2.25]])

test_box = np.array([[1.4450517892837524, 0.30431413650512695, 1.1674891710281372], [2.0563912391662598, 0.30431413650512695, 1.1674891710281372], [2.0563912391662598, 0.30431413650512695, 1.5774012804031372], [1.4450517892837524, 0.30431413650512695, 1.5774012804031372],
           [1.4450517892837524, 1.262557864189148, 1.1674890518188477], [2.0563912391662598, 1.262557864189148, 1.1674890518188477], [2.0563912391662598, 1.262557864189148, 1.5774012804031372], [1.4450517892837524, 1.262557864189148, 1.5774012804031372]])

if __name__ == "__main__":
    p = np.array([1,0,0])
    c = np.array([0,0,0])
    for i in range(0,len(test_constraints)):
        test_constraints[i,2] -= 0.5


    object_center = [(1.4450517892837524 + 2.0563912391662598)/2.0,
                     (1.262557864189148 + 0.30431413650512695)/2.0,
                     (1.5774012804031372 + 1.1674890518188477)/2.0]
    rot = R.from_euler('xyz',[np.pi,-np.pi/2,0])
    print(f"rotated {p} around {c} by {rot} and got {rotate_around(p,c,rot.as_matrix())}")
    print(f"how to fit box inside constraints {rotate_to_fit(test_constraints, test_box, object_center)}")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(f"{test_constraints}")
    ax.scatter(test_constraints[:,0], test_constraints[:,1], zs=test_constraints[:,2])
    ax.scatter(test_box[:,0], test_box[:,1], zs=test_box[:,2])
    plt.show()

    """

    # check trivial case
    already_fits = True
    for i in range(0,3):
        if object_dims[i] > max_dims[i]:
            already_fits = False
    if (already_fits):
        print("No realign:{},{}".format(object_dims,max_dims))
        return True

    # check axis aligned case
    ordered_obj_dims = object_dims.copy()
    ordered_max_dims = max_dims.copy()
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
                if max_dims[i] == ordered_max_dims[j]:
                    constraint_dim_ordering[i] = j

        # check one rotation solutions
        if dim_object_ordering[0] == constraint_dim_ordering[0]:
            controller.step("RotateHandRelative", x=90)
            return True
        elif dim_object_ordering[1] == constraint_dim_ordering[1]:
            controller.step("RotateHandRelative", y=90)
            return True
        elif dim_object_ordering[2] == constraint_dim_ordering[2]:
            controller.step("RotateHandRelative", z=90)
            return True

        print("All offset")

        #if they're all offset, use a two rotation solution
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


        return True

    #since we don't have the generalized checker implemented yet, if we fail to fit in either of these scenarios, then we return false
    return False
    
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
    """




