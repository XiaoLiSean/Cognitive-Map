import matplotlib.pyplot as plt
import numpy as np

def fit_to_constraints(obj_vertices, robot_info, env_constraint, other_constraints):
    """
    determine what rotations/translations necessary to fit the object into these constraints
    @param obj_vertices: the vertices of the object in the workspace
    @param robot_info: dict representing the center of the robot in the x,y,z as well as the orientation of the robot,
    and the size of the robot in x-z
    @param env_constraint: directional constraint dimensions
    @param other_constraints: other directional constraint dimensions to be integrated together
    @return: rotations around object's center and translations to put the object inside the constraints
    """

    final_constraint_vertices = combine_constraints(robot_info, env_constraint, other_constraints)
    rotations,translation = fit_object(obj_vertices, final_constraint_vertices,robot_info)

def fit_object(object_vertices, constraint_vertices, robot_info):
    """
    deteermine what rotations/translations are necessary to fit the object into these constraints
    @param object_vertices: vertices representing the corners of the object
    @param constraint_vertices: vertices representing the corners of the constraining box
    @return: rotations and translations necessary to fit the object in the constraints
    """

    #Method: using the robot's orientation, we will fix the face of the object closest to the robot
    # to the face of the constraint also closest to the robot
    # from here, we then try different orientation (reaffixing the faces when necessary),
    # and if we see exclusions in the x/y axes, then we correct for them with translations
    # if at any point, we find a fit, we ship it


def combine_constraints(robot_info, env_constraints, other_constraints):
    """
    orient each constraint so we get a set of vertices relative to the
    @param robot_info: dict representing the center of the robot in the x,y,z as well as the orientation of the robot,
    and the size of the robot in x-z
    @param env_constraints: directional constraint dimensions (North, South, East, West, Up Down)
    @param other_constraints: other directional constraint dimensions to be integrated together
    @return: vertices of the constraint box after the other constraints are integrated together
    """


    dir_constraint = create_dir_constraints(robot_info)
    final_constraint = integrate_constraint(env_constraints, dir_constraint)
    for c in other_constraints:
        final_constraint = integrate_constraint(final_constraint, c)

    return orient_constraint(robot_info, final_constraint)


def orient_constraint(robot_info, constraint):
    """
    generate a list of vertices from the constraint relative to the robot center
    @param robot_info: dict representing robot center, orientation of the robot, and size of robot in x-z
    @param constraint: directional constraint dimensions (North, South, East, West, Up Down)
    @return: a list of vertices representing the constraint
    """

    verts = []
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):

                NS = (1-2*i)*constraint[i]
                EW = (1 - 2 * j) * constraint[2+j]
                UD = (1-2*k)*constraint[4+k]
                verts.append([EW+robot_info['x'], UD+robot_info['y'], NS+robot_info['z']])

    return verts

def integrate_constraint(main_constraint, additional_constraint):
   """
   take the intersection of these constraints
   @param main_constraint: directional constraint dimensions (no Nones)
   @param additional_constraint: constraint to integrate into the main concstraint
   Both must be lists of length 6, main_Constraint can have no "None" values
   @return: the intersection of these constraints
   """

   for i in range(0,len(main_constraint)):
        if additional_constraint[i] is not None:
            main_constraint[i] = min(main_constraint[i],additional_constraint[i])

   return main_constraint

def create_dir_constraints(robot_info):
    """
    create a constraint so we are holding the object in front of the robot
    @param robot_info: dict of robot information
    @return: a partial dimensional constraint (some values are None)
    """

    constraint = [None]*6
    robot_dist = robot_info['depth'] / 2
    if robot_info['orientation'] == 0:
        constraint[1] = -robot_dist
    elif robot_info['orientation'] == 90:
        constraint[3] = -robot_dist
    elif robot_info['orientation'] == 180:
        constraint[0] = -robot_dist
    elif robot_info['orientation'] == 270:
        constraint[2] = -robot_dist
    return constraint

if __name__ == "__main__":
    robot_info = {'x': 1,
                  'y': 1,
                  'z': 0,
                  'orientation': 90,
                  'depth': 0.5}
    env_constraint = [1]*6
    oriented = orient_constraint(robot_info, env_constraint)
    print(oriented)
    combined = combine_constraints(robot_info, env_constraint, [])
    print(combined)
