from ai2thor.controller import Controller
from termcolor import colored
from lib.params import *
import numpy as np
import time, copy, random

# ------------------------------------------------------------------------------
# Notes: For Attributes which can or cannot be use in iTHOR and RoboTHOR
#        to move small objects (except table, chair and etc...)
# ------------------------------------------------------------------------------
# Attributes                              |   iTHOR       |   RoboTHOR
# ------------------------------------------------------------------------------
# InitialRandomSpawn                      |   Work        |   Not work
# GetSpawnCoordinatesAboveReceptacle      |   Work        |   Work
# SetObjectPoses                          |   Work        |   Work
# PlaceObjectAtPoint                      |   Work        |   Not work
# ------------------------------------------------------------------------------

def get_surf(obj):
    size = obj['axisAlignedBoundingBox']['size']
    return size['x']*size['z']

def get_volumn(obj):
    size = obj['axisAlignedBoundingBox']['size']
    return size['x']*size['y']*size['z']

# ------------------------------------------------------------------------------
# This function return all possible parentReceptacles for a certain
# furniture which has smaller surface area than its pR. The return list
# is sorted by the parentReceptacles' surface area in descend order
def get_possible_parentReceptacles(furniture, objs, floor_ID):
    surf_furniture = get_surf(furniture)
    possible_pRs = []
    objs.sort(key=get_surf, reverse=True)
    for obj in objs:
        if obj['objectId'] == floor_ID or obj['objectType'] == furniture['objectType']:
            continue
        surf_obj = get_surf(obj)
        if surf_obj > surf_furniture:
            possible_pRs.append(obj['objectId'])
    return possible_pRs

# ------------------------------------------------------------------------------
# This function is used to shuffle objects in scenes
# Input the controller handle and output the event data
def shuffle_scene_layout(controller, num_attempts=40, floor_obstacle_avoidance=False, verbose=False):

    event = controller.step("Pass")
    # Disable small objects
    small_objs = copy.deepcopy([obj for obj in event.metadata['objects'] if obj['pickupable']])
    for obj in small_objs:
        controller.step('DisableObject', objectId=obj['objectId'])
    # --------------------------------------------------------------------------
    # Prepare furnitures data
    furnitures = copy.deepcopy({obj['name']:obj for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']})
    furnitures_sorted = copy.deepcopy([{'name': obj['name'], 'axisAlignedBoundingBox': obj['axisAlignedBoundingBox']}
                                      for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']])
    furnitures_sorted.sort(key=get_volumn, reverse=True) # Sort object by its volume from large to small
    successfully_placed = copy.deepcopy({obj['name']:False for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']}) # Used to store info if i_th furniture has been successfully placed in

    # --------------------------------------------------------------------------
    # Try to place the object on the floor
    floor = [[obj for obj in event.metadata['objects'] if obj['objectType'] == 'Floor']]
    floor = floor[0][0]
    event = controller.step('GetSpawnCoordinatesAboveReceptacle', objectId=floor['objectId'], anywhere=True)
    floor_poses = event.metadata['actionReturn']
    random.shuffle(floor_poses)

    for obj_sorted in furnitures_sorted:
        obj = furnitures[obj_sorted['name']] # Get object information using 'name' key out of furnitures dict
        # This move this object is not reasonable e.g. Microwave in FloorPlan1
        # And no need to move objects which are already on the floor
        if obj['parentReceptacles'] is None:
            successfully_placed[obj_sorted['name']] = True
            continue
        if obj['parentReceptacles'][0] == floor['objectId']:
            continue
        # Try to place the object on the floor
        if num_attempts is not None and num_attempts < len(floor_poses):
            floor_poses_tmp = copy.deepcopy(floor_poses[0:num_attempts])
        else:
            floor_poses_tmp = copy.deepcopy(floor_poses)

        for position in floor_poses_tmp:
            event = controller.step(action='PlaceObjectAtPoint', objectId=obj['objectId'], position=position)
            if event.metadata['lastActionSuccess']:
                floor_poses.remove(position)
                if verbose:
                    print(colored('Config Info: ','blue')+'Placed '+obj['objectId']+' on floor')
                break

    # --------------------------------------------------------------------------
    # Try to place the object back with Gaussian Noise back on parentReceptacles
    event = controller.step(action='Pass')
    furnitures_prev = copy.deepcopy(furnitures)
    # Prepare furnitures data
    furnitures = copy.deepcopy({obj['name']:obj for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']})
    furnitures_sorted = copy.deepcopy([{'name': obj['name'], 'axisAlignedBoundingBox': obj['axisAlignedBoundingBox']}
                                      for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']])
    furnitures_sorted.sort(key=get_volumn, reverse=True) # Sort object by its volume from large to small

    for obj_sorted in furnitures_sorted:
        obj = furnitures[obj_sorted['name']] # Get object information using 'name' key out of furnitures dict
        # This move this object is not reasonable e.g. Microwave in FloorPlan1
        if obj['parentReceptacles'] is None or furnitures_prev[obj_sorted['name']]['parentReceptacles'] is None:
            continue
        # Get spawn coordinates on original parentReceptacles
        event = controller.step('GetSpawnCoordinatesAboveReceptacle',
                                objectId=furnitures_prev[obj_sorted['name']]['parentReceptacles'][0], anywhere=True)
        spawn_poses = event.metadata['actionReturn']
        if spawn_poses == None or len(spawn_poses) == 0:
            print(furnitures_prev[obj_sorted['name']]['parentReceptacles'], ' has no spawn position for obj:')
            print(furnitures_prev[obj_sorted['name']])
            continue

        random.shuffle(spawn_poses) # Randomize the positions order
        # High dynamics object move randomly on its parentReceptacles
        if obj['objectType'] in HIGH_DYNAMICS:
            pass

        # Case there is no obstical avoidance code, object previously on the floor will not moved
        if not floor_obstacle_avoidance and furnitures_prev[obj_sorted['name']]['parentReceptacles'][0] == floor['objectId']:
            successfully_placed[obj_sorted['name']] = True
            continue

        # Low dynamics object can only move on its parentReceptacles within certain range
        if obj['objectType'] in LOW_DYNAMICS:
            pose_prev = furnitures_prev[obj_sorted['name']]['position']
            spawn_poses_tmp = copy.deepcopy(spawn_poses)
            for pose in spawn_poses_tmp:
                x_diff = abs(pose['x']-pose_prev['x'])
                z_diff = abs(pose['z']-pose_prev['z'])
                size = obj['axisAlignedBoundingBox']['size']
                distance = np.sqrt(x_diff**2 + z_diff**2)
                threshold = np.sqrt(size['x']**2 + size['z']**2)*LOW_DYNAMICS_MOVING_RATIO
                if distance > threshold:
                    spawn_poses.remove(pose)

        # Exception for no spawn_pose
        if len(spawn_poses) != 0:
            # Try to place object on possible pose on parentReceptacles
            if num_attempts is not None and num_attempts < len(spawn_poses):
                spawn_poses_tmp = copy.deepcopy(spawn_poses[0:num_attempts])
            else:
                spawn_poses_tmp = copy.deepcopy(spawn_poses)
            for position in spawn_poses_tmp:
                # Add Gaussian oise to rotation date
                rotation = obj['rotation']
                sigma_angle = (MASS_MAX - obj['mass'])/(MASS_MAX - MASS_MIN)*ROTATE_MAX_DEG
                rotation['y'] += np.clip(np.random.normal(0, sigma_angle), a_min = -3*sigma_angle, a_max = 3*sigma_angle)
                event = controller.step(action='PlaceObjectAtPoint', objectId=obj['objectId'], position=position, rotation=rotation)
                if event.metadata['lastActionSuccess']:
                    successfully_placed[obj_sorted['name']] = True
                    if verbose:
                        print(colored('Config Info: ','blue')+'Change Pos of '+obj['objectId'])
                    break

        # At the end: if the object which is previous on their pRs and did not placed to new pose
        # Ignore it and set the task done
        if furnitures_prev[obj_sorted['name']]['parentReceptacles'][0] == obj['parentReceptacles'][0]:
            successfully_placed[obj_sorted['name']] = True

    # --------------------------------------------------------------------------
    # Handling Exceptions
    event = controller.step(action='Pass')
    # Prepare furnitures data
    furnitures = copy.deepcopy({obj['name']:obj for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']})
    furnitures_sorted = copy.deepcopy([{'name': obj['name'], 'axisAlignedBoundingBox': obj['axisAlignedBoundingBox']}
                                      for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']])
    furnitures_sorted.sort(key=get_volumn, reverse=True) # Sort object by its volume from large to small

    for obj_sorted in furnitures_sorted:
        obj = furnitures[obj_sorted['name']] # Get object information using 'name' key out of furnitures dict
        if not successfully_placed[obj_sorted['name']]:
            # Place the object to initialized place and position
            position = furnitures_prev[obj_sorted['name']]['position']
            event = controller.step(action='PlaceObjectAtPoint', objectId=obj['objectId'], position=position)
            if event.metadata['lastActionSuccess']:
                successfully_placed[obj_sorted['name']] = True
            else:
                # Try to place on other parentReceptacles
                possible_pRs = get_possible_parentReceptacles(obj, event.metadata['objects'], floor['objectId'])
                for pR_ID in possible_pRs:
                    event = controller.step('GetSpawnCoordinatesAboveReceptacle', objectId=pR_ID, anywhere=True)
                    spawn_poses = event.metadata['actionReturn']
                    if spawn_poses == None:
                        continue
                    random.shuffle(spawn_poses) # Randomize the positions order
                    if num_attempts is not None and num_attempts < len(spawn_poses):
                        spawn_poses_tmp = copy.deepcopy(spawn_poses[0:num_attempts])
                    else:
                        spawn_poses_tmp = copy.deepcopy(spawn_poses)
                    for position in spawn_poses_tmp:
                        event = controller.step(action='PlaceObjectAtPoint', objectId=obj['objectId'], position=position)
                        if event.metadata['lastActionSuccess']:
                            successfully_placed[obj_sorted['name']] = True
                            if verbose:
                                print(colored('Config Info: ','blue')+'Change Pos of '+obj['objectId'])
                            break
                    if successfully_placed[obj_sorted['name']]:
                        break

    furnitures = copy.deepcopy([obj for obj in event.metadata['objects'] if not obj['pickupable'] and obj['moveable']])

    # Enable small objects
    for obj in small_objs:
        controller.step('EnableObject', objectId=obj['objectId'])
    # Randomize position of small objects on previous receptacles/furniture
    controller.step(action='InitialRandomSpawn', randomSeed=random.randint(1, 100), forceVisible=False, numPlacementAttempts=num_attempts, placeStationary=True)
    event = controller.step("Pass")

    return event
