from ai2thor.controller import Controller
from termcolor import colored
from lib.params import *
from lib.object_dynamics import *
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


if __name__ == '__main__':
    # controller = Controller(scene='FloorPlan1', gridSize=0.25)
    # controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)
    # controller.step('Pass')
    # event = controller.step(action='RotateLeft', degrees=90.0)
    # # Id = 'CoffeeMachine|-01.98|+00.90|-00.19'
    # # Id = 'Bread|-00.52|+01.17|-00.03'
    # Id = [obj['objectId'] for obj in event.metadata['objects'] if obj['objectType'] == 'Bread'][0]
    # # counterTopId = 'CounterTop|-00.08|+01.15|00.00'
    # objs = [obj for obj in event.metadata['objects'] if obj['objectId'] == Id]
    # spawn_pose = objs[0]['position']
    # time.sleep(2)
    # for i in range(10):
    #     Id = [obj['objectId'] for obj in event.metadata['objects'] if obj['objectType'] == 'Bread'][0]
    #     spawn_pose['x'] +=0.05
    #     if i == 0:
    #         object_poses = [{"objectName": objs[0]['name'], "rotation": objs[0]['rotation'], "position": spawn_pose}]
    #         event = controller.step(action='SetObjectPoses', objectPoses=object_poses)
    #         print(event.metadata['actionReturn'], event.metadata['lastActionSuccess'], event.metadata['errorMessage'])
    #         time.sleep(2)
    #     else:
    #         print('PlaceObjectAtPoint')
    #         event = controller.step(action = 'PlaceObjectAtPoint', objectId=Id, position=spawn_pose) # won't stuck for 2.4.15
    #         print(event.metadata['actionReturn'], event.metadata['lastActionSuccess'], event.metadata['errorMessage'])
    #         time.sleep(2)
    #     time.sleep(1)
    #     print(spawn_pose)
    # # event = controller.step(action = 'PlaceObjectAtPoint', objectId=breadId, position=spawn_pose) # stuck
    # time.sleep(5)

    # --------------------------------------------------------------------------
    # Test for move large objects
    # --------------------------------------------------------------------------
    # controller = Controller(scene='FloorPlan_Train1_1', gridSize=0.25)
    controller = Controller(scene='FloorPlan1', gridSize=0.25)
    controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)
    event = controller.step(action='RotateLeft', degrees=90.0)
    time.sleep(5)
    event = shuffle_scene_layout(controller)
    time.sleep(5)

    # --------------------------------------------------------------------------
    # Test for move small objects
    # --------------------------------------------------------------------------
    # #controller = Controller(scene='FloorPlan_Train1_1', gridSize=0.25)
    # controller = Controller(scene='FloorPlan1', gridSize=0.25)
    # controller.step('ChangeResolution', x=SIM_WINDOW_WIDTH, y=SIM_WINDOW_HEIGHT)
    # event = controller.step('Pass')
    # #controller.step(action='RotateLeft', degrees=90.0)
    # # time.sleep(10)
    # # for i in range(1000):
    # #     print(i)
    # #     event = controller.step(action='InitialRandomSpawn', randomSeed=i, forceVisible=False, numPlacementAttempts=5, placeStationary=True)
    # # event = controller.step(action='InitialRandomSpawn', randomSeed=1, forceVisible=False, numPlacementAttempts=5, placeStationary=True)
    #
    # name = 'Bread'
    # receptacleId = ''
    # objId = ''
    # object_poses = []
    # # name = 'Mug'
    # for obj in event.metadata['objects']:
    #     object_poses.append({"objectName": obj['name'], "rotation": obj['rotation'], "position": obj['position']})
    #     if obj['objectType'] == name:
    #         print(obj['pickupable'], obj['moveable'])
    #         objId = obj['objectId']
    #         print(objId)
    #         receptacleId = obj['parentReceptacles'][0]
    #         print(receptacleId,objId)
    #         event = controller.step('GetSpawnCoordinatesAboveReceptacle', objectId=receptacleId, anywhere=True) # Can return None?
    #         print('Got pose!!!')
    #         spawn_pose = event.metadata['actionReturn'][10]
    #         print('Before: ', object_poses[-1])
    #         print('Current pose: ', obj['position'])
    #         print('Desired pose: ', spawn_pose)
    #         object_poses[-1]['position'] = spawn_pose
    #         print('After: ', object_poses[-1])
    #
    # time.sleep(5)
    # print('Prepare for PlaceObjectAtPoint: ', objId, spawn_pose)
    # #event = controller.step(action = 'PlaceObjectAtPoint', objectId=objId, position=spawn_pose) # stuck
    # event = controller.step(action='SetObjectPoses', objectPoses=object_poses) # not stuck even if did not place
    # print('Changed Pose')
    # print([obj['objectId'] for obj in event.metadata['objects'] if obj['objectType'] == name])
    # # controller.step(action='RotateLeft', degrees=90.0)
    # time.sleep(10)
