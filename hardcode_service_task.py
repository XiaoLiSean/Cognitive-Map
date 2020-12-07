from ai2thor.controller import Controller
import time
controller = Controller(scene='FloorPlan28', gridSize=0.25)

# change starting locations
controller.step(action='Teleport', x=-2.5, y=0.900998235, z=-3.0)
time.sleep(1)
event = controller.step(action='Rotate', rotation=270)
time.sleep(2)
controller.step(action='LookDown', degrees=30)
time.sleep(2)

# in FloorPlan28, the agent should now be looking at a mug
for o in event.metadata['objects']:
    if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
        # pick up the mug
        event = controller.step(action='PickupObject',
                                objectId=o['objectId'],
                                raise_for_failure=True)
        time.sleep(1)
        mug_object_id = o['objectId']
        break

controller.step('HideObject', objectId=mug_object_id)
# the agent now has the Mug in its inventory
# to put it into the Microwave, we need to open the microwave first
time.sleep(1)
# move to the microwave
event = controller.step(action='LookUp', degrees=50)
time.sleep(1)
controller.step(action='Teleport', x=-1.0, y=0.900998235, z=-2.0)
event = controller.step(action='Rotate', rotation=90)
time.sleep(2)

# the agent should now be looking at the microwave
for o in event.metadata['objects']:
    if o['visible'] and o['openable'] and o['objectType'] == 'Microwave':
        # open the microwave
        event = controller.step(action='OpenObject',
                                objectId=o['objectId'],
                                raise_for_failure=True)
        time.sleep(1)
        receptacle_object_id = o['objectId']
        break

controller.step('UnhideObject', objectId=mug_object_id)
time.sleep(1)
# put the object in the microwave
event = controller.step(
    action='PutObject',
    receptacleObjectId=receptacle_object_id,
    objectId=mug_object_id,
    raise_for_failure=True)

time.sleep(1)
# close the microwave
event = controller.step(
    action='CloseObject',
    objectId=receptacle_object_id,
    raise_for_failure=True)
time.sleep(1)
