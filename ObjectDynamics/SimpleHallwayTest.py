from ai2thor.controller import Controller
import cv2
from ObjectDynamics import FitObject as fo

picNum=0

def saveImg(event, picNum):
    #print("saving pic {}".format(picNum))
    frame = event.frame
    cv2.imwrite("TestPics/FloorPlan604.png".format(picNum),cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

exec_path = "/Users/Jake/MastersFirstYear/PROGRESS_LAB/ai2thor/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-Thor"
controller = Controller(scene='FloorPlan604', gridSize=0.25,
                        local_executable_path=exec_path)
event = controller.step({"action":"ToggleMapView"})

saveImg(event, 0)

assert False

event = controller.step("AddThirdPartyCamera",
                rotation=dict(x=45,y=180,z=0),
                position=dict(x=1.75, y=2.9, z=2.2)
                )



event = controller.step("LookDown")
print("agent pos: {}, rot: {}".format(event.metadata['agent']['position'], event.metadata['agent']['rotation']))
saveImg(event, picNum)
picNum += 1
event = controller.step("MoveRight")
saveImg(event, picNum)
picNum += 1
event = controller.step("MoveRight")
saveImg(event, picNum)
picNum += 1

stool_obect_id = None
count = 0
for o in event.metadata['objects']:
     count += 1
     if o['visible'] and o['pickupable'] and o['objectType'] == 'Stool':
         event = controller.step(action="PickupObject", objectId=o['objectId'], raise_for_failure=True)
         saveImg(event, picNum)
         picNum += 1
         stool_object_id = o['objectId']

         break
event=controller.step("RotateLeft")
saveImg(event, picNum)
picNum += 1
event=controller.step("RotateHandRelative", z=-90, raise_for_failure=True)

saveImg(event, picNum)
picNum += 1
while(event.metadata['lastActionSuccess']):
    event = controller.step("MoveAhead",manualInteract=True)
    saveImg(event, picNum)
    picNum += 1

event = controller.step("MoveBack", manualInteract=True)
saveImg(event, picNum)
picNum += 1
print("chair size: {}".format(event.metadata['objects'][count-1]['axisAlignedBoundingBox']['size']))
fo.fitObject(controller, [0.4,1,0.6])
event = controller.step("Pass", manualInteract=True)
print("chair size after rotate: {}, pic num: {}".format(event.metadata['objects'][count-1]['axisAlignedBoundingBox']['size'], picNum))
#event=controller.step("RotateHandRelative", z=90, raise_for_failure=True)
saveImg(event, picNum)
picNum += 1

while(event.metadata['lastActionSuccess']):
    event = controller.step("MoveAhead",manualInteract=True)
    saveImg(event, picNum)
    picNum += 1

#event=controller.step("MoveAhead",manualInteract=True)
#input("Press Enter to continue...")

#event=controller.step("RotateHandRelative",z=-90)
#event=controller.step("MoveAhead",manualInteract=True,moveMagnitude=1)





