from ai2thor.controller import Controller
from scene_graph_generation import *

controller = Controller(scene='FloorPlan_Train1_1', gridSize=0.25, agentMode='bot')
event = controller.step(action='MoveBack', actionMagnitude=0.25 * 2)

# initialized SG instance and update from objects data in AI2THOR
SG = Scene_Graph()
objs = [obj for obj in event.metadata['objects'] if obj['visible']]
SG.update_from_data(objs, True)
