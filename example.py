from lib.scene_graph_generation import *
from lib.ithor_env import *
import random

robot = Agent_Sim('Kitchen', 1)

# initialized SG instance and update from objects data in AI2THOR
SG = Scene_Graph()
objs = [obj for obj in robot.get_object() if obj['visible']]
SG.update_from_data(objs, True)


navigator = Dumb_Navigetor(robot)
reachable_poses = navigator._agent_sim.get_reachable_coordinate()
goal = reachable_poses[random.randint(int(len(reachable_poses) / 3), len(reachable_poses))]
navigator.dumb_navigate(goal)
time.sleep(2)
