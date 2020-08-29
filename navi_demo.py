from lib.ithor_env import *
from lib.scene_graph_generation import *
import random

if __name__ == '__main__':
    robot = Agent_Sim('Kitchen', 1)
    navigator = Dumb_Navigetor(robot)
    reachable_poses = navigator._agent_sim.get_reachable_coordinate()
    goal = reachable_poses[random.randint(int(len(reachable_poses) / 3), len(reachable_poses))]
    navigator.dumb_navigate(goal)
