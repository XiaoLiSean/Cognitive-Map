from lib.ithor_env import *
from lib.scene_graph_generation import *
from termcolor import colored
import multiprocessing, random

def server_nav(server, comfirmed):
    # prepare date for navigator process
    robot = Agent_Sim(scene_type='Kitchen', scene_num=1)
    navigator = Dumb_Navigetor(robot)
    reachable_poses = navigator._agent_sim.get_reachable_coordinate()
    goal = reachable_poses[random.randint(int(len(reachable_poses) / 3), len(reachable_poses))]
    navigator.dumb_navigate(goal, server=server, comfirmed=comfirmed)

def client_SG(client, comfirmed):
    SG = Scene_Graph()
    while True:
        data = client.recv()
        if data == 'END':
            break
        SG.update_from_data(data, visualization_on=True, comfirmed=comfirmed)
        SG.reset()

    print(colored('Client: ','green') + 'END')

if __name__ == '__main__':

    comfirmed = multiprocessing.Value('i')  # Int value: 1 for confirm complete task and other process can go on while 0 otherwise
    comfirmed.value = 0
    server, client = multiprocessing.Pipe()  # server send date and client receive data

    # Create navigator node/process
    navigator_node = multiprocessing.Process(target=server_nav, args=(server, comfirmed))

    # Create sg node to visualize scene graph
    sg_node = multiprocessing.Process(target=client_SG, args=(client, comfirmed))

    # Start processes
    navigator_node.start()
    sg_node.start()
    navigator_node.join()
    sg_node.join()
