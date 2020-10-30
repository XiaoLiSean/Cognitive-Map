from lib.robot_patrol import *
from lib.scene_graph_generation import *
from termcolor import colored
from lib.params import VISBILITY_DISTANCE

scene_types = ['Kitchen', 'Living room', 'Bedroom', 'Bathroom']
train_fraction = 0.7
val_fraction = 0.15
test_fraction = 0.15
total_scene_num = 30
robot = Agent_Sim(scene_type='Kitchen', scene_num=1, node_radius=VISBILITY_DISTANCE)

def data_collection():
    grid_steps = 2
    # Iterate through all the scenes to collect data and separate them into train, val and test sets
    for scene_type in scene_types:
        for scene_num in range(1,total_scene_num+1):
            if scene_num <= int(total_scene_num*train_fraction):
                FILE_PATH = './Network/retrieval_network/datasets/train'
            elif scene_num <= int(total_scene_num*(train_fraction+val_fraction)):
                FILE_PATH = './Network/retrieval_network/datasets/val'
            else:
                FILE_PATH = './Network/retrieval_network/datasets/test'

            robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
            robot.coordnates_patroling(saving_data=True, file_path=FILE_PATH, dynamics_rounds=10, grid_steps=grid_steps)

def iter_test_scene():
    test_idx_initial = int(total_scene_num*(train_fraction+val_fraction)) + 1
    for scene_type in scene_types:
        for scene_num in range(test_idx_initial, total_scene_num + 1):
            robot.reset_scene(scene_type=scene_type, scene_num=scene_num, ToggleMapView=True)
            robot.show_map(show_nodes=True)

if __name__ == '__main__':
    # Uncomment to collect data
    # data_collection()
    # Uncomment to see visualization of each test scene and record the topological node manually
    iter_test_scene()
