from lib.robot_patrol import *
from lib.scene_graph_generation import *
from termcolor import colored


scene_types = ['Kitchen', 'Living room', 'Bedroom', 'Bathroom']
train_fraction = 0.7
val_fraction = 0.15
test_fraction = 0.15
total_scene_num = 30
node_radius = 1.5
grid_steps = 2
robot = Agent_Sim(scene_type='Kitchen', scene_num=1, node_radius=node_radius)

for scene_type in scene_types:
    for scene_num in range(1,total_scene_num+1):
        if scene_num <= int(total_scene_num*train_fraction):
            FILE_PATH='./Network/retrieval_network/image_data/train'
        elif scene_num <= int(total_scene_num*(train_fraction+val_fraction)):
            FILE_PATH='./Network/retrieval_network/image_data/val'
        else:
            FILE_PATH='./Network/retrieval_network/image_data/test'

        robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
        robot.coordnates_patroling(saving_data=True, file_path=FILE_PATH, grid_steps=grid_steps)
