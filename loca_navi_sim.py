import argparse, copy
import multiprocessing, time
from lib.navigation import Navigation
from Map.map_plotter import Plotter
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument("--scene_type", type=int, default=1,  help="Choose scene type for simulation, 1 for Kitchens, 2 for Living rooms, 3 for Bedrooms, 4 for Bathrooms")
parser.add_argument("--scene_num", type=int, default=26,  help="Choose scene num for simulation, from 1 - 30")
parser.add_argument("--grid_size", type=float, default=0.25,  help="Grid size of AI2THOR simulation")
parser.add_argument("--rotation_step", type=float, default=10,  help="Rotation step of AI2THOR simulation")
parser.add_argument("--sleep_time", type=float, default=0.005,  help="Sleep time between two actions")
parser.add_argument("--save_directory", type=str, default='./data',  help="Data saving directory")
parser.add_argument("--overwrite_data", type=lambda x: bool(strtobool(x)), default=False, help="overwrite the existing data or not")
parser.add_argument("--log_level", type=int, default=2,  help="Level of showing log 1-5 where 5 is most detailed")
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False,  help="Output debug info if True")
parser.add_argument("--test_data", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting test dataset")
parser.add_argument("--special", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting special long range dataset")
parser.add_argument("--AI2THOR", type=lambda x: bool(strtobool(x)), default=False, help="True for RobotTHOR false for ITHOR")
args = parser.parse_args()


if __name__ == '__main__':
	navigation = Navigation(scene_type=args.scene_type, scene_num=args.scene_num, save_directory=args.save_directory, AI2THOR=args.AI2THOR)
	navigation.Update_node_generator()
	navigation.Update_topo_map_env()
	navigation.Update_planner_env()
	navigation.node_generator.Shuffle_scene()
	navigation.nav_test_simplified()