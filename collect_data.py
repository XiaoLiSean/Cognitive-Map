import argparse
from termcolor import colored
from lib.robot_env import Agent_Sim
from lib.params import VISBILITY_DISTANCE, SCENE_TYPES, SCENE_NUM_PER_TYPE
from Network.retrieval_network.params import DYNAMICS_ROUNDS, LOCALIZATION_GRID_TOL, TRAIN_FRACTION, VAL_FRACTION, DATA_DIR

# Initialize robot
robot = Agent_Sim()

# ------------------------------------------------------------------------------
# collect train, validation and test data for network image branch
def data_collection(is_train=True, is_test=True):

    # Iterate through all the scenes to collect data and separate them into train, val and test sets
    if is_train:
        grid_steps = LOCALIZATION_GRID_TOL + 1
        for scene_type in SCENE_TYPES:
            for scene_num in range(1, int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1):
                if scene_num <= int(SCENE_NUM_PER_TYPE*TRAIN_FRACTION):
                    FILE_PATH = DATA_DIR + '/train'
                elif scene_num <= int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)):
                    FILE_PATH = DATA_DIR + '/val'

                robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
                robot.coordnates_patroling(saving_data=True, file_path=FILE_PATH, dynamics_rounds=DYNAMICS_ROUNDS, grid_steps=grid_steps)

    if is_test:
        FILE_PATH = DATA_DIR + '/test'
        for scene_type in SCENE_TYPES:
            for scene_num in range(int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1, SCENE_NUM_PER_TYPE + 1):
                robot.reset_scene(scene_type=scene_type, scene_num=scene_num)
                robot.coordnates_patroling(saving_data=True, file_path=FILE_PATH, dynamics_rounds=DYNAMICS_ROUNDS, is_test=True)

# ------------------------------------------------------------------------------
# manually update topological map info
def iter_test_scene(is_xiao=False, is_yidong=False):
    test_idx_initial = int(SCENE_NUM_PER_TYPE*(TRAIN_FRACTION+VAL_FRACTION)) + 1
    test_idx_end = SCENE_NUM_PER_TYPE + 1
    add_on = [2, 3, 2, 3]
    for idx, scene_type in enumerate(SCENE_TYPES):
        if is_xiao:
            test_idx_end = test_idx_initial + add_on[idx]
        if is_yidong:
            test_idx_initial = test_idx_end - ( 5 - add_on[idx])

        for scene_num in range(test_idx_initial, test_idx_end):
            robot.reset_scene(scene_type=scene_type, scene_num=scene_num, ToggleMapView=True, Show_doorway=True)
            robot.show_map(show_nodes=True, show_edges=True)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Get argument from CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", help="manually update topological map info", action="store_true")
    parser.add_argument("--train", help="collect train and validation data for network image branch", action="store_true")
    parser.add_argument("--test", help="collect test data for network image branch", action="store_true")
    parser.add_argument("--yidong", help="manually collect topological map node for yidong", action="store_true")
    parser.add_argument("--xiao", help="manually collect topological map node for xiao", action="store_true")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Used to collect data
    if args.train or args.test:
        data_collection(is_train=args.train, is_test=args.test)

    # Used to see visualization of each test scene and record the topological node manually
    if args.topo:
        iter_test_scene(is_xiao=args.xiao, is_yidong=args.yidong)
