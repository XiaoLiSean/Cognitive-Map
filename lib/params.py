# Module for setup params and global variables
from os.path import dirname, abspath
import numpy as np

# ------------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 300
SIM_WINDOW_HEIGHT = DEFAULT_IMG_SIZE
SIM_WINDOW_WIDTH = DEFAULT_IMG_SIZE
BAN_TYPE_LIST = ['Floor']   # Ignore non-informative objectType e.g. 'Floor
SCENE_TYPES = ['Kitchen', 'Living room', 'Bedroom', 'Bathroom']
SCENE_NUM_PER_TYPE = 30
# ------------------------------------------------------------------------------
INFO_FILE_PATH = dirname(dirname(abspath(__file__))) + '/AI2THOR_info' # File path for info of iTHOR Env.
obj_2_idx_dic = np.load(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', allow_pickle='TRUE').item()
idx_2_obj_list = np.load(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy')
OBJ_TYPE_NUM = len(idx_2_obj_list) # Maximum numbers of objectType in iTHOR Env.
# ------------------------------------------------------------------------------
ITHOR_FLOOR_PLANS = np.load(INFO_FILE_PATH + '/' + 'iTHOR_FloorPlan.npy') # list of floorplan name for iTHOR
ROBOTHOR_FLOOR_PLANS = np.load(INFO_FILE_PATH + '/' + 'RoboTHOR_FloorPlan.npy') # list of floorplan name for RoboTHOR
# ------------------------------------------------------------------------------
THIRD_PARTY_PATH = dirname(dirname(abspath(__file__))) + '/3rdparty' # File path for 3rdparty info
GLOVE_FILE_NAME = 'glove.42B.300d.txt' # file name of glove vector
glove_embedding = np.load(THIRD_PARTY_PATH + '/' + 'glove_embedding.npy', allow_pickle='TRUE').item() # all Glove word Embedding vectors
THOR_2_VEC = np.load(THIRD_PARTY_PATH + '/' + 'THOR_2_VEC.npy', allow_pickle='TRUE').item() # objectType embedding using GLoVe vectors
# ------------------------------------------------------------------------------
CLUSTERING_RADIUS_RATIO = 1.0 # used to cluster drawers and cabinets, increase to allow larger tolerance
VISBILITY_DISTANCE = 1.5 # default 1.5 meter, object within the radius of a cylinder centered about the y-axis of the agent is visible
FIELD_OF_VIEW = 120 # default 90 degree, 120 degree is binocular FoV
SUB_NODES_NUM = 4
# ------------------------------------------------------------------------------
# This is after group up
# ------------------------------------------------------------------------------
'''
Used to group up massive numbers of receptacles or identical object in SG module
Objects in these objectTypes are typically presented in array or regular spatial order.
'''

GROUP_UP_LIST = ['Drawer', 'Cabinet', 'Shelf', 'StoveBurner', 'StoveKnob', 'Window', 'Book',
                 'Fork', 'Plate', 'Vase', 'ButterKnife', 'Statue', 'Faucet',
                 'Ladle', 'Pencil', 'Blinds', 'Mirror', 'Painting', 'Pillow', 'Newspaper',
                 'Candle', 'RoomDecor', 'CD', 'TennisRacket', 'Poster', 'Cloth', 'TowelHolder',
                 'ToiletPaper', 'HandTowelHolder']
# ------------------------------------------------------------------------------
# Maximum numbers of receptacles appeared in one scene after group_up()
# ------------------------------------------------------------------------------
'''
2021 Feb 02: Modify the REC_MAX_DIC from receptacle to general objects
since visual information is utilized in the Network implementation.
Cases of multiple general objects in FoV (e.g. two statues) need to be considered
'''
REC_MAX_DIC = {'CounterTop': 4, 'Sink': 2, 'HousePlant': 2, 'Pot': 2, 'Cup': 3, 'SinkBasin': 2, 'ShelvingUnit': 2,
               'Stool': 2, 'Drawer': 6, 'Cabinet': 8, 'Shelf': 6, 'StoveBurner': 2, 'StoveKnob': 4, 'Window': 4,
               'Book': 3, 'Fork': 6, 'Plate': 6, 'Vase': 7, 'ButterKnife': 5, 'Statue': 7, 'Faucet': 3, 'CellPhone': 2,
               'Chair': 8, 'Ladle': 2, 'SideTable': 8, 'DiningTable': 3, 'Curtains': 3, 'Pencil': 3, 'Blinds': 4,
               'AluminumFoil': 3, 'Box': 4, 'FloorLamp': 3, 'DeskLamp': 2, 'ArmChair': 4, 'CoffeeTable': 4, 'TVStand': 2,
               'Sofa': 2, 'Painting': 4, 'Pillow': 2, 'Newspaper': 2, 'Desk': 5, 'Dresser': 4, 'Candle': 3, 'RoomDecor': 5,
               'Bed': 2, 'BaseballBat': 2, 'CD': 3, 'TennisRacket': 2, 'Dumbbell': 2, 'Poster': 2, 'Cloth': 2, 'Footstool': 2,
               'ShowerHead': 2, 'TowelHolder': 2, 'ToiletPaper': 2, 'HandTowelHolder': 4}

# ------------------------------------------------------------------------------
# This is a keys dictionary pairing keys in idx_2_obj_list and glove_embedding
# ------------------------------------------------------------------------------
THOR_2_GLOVE = {'StoveBurner': 'cookstove', 'Drawer': 'drawer', 'CounterTop': 'countertop', 'Cabinet': 'cabinet', 'StoveKnob': 'knobset', 'Window': 'window', 'Sink': 'sink', 'Book': 'book', 'Bottle': 'bottle', 'Knife': 'knife', 'Microwave': 'microwave', 'Bread': 'bread', 'Fork': 'fork', 'Shelf': 'shelf', 'Potato': 'potato', 'HousePlant': 'houseplant', 'Toaster': 'toaster', 'SoapBottle': 'bottle', 'Kettle': 'kettle', 'Pan': 'pan', 'Plate': 'plate', 'Tomato': 'tomato', 'Vase': 'vase', 'GarbageCan': 'garbage-can', 'Egg': 'egg', 'CreditCard': 'creditcard', 'WineBottle': 'winebottle', 'Pot': 'pot', 'Spatula': 'spatula', 'PaperTowelRoll': 'papertowel', 'Cup': 'cup', 'Fridge': 'fridge', 'CoffeeMachine': 'coffeemachine', 'Bowl': 'bowl', 'SinkBasin': 'sink', 'SaltShaker': 'saltshaker', 'PepperShaker': 'saltshaker', 'Lettuce': 'lettuce', 'ButterKnife': 'butterknife', 'Apple': 'apple', 'DishSponge': 'sponge', 'Spoon': 'spoon', 'LightSwitch': 'lightswitch', 'Mug': 'mug', 'ShelvingUnit': 'shelf', 'Statue': 'statue', 'Stool': 'stool', 'Faucet': 'faucet', 'Ladle': 'ladle', 'CellPhone': 'cellphone', 'Chair': 'chair', 'SideTable': 'sidetable', 'DiningTable': 'diningtable', 'Pen': 'pen', 'SprayBottle': 'sprayer', 'Curtains': 'curtains', 'Pencil': 'pencil', 'Blinds': 'blinds', 'GarbageBag': 'plastic-bag', 'Safe': 'safe', 'Mirror': 'mirror', 'AluminumFoil': 'foil', 'Painting': 'painting', 'Box': 'box', 'Laptop': 'laptop', 'Television': 'television', 'TissueBox': 'tissue', 'KeyChain': 'keychain', 'FloorLamp': 'floorlamp', 'DeskLamp': 'desklamp', 'Pillow': 'pillow', 'RemoteControl': 'remotecontrol', 'Watch': 'watch', 'Newspaper': 'newspaper', 'ArmChair': 'armchair', 'CoffeeTable': 'coffeetable', 'TVStand': 'shelf', 'Sofa': 'sofa', 'WateringCan': 'sprayer', 'Boots': 'boots', 'Ottoman': 'ottoman', 'Desk': 'desk', 'Dresser': 'dresser', 'DogBed': 'dogbed', 'Candle': 'candle', 'RoomDecor': 'decoration', 'Bed': 'bed', 'BaseballBat': 'baseballbat', 'BasketBall': 'basketball', 'AlarmClock': 'alarmclock', 'CD': 'cd', 'TennisRacket': 'racket', 'TeddyBear': 'teddybear', 'Poster': 'poster', 'Cloth': 'cloth', 'Dumbbell': 'dumbbell', 'LaundryHamper': 'basket', 'TableTopDecor': 'decoration', 'Desktop': 'desktop', 'Footstool': 'footstool', 'VacuumCleaner': 'vacuumcleaner', 'BathtubBasin': 'bathtub', 'ShowerCurtain': 'shower-curtain', 'ShowerHead': 'showerhead', 'Bathtub': 'bathtub', 'Towel': 'towel', 'HandTowel': 'handtowel', 'Plunger': 'plunger', 'TowelHolder': 'hanger', 'ToiletPaperHanger': 'hanger', 'SoapBar': 'soapbar', 'ToiletPaper': 'toiletpaper', 'HandTowelHolder': 'hanger', 'ScrubBrush': 'brush', 'Toilet': 'toilet', 'ShowerGlass': 'glassdoor', 'ShowerDoor': 'door'}
# ------------------------------------------------------------------------------
# These two lists are used to determine the extend of object dynamics based on intrinsic convention of human activity
# ------------------------------------------------------------------------------
HIGH_DYNAMICS = ['GarbageCan', 'Stool', 'Chair', 'GarbageBag',
                 'LaundryHamper', 'Desktop', 'VacuumCleaner',
                 'Ottoman', 'DogBed']

LOW_DYNAMICS = ['Microwave', 'CoffeeMachine', 'ShelvingUnit', 'DiningTable',
                'DeskLamp', 'ArmChair', 'Toaster', 'SideTable', 'RoomDecor',
                'CoffeeTable', 'TVStand', 'Sofa', 'Safe', 'Television',
                'Desk', 'Dresser', 'Bed', 'HousePlant', 'FloorLamp']
# ------------------------------------------------------------------------------
LOW_DYNAMICS_MOVING_RATIO = 0.1 # THreshold for object shuffling range
MASS_MIN = 0.0 # Minimum furniture mass
MASS_MAX = 103.999992 # Maximum furniture mass
ROTATE_MAX_DEG = 10 # Maximum furniture rotate angle in degree during random shuffling
# ------------------------------------------------------------------------------

'''
This part is used to document node list (manual construction)
'''
# ------------------------------------------------------------------------------
# TOPOLOGICAL MAP NODES [X,Z]
# ------------------------------------------------------------------------------
# node A and node B have same z coordnates and x coordnates is shifted by ADJACENT_NODES_SHIFT_GRID grids ---> consider to be adjacent to each other
# This is set based on yidong's navigation heatmap
ADJACENT_NODES_SHIFT_GRID = 2
FORWARD_GRID = ADJACENT_NODES_SHIFT_GRID*2
NODES = {'FloorPlan26': [[-2.00, 3.75], [-1.25, 3.75], [-2.00, 3.00], [-1.25, 3.00],
                         [-2.00, 1.75], [-1.25, 1.75], [-2.00, 1.00], [-1.25, 1.00],
                         [-2.00, 4.50], [-1.25, 4.50], [-2.75, 1.00], [-2.75, 1.75],
                         [-0.50, 1.00], [-0.50, 1.75], [-2.00, 2.25], [-1.25, 2.25],
                         [-0.50, 2.25]],
         'FloorPlan27': [[0.00, -0.50], [0.75, -0.50], [1.50, -0.50], [0.50, 0.25],
                         [1.50, 0.25], [0.50, 1.00], [1.25, 1.00], [0.50, 2.00], [1.25, 2.00]],
         'FloorPlan226': [[-2.50, -1.00], [-1.75, -1.00], [-1.75, -0.50], [-1.75, 0.50], [-2.50, -0.50], [-2.25, 0.00],
                          [-0.75, 0.25], [0.25, 0.25], [1.25, 0.25], [1.25, 0.75], [0.25, 0.75], [0.75, -0.50],
                          [0.25, -0.50], [0.75, -1.25], [1.75, -1.25], [1.25, -2.00], [0.75, -2.00], [-1.75, 0.25],
                          [-0.75, -1.00], [0.00, -1.00], [-0.75, 0.50], [-0.75, -1.25], [0.00, -1.25], [-1.25, -1.25]],
         'FloorPlan227': [[-5.75, 0.75], [-6.75, 0.75], [-5.75, 0.25], [-6.75, 4.25], [-6.75, 4.75], [-5.00, 3.75],
                          [-5.75, 4.25], [-4.75, 3.75], [-4.75, 2.00], [-4.75, 0.75], [-3.50, 2.00], [-3.50, 0.75],
                          [-3.50, 3.00], [-2.50, 0.75], [-1.50, 0.75], [-1.75, 1.75], [-0.50, 0.75], [-0.50, 2.75],
                          [-0.25, 2.75], [-0.25, 3.25], [-0.25, 4.25], [-1.25, 4.25], [-2.25, 4.25], [-3.25, 4.25],
                          [-4.75, 4.25], [-5.75, 4.75], [-4.75, 4.75], [-1.25, 4.75], [-2.25, 4.75], [-3.25, 4.75],
                          [-4.75, 1.25], [-3.50, 1.25], [-2.50, 1.25], [-1.50, 1.25], [-0.75, 3.75], [-4.00, 4.25],
                          [-4.00, 4.75], [-0.25, 1.50], [-0.50, 1.25], [-4.75, 3.00], [-4.00, 2.00], [-4.00, 3.00],
                          [-4.00, 0.75], [-4.00, 1.25]],
         'FloorPlan228': [[-4.75, 3.75], [-4.25, 3.75], [-3.25, 3.75], [-2.50, 3.75], [-1.75, 3.75],
                          [-1.00, 3.75], [-0.25, 3.75], [-4.75, 3.50], [-4.25, 3.50], [-3.25, 3.50],
                          [-4.75, 4.75], [-4.75, 1.50], [-4.25, 1.50], [-3.25, 1.75], [-2.50, 1.75],
                          [-4.75, 3.00], [-4.75, 2.25], [-3.25, 3.00], [-3.25, 2.25], [-2.50, 3.00],
                          [-2.50, 2.25], [-2.50, 3.25], [-1.75, 3.25], [-1.00, 3.25], [-0.25, 3.25],
                          [-1.75, 4.25], [-1.00, 4.25], [-0.25, 4.25], [-3.50, 1.25], [-2.50, 1.25],
                          [-2.00, 4.75], [-1.00, 4.75], [-1.25, 2.50], [-0.50, 2.50]],
         'FloorPlan326': [[0.25, -0.50], [0.25, -1.00], [0.25, -2.00], [-0.50, -2.00], [-1.50, -2.00], [-2.25, -2.00],
                          [-1.50, -1.50], [-1.50, -0.75], [-1.50, 0.00], [1.00, -1.00], [1.00, -0.50], [2.00, -0.50],
                          [2.00, -1.00], [1.75, -1.75], [2.50, -1.75], [0.50, -2.75], [-0.50, -2.75], [-1.50, -2.75],
                          [-2.50, -2.75], [0.25, 0.25], [1.00, 0.25], [2.00, 0.25], [2.25, -0.50], [2.25, -2.50],
                          [3.00, -2.75]],
         'FloorPlan327': [[-0.75, -2.00], [-0.25, -1.50], [-0.25, -0.50], [-0.25, 0.50], [-0.25, 1.25], [-1.00, 1.25],
                          [0.25, 1.25], [0.25, 0.50], [0.25, -0.50], [0.25, -1.50], [0.25, 1.75], [-1.00, 1.75],
                          [-0.25, 1.75], [0.75, 1.25], [0.75, 0.50], [0.75, -0.50], [0.75, -1.50], [1.25, 0.75],
                          [1.25, 1.50]],
         'FloorPlan426': [[-1.25, 3.00], [-1.25, 2.50], [-0.25, 2.50], [-1.25, 2.00], [-0.25, 2.00],
                          [-1.50, 1.50], [-0.50, 1.50], [-1.50, 1.00], [-0.50, 1.00], [-2.00, 1.00],
                          [-2.00, 1.50], [-2.75, 1.00], [-2.75, 0.50]],
         'FloorPlan427': [[-2.75, 1.25], [-2.75, 0.25], [-2.25, 1.25], [-2.25, 0.25], [-0.75, 1.25],
                          [-0.75, 0.25], [-1.50, 1.25], [-1.50, 0.25], [-0.75, 2.00], [-2.50, 2.00]],
         'FloorPlan428': [[0.75, 3.75], [-0.25, 3.75], [-1.25, 3.75], [0.75, 3.00], [-0.25, 3.00],
                          [-1.25, 3.00], [-0.25, 2.50], [-1.25, 2.50], [-0.75, 2.00], [0.25, 2.00],
                          [-0.75, 1.50], [0.25, 1.50]],
#---------------------------
         'FloorPlan28': [[-1.00, -3.00], [-1.00, -2.00], [-1.00, -1.00], [-1.00, -0.25],
                         [-1.75, -3.00], [-1.75, -2.00], [-1.75, -1.00], [-1.75, -0.25],
                         [-2.50, -3.00], [-2.50, -2.00], [-2.50, -1.00], [-2.50, -0.25],
                         [-3.25, -2.25], [-3.25, -1.25], [-4.00, -2.25], [-4.00, -1.25]],
         'FloorPlan29': [[1.00, 2.25], [1.00, 1.50], [0.25, 2.25], [0.25, 1.50], [1.75, 2.00],
                         [-0.50, 2.25], [-0.50, 1.50], [-0.50, 0.50], [-1.25, 0.50],
                         [-0.50, -0.50], [-1.25, -0.5], [0.5, -0.25], [0.5, -0.75],
                         [1.5, -0.25], [1.5, -0.75], [-1.25, 1.00]],
         'FloorPlan30': [[2.50, -1.50], [1.75, -1.50], [2.50, -0.50], [2.00, -0.50],
                         [2.50, 0.50], [2.00, 0.50], [2.50, 1.50], [2.00, 1.50],
                         [1.25, -0.50], [0.00, -0.50], [0.00, 0.50], [0.00, 1.50],
                         [1.25, 1.50], [0.50, -1.00], [0.50, -0.50], [0.50, 1.50]],

          'FloorPlan229': [[-0.25, 3.00], [-0.25, 2.25], [-1.00, 3.00], [-1.00, 2.25],
                           [-1.00, 1.25], [-1.50, 1.25], [-1.00, 0.25], [-1.50, 0.25],
                           [-1.50, 2.25], [-2.50, 1.50], [-3.50, 1.50], [-4.50, 1.50],
                           [-5.25, 1.50], [-4.75, 1.00], [-5.75, 1.00], [-0.50, 4.00],
                           [-0.75, 4.25], [-1.25, 4.50], [-2.25, 4.50], [-3.25, 4.50],
                           [-4.25, 4.50], [-5.25, 4.50], [-3.25, 3.75], [-2.75, 3.75],
                           [-3.25, 3.00], [-2.75, 3.00], [-1.75, 3.00], [-2.25, 2.25],
                           [-1.75, 1.50], [-4.25, 3.00], [-0.50, 3.50], [-0.50, 3.50],
                           [-4.25, 2.25], [-5.25, 2.25], [-5.25, 3.00], [-5.25, 3.75],
                           [-4.50, 3.75], [-5.75, 3.25], [-4.00, 1.75]],
          'FloorPlan230': [[-5.75, 4.25], [-5.75, 3.25], [-5.75, 2.25], [-5.00, 4.25],
                           [-5.00, 3.25], [-5.00, 2.25], [-5.00, 1.25], [-5.00, 0.25],
                           [-5.50, 1.00],
                           [-4.25, 4.25], [-4.25, 3.25], [-4.25, 2.25], [-4.25, 1.25],
                           [-5.00, 5.25], [-4.25, 5.25], [-5.00, 6.25], [-4.25, 6.25],
                           [-5.00, 7.25], [-4.25, 7.25], [-5.00, 8.25], [-4.25, 8.25],
                           [-5.00, 8.75], [-4.25, 8.75], [-4.00, 7.75], [-3.25, 7.75],
                           [-2.50, 8.00], [-2.50, 7.75], [-2.50, 7.00], [-2.50, 6.25],
                           [-2.50, 5.25], [-3.25, 5.25], [-3.75, 0.75], [-2.75, 0.75],
                           [-3.50, 3.50], [-2.50, 3.50], [-3.50, 4.00], [-2.50, 4.00],
                           [-1.50, 8.00], [-1.50, 8.75], [-0.50, 8.00], [-0.50, 7.00],
                           [-0.50, 6.25], [-1.00, 8.00], [-1.00, 7.00], [-1.00, 6.25],
                           [-1.00, 5.25], [-0.25, 5.25], [-1.00, 4.25], [-0.25, 4.25],
                           [-2.00, 4.25], [-2.00, 3.50], [-1.00, 3.50], [-1.00, 2.50],
                           [-0.25, 3.50], [-0.25, 2.50], [-1.50, 3.50], [-1.50, 2.50],
                           [-1.50, 1.50], [-1.00, 1.50], [-1.75, 0.75], [-1.00, 0.75]],

           'FloorPlan328': [[2.00, 0.75], [3.00, 0.75], [1.00, 0.75], [0.00, 0.75],
                            [2.00, 0.25], [3.00, 0.25], [1.00, 0.25], [0.00, 0.25],
                            [2.00, -0.25],[1.25, -0.25], [2.75, -0.25],
                            [2.00, -0.75],[1.25, -0.75], [2.75, -0.75]],
           'FloorPlan329': [[2.25, -2.00], [1.75, -2.00], [1.00, -2.00], [2.25, -1.00],
                            [1.75, -1.00], [1.00, -1.00], [1.75, 0.00], [1.00, 0.00],
                            [1.75, 0.75], [1.00, 0.75], [0.50, -0.50], [0.50, 0.25],
                            [-0.50, -0.50], [-0.50, 0.25], [-1.25, -0.50], [-1.25, 0.25],
                            [-1.25, -1.50]],
           'FloorPlan330': [[-0.25, 2.00], [-1.00, 2.00], [-1.75, 2.00], [0.5, 2.00],
                            [-1.00, 1.00], [-0.25, 1.00], [0.5, 1.00], [-1.00, -2.75],
                            [-1.00, 0.00], [-0.25, 0.00], [-1.00, -1.00], [-0.25, -1.00],
                            [-1.00, -1.75], [-0.25, -1.75], [1.00, 2.00], [1.00, 1.00],
                            [1.75, 1.00], [2.50, 1.00], [1.75, 1.00], [2.50, 0.00],
                            [2.50, -1.00]],

           'FloorPlan429': [[0.25, -0.25], [0.25, -1.25], [0.25, -2.25], [-0.75, -0.25],
                            [-0.75, -1.25], [-0.75, -1.00], [-0.75, -2.25], [-0.75, -3.25],
                            [-1.75, -1.00], [0.25, -2.75], [0.75, -2.75], [0.75, -2.75],
                            [0.75, -3.50], [-1.75, -0.50]],
           'FloorPlan430': [[-1.50, 2.00], [-2.25, 2.00], [-2.25, 1.25], [-1.50, 1.25],
                            [-1.50, 1.25], [-1.50, 0.25], [-2.25, 0.25], [-0.50, 0.25],
                            [0.25, 0.25], [-0.50, 1.25], [0.25, 1.25], [0.25, -0.50],
                            [-0.50, -0.50], [-1.50, -0.50], [-0.50, -1.25], [0.00, -1.25]]
        }

# The door is visible in node_i subnode_j: thus DOOR[scene_name] = (node_i, subnode_j)
# Note Subnode is in degree [00, 90.0, 180.0, 270.0]
DOOR_NODE = {'FloorPlan26': (7, 90.0),
             'FloorPlan27': (1, 180.0),
             'FloorPlan226': (0, 180.0),
             'FloorPlan227': None,
             'FloorPlan228': (4, 90.0),
             'FloorPlan326': (0, 0.0),
             'FloorPlan327': (4, 0.0),
             'FloorPlan426': (0, 90.0),
             'FloorPlan427': (4, 90.0),
             'FloorPlan428': (0, 90.0),

             'FloorPlan28': (15, 270.0),
             'FloorPlan29': (0, 0.0),
             'FloorPlan30': (0, 180.0),

             'FloorPlan229': (0, 90.0),
             'FloorPlan230': (0, 270.0),

             'FloorPlan328': (0, 0.0),
             'FloorPlan329': (0, 90.0),
             'FloorPlan330': (0, 0.0),

             'FloorPlan429': (0, 0.0),
             'FloorPlan430': (0, 0.0)
            }
