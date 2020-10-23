# Module for setup params and global variables
from os.path import dirname, abspath
import numpy as np

SIM_WINDOW_HEIGHT = 700
SIM_WINDOW_WIDTH = 900
BAN_TYPE_LIST = ['Floor']   # Ignore non-informative objectType e.g. 'Floor
THIRD_PARTY_PATH = dirname(dirname(abspath(__file__))) + '/3rdparty' # File path for 3rdparty info
GLOVE_FILE_NAME = 'glove.42B.300d.txt' # file name of glove vector
glove_embedding = np.load(THIRD_PARTY_PATH + '/' + 'glove_embedding.npy', allow_pickle='TRUE').item() # all Glove word Embedding vectors
THOR_2_VEC = np.load(THIRD_PARTY_PATH + '/' + 'THOR_2_VEC.npy', allow_pickle='TRUE').item() # objectType embedding using GLoVe vectors
INFO_FILE_PATH = dirname(dirname(abspath(__file__))) + '/AI2THOR_info' # File path for info of iTHOR Env.
obj_2_idx_dic = np.load(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', allow_pickle='TRUE').item()
idx_2_obj_list = np.load(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy')
OBJ_TYPE_NUM = len(idx_2_obj_list) # Maximum numbers of objectType in iTHOR Env.
PROXIMITY_THRESHOLD = 3 # distance ratio threshold for proximity determination
CLUSTERING_RADIUS_RATIO = 1.0 # used to cluster drawers and cabinets, increase to allow larger tolerance
VISBILITY_DISTANCE = 1.5 # default 1.5 meter, object within the radius of a cylinder centered about the y-axis of the agent is visible
FIELD_OF_VIEW = 120 # default 90 degree, 120 degree is binocular FoV
SUB_NODES_NUM = 4
SIMILARITY_GRID_ORDER = 2 # Approx Grid Size of 10^SIMILARITY_GRID_ORDER for similarity score between views
ITHOR_FLOOR_PLANS = np.load(INFO_FILE_PATH + '/' + 'iTHOR_FloorPlan.npy') # list of floorplan name for iTHOR
ROBOTHOR_FLOOR_PLANS = np.load(INFO_FILE_PATH + '/' + 'RoboTHOR_FloorPlan.npy') # list of floorplan name for RoboTHOR
# This is after group up
GROUP_UP_LIST = ['Drawer', 'Cabinet', 'Shelf'] # Used to group up massive numbers of receptacles in SG module
# Maximum numbers of receptacles appeared in one scene after group_up()
REC_MAX_DIC = {'Drawer': 6, 'CounterTop': 4, 'Cabinet': 8, 'Shelf': 6, 'Pot': 2,
               'SinkBasin': 2, 'Stool': 2, 'Chair': 8, 'Sink': 2, 'SideTable': 8,
               'DiningTable': 3, 'Box': 4, 'ArmChair': 4, 'CoffeeTable': 4,
               'TVStand': 2, 'Sofa': 2, 'Desk': 5, 'Dresser': 4, 'Bed': 2, 'Footstool': 2}

# This is a keys dictionary pairing keys in idx_2_obj_list and glove_embedding
THOR_2_GLOVE = {'StoveBurner': 'cookstove', 'Drawer': 'drawer', 'CounterTop': 'countertop', 'Cabinet': 'cabinet', 'StoveKnob': 'knobset', 'Window': 'window', 'Sink': 'sink', 'Book': 'book', 'Bottle': 'bottle', 'Knife': 'knife', 'Microwave': 'microwave', 'Bread': 'bread', 'Fork': 'fork', 'Shelf': 'shelf', 'Potato': 'potato', 'HousePlant': 'houseplant', 'Toaster': 'toaster', 'SoapBottle': 'bottle', 'Kettle': 'kettle', 'Pan': 'pan', 'Plate': 'plate', 'Tomato': 'tomato', 'Vase': 'vase', 'GarbageCan': 'garbage-can', 'Egg': 'egg', 'CreditCard': 'creditcard', 'WineBottle': 'winebottle', 'Pot': 'pot', 'Spatula': 'spatula', 'PaperTowelRoll': 'papertowel', 'Cup': 'cup', 'Fridge': 'fridge', 'CoffeeMachine': 'coffeemachine', 'Bowl': 'bowl', 'SinkBasin': 'sink', 'SaltShaker': 'saltshaker', 'PepperShaker': 'saltshaker', 'Lettuce': 'lettuce', 'ButterKnife': 'butterknife', 'Apple': 'apple', 'DishSponge': 'sponge', 'Spoon': 'spoon', 'LightSwitch': 'lightswitch', 'Mug': 'mug', 'ShelvingUnit': 'shelf', 'Statue': 'statue', 'Stool': 'stool', 'Faucet': 'faucet', 'Ladle': 'ladle', 'CellPhone': 'cellphone', 'Chair': 'chair', 'SideTable': 'sidetable', 'DiningTable': 'diningtable', 'Pen': 'pen', 'SprayBottle': 'sprayer', 'Curtains': 'curtains', 'Pencil': 'pencil', 'Blinds': 'blinds', 'GarbageBag': 'plastic-bag', 'Safe': 'safe', 'Mirror': 'mirror', 'AluminumFoil': 'foil', 'Painting': 'painting', 'Box': 'box', 'Laptop': 'laptop', 'Television': 'television', 'TissueBox': 'tissue', 'KeyChain': 'keychain', 'FloorLamp': 'floorlamp', 'DeskLamp': 'desklamp', 'Pillow': 'pillow', 'RemoteControl': 'remotecontrol', 'Watch': 'watch', 'Newspaper': 'newspaper', 'ArmChair': 'armchair', 'CoffeeTable': 'coffeetable', 'TVStand': 'shelf', 'Sofa': 'sofa', 'WateringCan': 'sprayer', 'Boots': 'boots', 'Ottoman': 'ottoman', 'Desk': 'desk', 'Dresser': 'dresser', 'DogBed': 'dogbed', 'Candle': 'candle', 'RoomDecor': 'decoration', 'Bed': 'bed', 'BaseballBat': 'baseballbat', 'BasketBall': 'basketball', 'AlarmClock': 'alarmclock', 'CD': 'cd', 'TennisRacket': 'racket', 'TeddyBear': 'teddybear', 'Poster': 'poster', 'Cloth': 'cloth', 'Dumbbell': 'dumbbell', 'LaundryHamper': 'basket', 'TableTopDecor': 'decoration', 'Desktop': 'desktop', 'Footstool': 'footstool', 'VacuumCleaner': 'vacuumcleaner', 'BathtubBasin': 'bathtub', 'ShowerCurtain': 'shower-curtain', 'ShowerHead': 'showerhead', 'Bathtub': 'bathtub', 'Towel': 'towel', 'HandTowel': 'handtowel', 'Plunger': 'plunger', 'TowelHolder': 'hanger', 'ToiletPaperHanger': 'hanger', 'SoapBar': 'soapbar', 'ToiletPaper': 'toiletpaper', 'HandTowelHolder': 'hanger', 'ScrubBrush': 'brush', 'Toilet': 'toilet', 'ShowerGlass': 'glassdoor', 'ShowerDoor': 'door'}

# These two lists are used to determine the extend of object dynamics based on intrinsic convention of human activity
HIGH_DYNAMICS = ['GarbageCan', 'Stool', 'Chair', 'GarbageBag',
                 'FloorLamp', 'DeskLamp', 'ArmChair', 'Toaster', 'SideTable',
                 'LaundryHamper', 'Desktop', 'VacuumCleaner', 'RoomDecor',
                 'Ottoman', 'DogBed']

LOW_DYNAMICS = ['Microwave', 'CoffeeMachine', 'ShelvingUnit', 'DiningTable',
                'CoffeeTable', 'TVStand', 'Sofa', 'Safe', 'Television',
                'Desk', 'Dresser', 'Bed', 'HousePlant']
LOW_DYNAMICS_MOVING_RATIO = 0.5 # THreshold for object shuffling range
MASS_MIN = 0.0 # Minimum furniture mass
MASS_MAX = 103.999992 # Maximum furniture mass
ROTATE_MAX_DEG = 10 # Maximum furniture rotate angle in degree during random shuffling
