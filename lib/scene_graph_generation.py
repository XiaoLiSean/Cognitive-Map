# Scene graph Module for SSG
from scipy.sparse import lil_matrix, find
from scipy.sparse import find as find_sparse_idx
from mpl_toolkits.mplot3d import Axes3D
from termcolor import colored
from PIL import Image
from copy import deepcopy
from lib.params import *
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import sys, time, io, torch

#-------------------------------------------------------------------------------
# Function to generate points for plot_surface a cuboid
def cuboid_data(pos, size):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    o = [pos['x'], pos['z'], -pos['y']] # This is reorgnized to suit AI2THOR coordinates
    l = size['x']
    w = size['z']
    h = size['y']

    x = [[o[0] - 0.5*l, o[0] + 0.5*l, o[0] + 0.5*l, o[0] - 0.5*l, o[0] - 0.5*l],
         [o[0] - 0.5*l, o[0] + 0.5*l, o[0] + 0.5*l, o[0] - 0.5*l, o[0] - 0.5*l],
         [o[0] - 0.5*l, o[0] + 0.5*l, o[0] + 0.5*l, o[0] - 0.5*l, o[0] - 0.5*l],
         [o[0] - 0.5*l, o[0] + 0.5*l, o[0] + 0.5*l, o[0] - 0.5*l, o[0] - 0.5*l]]
    y = [[o[1] - 0.5*w, o[1] - 0.5*w, o[1] + 0.5*w, o[1] + 0.5*w, o[1] - 0.5*w],
         [o[1] - 0.5*w, o[1] - 0.5*w, o[1] + 0.5*w, o[1] + 0.5*w, o[1] - 0.5*w],
         [o[1] - 0.5*w, o[1] - 0.5*w, o[1] - 0.5*w, o[1] - 0.5*w, o[1] - 0.5*w],
         [o[1] + 0.5*w, o[1] + 0.5*w, o[1] + 0.5*w, o[1] + 0.5*w, o[1] + 0.5*w]]
    z = [[o[2] - 0.5*h, o[2] - 0.5*h, o[2] - 0.5*h, o[2] - 0.5*h, o[2] - 0.5*h],
         [o[2] + 0.5*h, o[2] + 0.5*h, o[2] + 0.5*h, o[2] + 0.5*h, o[2] + 0.5*h],
         [o[2] - 0.5*h, o[2] - 0.5*h, o[2] + 0.5*h, o[2] + 0.5*h, o[2] - 0.5*h],
         [o[2] - 0.5*h, o[2] - 0.5*h, o[2] + 0.5*h, o[2] + 0.5*h, o[2] - 0.5*h]]

    return np.array(x), np.array(y), np.array(z)

#-------------------------------------------------------------------------------
def plot_cube_at(pos, size, ax=None):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,alpha=0.5, linewidth=0,color='limegreen')

#-------------------------------------------------------------------------------
# This function is used to draw bounding boxs from objects
def draw_bb_from_objs(objs):
    fig = plt.figure()
    ax = fig.gca(projection='3d', aspect='auto')
    for obj in objs:
        pos = obj['axisAlignedBoundingBox']['center']
        size = obj['axisAlignedBoundingBox']['size']
        plot_cube_at(pos, size, ax=ax)

    plt.show()

#-------------------------------------------------------------------------------
# This function is used to judge if two objects: cabinets or drawers can form a group
def in_group(drawer_group, new_drawer):
    in_group = False
    for grouped_drawer in drawer_group:
        grouped = grouped_drawer['axisAlignedBoundingBox']
        # Test if those two are close enough
        new = new_drawer['axisAlignedBoundingBox']
        distance = np.linalg.norm(np.array([new['center']['x'], new['center']['y'], new['center']['z']])
                                  - np.array([grouped['center']['x'], grouped['center']['y'], grouped['center']['z']]))
        new_r = np.linalg.norm(np.array([0.5*new['size']['x'], 0.5*new['size']['y'], 0.5*new['size']['z']]))
        grouped_r = np.linalg.norm(np.array([0.5*grouped['size']['x'], 0.5*grouped['size']['y'], 0.5*grouped['size']['z']]))
        if distance < (new_r+grouped_r)*CLUSTERING_RADIUS_RATIO:
            in_group = True

    return in_group

#-------------------------------------------------------------------------------
# This function is used to indexing receptacles with multiple instances and groups
def add_index(objs):
    ith_obj = deepcopy(REC_MAX_DIC)
    for i in range(len(objs)):
        name = objs[i]['objectType']
        if name in REC_MAX_DIC:
            idx = REC_MAX_DIC[name] - ith_obj[name]
            if idx >= REC_MAX_DIC[name]:
                sys.stderr.write(colored('ERROR: ','red') + "Index of {} exceeds maximum instances {}/{}\n".format(name, idx, REC_MAX_DIC[name]))
                sys.exit(1)
            objs[i].update({'instance_i': idx})
            ith_obj[name] -= 1

    return objs

#-------------------------------------------------------------------------------
# This function is used to get cornerPoints out of center and size lists
def get_cornerPoints(center, size):
    o = center
    l = size / 2.0
    cornerPoints = [ [o[0]+l[0], o[1]+l[1], o[2]-l[2]],
                     [o[0]+l[0], o[1]+l[1], o[2]+l[2]],
                     [o[0]+l[0], o[1]-l[1], o[2]-l[2]],
                     [o[0]+l[0], o[1]-l[1], o[2]+l[2]],
                     [o[0]-l[0], o[1]+l[1], o[2]-l[2]],
                     [o[0]-l[0], o[1]+l[1], o[2]+l[2]],
                     [o[0]-l[0], o[1]-l[1], o[2]-l[2]],
                     [o[0]-l[0], o[1]-l[1], o[2]+l[2]] ]
    return cornerPoints

#-------------------------------------------------------------------------------
# This function is used to preprocess the object data before update the SG
# To group up the objectType 'Drawer' and 'Cabinet' based on their structure and position
# Group up their 2d and 3d bounding boxs
'''
Used to group up massive numbers of receptacles or identical object in SG module
Objects in these objectTypes are typically presented in array or regular spatial order.
'''
def group_up(objs, visualization_on=False, verbose=False):
    # iterate over all possible receptacles which needs to be grouped up
    for name in GROUP_UP_LIST:
        receptacles = [] # used to temporarily store all receptacles
        # Feed in all drawers and cabinets into lists
        tmp_objs = deepcopy(objs)
        for obj in tmp_objs:
            if obj['objectType'] == name:
                receptacles.append(obj)
                objs.remove(obj)

        # Case There is no receptacle
        if len(receptacles) == 0:
            if verbose:
                print(colored('Group info: ','blue')+'No '+name)
            continue

        # Visualize all the receptacles in same name
        if visualization_on:
            draw_bb_from_objs(receptacles)

        # used to output results of group
        output = '{} {} '.format(len(receptacles), name)
        # This is preliminary process of receptacle groups
        # receptacle_groups = [receptacle_group_1, receptacle_group_2] = [[receptacle_1, receptacle_2],[receptacle_3, receptacle_4]]
        # initialize receptacle_groups
        receptacle_groups = []
        receptacle_groups.append([receptacles.pop(0)])

        # iteration through all receptacles
        # Algorithm to radiatively get in group members for the last added group
        while True:
            least_one_grouped = False
            # [:] or deepcopy() is used to assign new list, not adding [:] or deepcopy() is essentially a pointer
            tmp_receptacles = deepcopy(receptacles)
            for new_receptacle in tmp_receptacles:
                # [:] or deepcopy() is used to assign new list, not adding [:] or deepcopy() is essentially a pointer
                tmp_receptacle_group = deepcopy(receptacle_groups[-1])
                # Test if those two are close enough
                if in_group(tmp_receptacle_group, new_receptacle):
                    receptacle_groups[-1].append(new_receptacle)
                    receptacles.remove(new_receptacle)
                    least_one_grouped = True
                if len(receptacles) == 0:
                    break
            if not least_one_grouped and len(receptacles) != 0:
                receptacle_groups.append([receptacles.pop(0)])
            if len(receptacles) == 0:
                break

        # Prepare output to show group info
        output += 'formed {} groups:'.format(len(receptacle_groups))
        for receptacle_group in receptacle_groups:
            output += ' [{}] '.format(len(receptacle_group))
        if verbose:
            print(colored('Group info: ','blue')+output)

        # Re-define the objs structure:
        # i.e. receptacle_groups = [receptacle_group_1, receptacle_group_2]
        # objs = [receptacle_group_1, receptacle_group_2, obj1, obj2]
        # receptacle_group_1 = {'objectType': name, 'receptacle': True,
        #                       'members': [receptacle_1, receptacle_2], 'parentReceptacles': None,
        #                       'axisAlignedBoundingBox': {'cornerPoints': [[]...[]],
        #                       'center': {'x': -1.31980932, 'y': 1.40200138, 'z': -0.994497657},
        #                       'size': {'x': 0.142013311, 'y': 0.3271618, 'z': 0.1831708}}
        #                       }
        for receptacle_group in receptacle_groups:
            XYZ = []
            UV = []
            # This is used to generate new axisAlignedBoundingBox
            for receptacle in receptacle_group:
                if '2DBoundingBox' in receptacle:
                    UV.append(receptacle['2DBoundingBox'])
                for point in receptacle['axisAlignedBoundingBox']['cornerPoints']:
                    XYZ.append(point)

            XYZ = np.asarray(XYZ)
            max_XYZ = np.max(XYZ,axis=0)
            min_XYZ = np.min(XYZ,axis=0)
            size = max_XYZ - min_XYZ
            center = (max_XYZ + min_XYZ) / 2.0
            cornerPoints = get_cornerPoints(center, size)
            axisAlignedBoundingBox = {'cornerPoints': cornerPoints,
                                      'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                                      'size': {'x': size[0], 'y': size[1], 'z': size[2]}}
            if len(UV) >= 1:
                UV = np.asarray(UV)
                objectBoundingBox = [np.min(UV[:,0]), np.min(UV[:,1]), np.max(UV[:,2]), np.max(UV[:,3])]
                objs.append({'objectType': name, 'receptacle': True,
                             'parentReceptacles': None, 'members': receptacle_group,
                             'axisAlignedBoundingBox': axisAlignedBoundingBox,
                             '2DBoundingBox': objectBoundingBox})
            else:
                objs.append({'objectType': name, 'receptacle': True,
                             'parentReceptacles': None, 'members': receptacle_group,
                             'axisAlignedBoundingBox': axisAlignedBoundingBox})

    return add_index(objs)

#-------------------------------------------------------------------------------
class Scene_Graph:
    #---------------------------------------------------------------------------
    def __init__(self, R_on=lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool),
                 R_in=lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool),
                 R_proximity=lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool),
                 fractional_bboxs=[np.array([0,0,0,0]) for i in range(OBJ_TYPE_NUM)]):
        # All vector and sparse matrix initialized with 'False' boolean
        self._obj_vec = lil_matrix((OBJ_TYPE_NUM, 1), dtype=np.bool)    # binary object occurence vector where index follow 'obj_2_idx_dic.npy'
        self._R_on = R_on    # Relationship sparse matrix _R_on[i,j] = True , obj_i on obj_j
        self._R_in = R_in
        self._R_proximity = R_proximity
        self._R_disjoint = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._fractional_bboxs = fractional_bboxs

    #---------------------------------------------------------------------------
    # reset all values in SG instance
    def reset(self):
        # All vector and sparse matrix initialized with 'False' boolean
        self._obj_vec = lil_matrix((OBJ_TYPE_NUM, 1), dtype=np.bool)    # binary object occurence vector where index follow 'obj_2_idx_dic.npy'
        self._R_on = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)    # Relationship sparse matrix _R_on[i,j] = True , obj_i on obj_j
        self._R_in = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._R_proximity = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._R_disjoint = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._fractional_bboxs = [np.array([0,0,0,0]) for i in range(OBJ_TYPE_NUM)]

    def visibleFilter_by_2Dbbox(self, objs, objectsBboxsDict):
        if objectsBboxsDict == None:
            print(colored('visibleFilter_2Dbbox info: No object detection or renderObjectImage is default in initialization','blue'))
            return []
        tmp_objs = deepcopy(objs)

        for obj in tmp_objs:
            if obj['objectId'] not in objectsBboxsDict:
                objs.remove(obj)
        for idx in range(len(objs)):
            objectId = objs[idx]['objectId']
            objs[idx].update({'2DBoundingBox': objectsBboxsDict[objectId].tolist()})

        return objs

    #---------------------------------------------------------------------------
    # Update Scene_Graph by object pair (obj_i, obj_j) and their Relationship r_ij.
    # r_ij = 'on', 'in', 'proximity' or 'disjoint'
    # eg: r_ij = 'on' implies obj_i on obj_j
    def update_SG(self, obj_i, obj_j, r_ij):
        self._obj_vec[obj_i, 0] = True # Note the presence of i'th obj
        self._obj_vec[obj_j, 0] = True # Note the presence of i'th obj
        if r_ij == 'on':
            self._R_on[obj_i, obj_j] = True
            self._R_on[obj_j, obj_i] = False     # The Relationship arrow in SG should be directional
            # priority filter in case two object with same objectType have distinct R with another obj:
            # 'on' > 'in' > 'proximity' > 'disjoint'
            # This is also used to rguarantee unique i-j Relationship
            self._R_in[obj_i, obj_j] = False
            self._R_in[obj_j, obj_i] = False
            self._R_proximity[obj_i, obj_j] = False
            self._R_proximity[obj_j, obj_i] = False
            self._R_disjoint[obj_i, obj_j] = False
            self._R_disjoint[obj_j, obj_i] = False
        elif r_ij == 'in':
            self._R_in[obj_i, obj_j] = True
            self._R_in[obj_j, obj_i] = False     # The Relationship arrow in SG should be directional
            self._R_proximity[obj_i, obj_j] = False # priority filter
            self._R_proximity[obj_j, obj_i] = False
            self._R_disjoint[obj_i, obj_j] = False # priority filter
            self._R_disjoint[obj_j, obj_i] = False
        # 'proximity' and 'disjoint' belong to mutual Relationship
        # r_ij point from small to larger obj: chair proximity to table
        elif r_ij == 'proximity':
            self._R_proximity[obj_i, obj_j] = True
            self._R_proximity[obj_j, obj_i] = True     # The Relationship arrow in SG should be directional
            self._R_disjoint[obj_i, obj_j] = False # priority filter
            self._R_disjoint[obj_j, obj_i] = False
        elif r_ij == 'disjoint':
            self._R_disjoint[obj_i, obj_j] = True
            self._R_disjoint[obj_j, obj_i] = True     # The Relationship arrow in SG should be directional
        else:
            sys.stderr.write(colored('ERROR: ','red')
                             + "Expect input r_ij = 'on', 'in', 'proximity' or 'disjoint' while get {}\n".format(r_ij))
            sys.exit(1)

    #---------------------------------------------------------------------------
    # Visualize Scene Graph
    def visualize_SG(self, comfirmed=None, axis=None):
        # comfirm is not none --> this function is used as a client node
        node_list = find_sparse_idx(self._obj_vec)[0]  # objectType/Node index list
        edges = []
        edge_labels = {}

        # append edges and edge_labels for r_ij = 'on', 'in', 'proximity' & 'disjoint'
        # eg: r_ij = 'on' implies obj_i on obj_jpurple
        edge_data = [self._R_on, self._R_in, self._R_proximity, self._R_disjoint]
        edge_type = ['on', 'in', 'proximity', 'disjoint']
        for k in range(len(edge_data)-1): # '-1' to Exclude 'disjoint' for visualization
            R = edge_data[k] # self._R_XXX
            i_list = find_sparse_idx(R)[0] # index for obj_i
            j_list = find_sparse_idx(R)[1] # index for obj_j
            for idx in range(len(i_list)):
                edges.append([str(i_list[idx]), str(j_list[idx])])
                if i_list[idx] == j_list[idx]:
                    edge_labels[(str(i_list[idx]), str(j_list[idx]))] = ''
                else:
                    edge_labels[(str(i_list[idx]), str(j_list[idx]))] = edge_type[k]

        # Construct Node-Edge graph
        G = nx.DiGraph(directed=True)
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)

        nx.draw(G, pos, edge_color='black',
                node_color='skyblue', alpha=0.9, ax=axis,
                labels={node:idx_2_obj_list[int(node)] for node in G.nodes()})
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=axis)

        if axis != None:
            return

        if comfirmed is None:
            plt.show() # plt.show() is a blocking function...

        # this SG function serve as a server node
        else:
            plt.show()
            print(colored('Client: ','green') + 'Receive Data from navigator')
            comfirmed.value = 1

    #---------------------------------------------------------------------------
    # Used to visualize SG and original Img with bounding boxs
    def visualize_data_point(self, img):
        fig, axs = plt.subplots(1, 2, figsize=(17,10))
        plt.axis('off')

        axs[0].axis('off')

        axs[0].imshow(img)
        image_size = img.size[0]

        for idx, uv in enumerate(self._fractional_bboxs):
            if np.all(uv == 0):
                continue
            else:
                uv = uv*image_size
                bbox = Rectangle((uv[0],uv[1]), uv[2]-uv[0], uv[3]-uv[1], ec='green', fill=False, linewidth=2)
                axs[0].add_patch(bbox)
                axs[0].text(uv[0],uv[1], idx_2_obj_list[idx])

        axs[1].axis('off')
        self.visualize_SG(axis=axs[1])
        plt.show()

    def visualize_one_in_triplet(self, img, ax1, ax2):
        ax1.axis('off')
        ax1.imshow(img)
        image_size = img.size[0]

        for idx, uv in enumerate(self._fractional_bboxs):
            if np.all(uv == 0):
                continue
            else:
                uv = uv*image_size
                bbox = Rectangle((uv[0],uv[1]), uv[2]-uv[0], uv[3]-uv[1], ec='green', fill=False, linewidth=2)
                ax1.add_patch(bbox)
                ax1.text(uv[0],uv[1], idx_2_obj_list[idx])

        ax2.axis('off')
        self.visualize_SG(axis=ax2)

    #---------------------------------------------------------------------------
    # This function is used to get object index
    def get_obj_idx(self, obj):
        if obj['objectType'] not in REC_MAX_DIC:
            idx = obj_2_idx_dic[obj['objectType']]
        else:
            idx = obj_2_idx_dic[obj['objectType']][obj['instance_i']]
        return idx

    #---------------------------------------------------------------------------
    # This function is used to update Relationships except 'on'
    def update_except_on(self,objs,i,j,R_on_stored,R_in_stored,R_proximity_stored):
        # Precalculations for later Relationship identification
        idx_ij = [i, j]
        center_ij = [objs[i]['axisAlignedBoundingBox']['center'],
                     objs[j]['axisAlignedBoundingBox']['center']]
        size_ij = [objs[i]['axisAlignedBoundingBox']['size'],
                   objs[j]['axisAlignedBoundingBox']['size']]
        # 'proximity' and 'disjoint' belong to mutual Relationship
        # r_ij point from small to larger obj: chair proximity to table
        # from_i_to_j = True: i 'proximity'/'disjoint' to j
        # from_i_to_j = False: j 'proximity'/'disjoint' to i
        from_i_to_j = (size_ij[0]['x']*size_ij[0]['y']*size_ij[0]['z'] <
                       size_ij[1]['x']*size_ij[1]['y']*size_ij[1]['z'])

        # from_i_to_j = True: smaller_obj = 0, it's object i smaller and been tested against j which is reference in dimension
        # from_i_to_j = False: smaller_obj = 1, it's object j smaller and been tested against i which is reference in dimension
        smaller_obj = int(not from_i_to_j)
        larger_obj = int(from_i_to_j)
        # Exam on Relationship 'in'
        is_in = True
        ref_center = np.array([center_ij[larger_obj]['x'], center_ij[larger_obj]['y'], center_ij[larger_obj]['z']])
        ref_size = np.array([size_ij[larger_obj]['x'], size_ij[larger_obj]['y'], size_ij[larger_obj]['z']])
        # smaller object is in larger object if and only if all corner points of smaller one is in axisAlignedBoundingBox of larger one

        cornerPoints_small = objs[idx_ij[smaller_obj]]['axisAlignedBoundingBox']['cornerPoints']
        # in case one of the bbox is None
        if cornerPoints_small == None:
            cornerPoints_small = objs[idx_ij[smaller_obj]]['objectOrientedBoundingBox']['cornerPoints']

        for point in cornerPoints_small:
            point = np.array(point)
            diff = np.abs(point - ref_center) - ref_size / 2.0
            if np.max(diff) > 0:
                is_in = False
                break

        idx_smaller = self.get_obj_idx(objs[idx_ij[smaller_obj]])
        idx_larger = self.get_obj_idx(objs[idx_ij[larger_obj]])

        if is_in and not R_on_stored:   # priority filter
            self.update_SG(idx_smaller, idx_larger, 'in')
        else:
            '''
            Take the two objects as two balls
            Objects proximity to each other if and only if balls are intercept with each other
            '''
            # Exam the 'proximity' Relationship
            distance_ij = np.linalg.norm(np.array([center_ij[0]['x'], center_ij[0]['y'], center_ij[0]['z']])
                                         - np.array([center_ij[1]['x'], center_ij[1]['y'], center_ij[1]['z']]))
            # Note: z is the forward axis, x is the horizon axis and y is the upward axis
            small_r = np.linalg.norm([size_ij[smaller_obj]['x'], size_ij[smaller_obj]['y'], size_ij[smaller_obj]['z']]) / 2.0
            large_r = np.linalg.norm([size_ij[larger_obj]['x'], size_ij[larger_obj]['y'], size_ij[larger_obj]['z']]) / 2.0

            is_proximity = (distance_ij < (small_r + large_r))

            if is_proximity and not R_in_stored:   # priority filter
                self.update_SG(idx_smaller, idx_larger, 'proximity')
            elif not R_proximity_stored:  # priority filter
                self.update_SG(idx_smaller, idx_larger, 'disjoint')

    #-----------------------------------------------------------------------
    # update the adjancency matrices at entity [i,i] which represents the occurence of object i
    # update the self._obj_vec
    def add_diagonal_entries(self, objs):
        for obj in objs:
            if obj['objectType'] in BAN_TYPE_LIST or obj['objectType'] not in idx_2_obj_list:  # Ignore non-informative objectType e.g. 'Floor' and abnormal obj
                continue
            idx = self.get_obj_idx(obj)
            self._obj_vec[idx] = True
            self._R_on[idx, idx] = True
            self._R_in[idx, idx] = True
            self._R_proximity[idx, idx] = True
            self._R_disjoint[idx, idx] = True
        return

    # Update self._frational_bboxs
    def update_fractional_bboxs(self, objs, image_size):
        for obj in objs:
            if obj['objectType'] in BAN_TYPE_LIST or obj['objectType'] not in idx_2_obj_list:  # Ignore non-informative objectType e.g. 'Floor' and abnormal obj
                continue
            bboxs = obj['2DBoundingBox']
            idx = self.get_obj_idx(obj)
            self._fractional_bboxs[idx] = np.asarray(bboxs)/image_size
        return

    #---------------------------------------------------------------------------
    # Input object data 'event.metadata['objects']'
    def update_from_data(self, objs, image_size=None, visualization_on=False, comfirmed=None):
        #-----------------------------------------------------------------------
        # Group up objects and indexing instances
        objs = group_up(objs)
        #-----------------------------------------------------------------------
        # update the adjancency matrices at entity [i,i] which represents the occurence of object i
        # update the self._obj_vec
        self.add_diagonal_entries(objs)
        #-----------------------------------------------------------------------
        # Update self._frational_bboxs
        if image_size != None:
            self.update_fractional_bboxs(objs, image_size)

        #-----------------------------------------------------------------------
        # comfirm is not none --> this function is used as a client node
        # Loop through current observation and update SG
        for i in range(len(objs)-1):
            if objs[i]['objectType'] in BAN_TYPE_LIST or objs[i]['objectType'] not in idx_2_obj_list:  # Ignore non-informative objectType e.g. 'Floor' and abnormal obj
                continue

            for j in range(i+1, len(objs)):
                #---------------------------------------------------------------
                # 1. Ignore non-informative objectType e.g. 'Floor'
                # 2. Rule out the exceptions of two objects belonging to same Type
                # 3. priority filter in case two object with same objectType have distinct R with another obj:
                #   'on' > 'in' > 'proximity' > 'disjoint'
                if objs[j]['objectType'] in BAN_TYPE_LIST or objs[j]['objectType'] not in idx_2_obj_list:
                    continue
                if (objs[i]['objectType'] not in REC_MAX_DIC and objs[j]['objectType'] not in REC_MAX_DIC
                    and objs[i]['objectType'] == objs[j]['objectType']):
                    continue

                # Get object index
                idx_i = self.get_obj_idx(objs[i])
                idx_j = self.get_obj_idx(objs[j])
                R_on_stored = (self._R_on[idx_i, idx_j] or self._R_on[idx_j, idx_i])
                R_in_stored = (self._R_in[idx_i, idx_j] or self._R_in[idx_j, idx_i])
                R_proximity_stored = (self._R_proximity[idx_i, idx_j] or self._R_proximity[idx_j, idx_i])

                if R_on_stored:
                    continue

                i_on_j = False
                j_on_i = False
                #---------------------------------------------------------------
                # First exam the Receptacle Relationship 'on', high priority defined by the simulation system attributes setting
                if objs[i]['parentReceptacles'] is not None: # This automatically rule out objs[i] in GROUP_UP_LIST
                    # Update normal receptacles
                    if objs[j]['objectType'] not in GROUP_UP_LIST and objs[i]['parentReceptacles'][0] == objs[j]['objectId']:
                        i_on_j = True
                    # Update if objs[j] in GROUP_UP_LIST
                    elif objs[j]['objectType'] in GROUP_UP_LIST:
                        for receptacle in objs[j]['members']:
                            if objs[i]['parentReceptacles'][0] == receptacle['objectId']:
                                i_on_j = True

                if objs[j]['parentReceptacles'] is not None: # This automatically rule out objs[j] in GROUP_UP_LIST
                    # Update normal receptacles
                    if objs[i]['objectType'] not in GROUP_UP_LIST and objs[j]['parentReceptacles'][0] == objs[i]['objectId']:
                        j_on_i = True
                    # Update if objs[i] in GROUP_UP_LIST
                    elif objs[i]['objectType'] in GROUP_UP_LIST:
                        for receptacle in objs[i]['members']:
                            if objs[j]['parentReceptacles'][0] == receptacle['objectId']:
                                j_on_i = True

                # Update SG
                if i_on_j:
                    self.update_SG(idx_i, idx_j, 'on')
                elif j_on_i:
                    self.update_SG(idx_j, idx_i, 'on')
                else:
                    self.update_except_on(objs,i,j,R_on_stored,R_in_stored,R_proximity_stored)

        #-----------------------------------------------------------------------
        # visualize Scene Graph
        if visualization_on:
            self.visualize_SG(comfirmed)
    #---------------------------------------------------------------------------
    # get current scene graph as data
    def get_SG_as_dict(self):
        data = deepcopy({'in':self._R_in, 'on': self._R_on, 'proximity': self._R_proximity, 'disjoint': self._R_disjoint, 'vec': self._obj_vec, 'fractional_bboxs':self._fractional_bboxs})
        return data
