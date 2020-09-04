# Scene graph Module for SSG
from scipy.sparse import lil_matrix, find
from scipy.sparse import find as find_sparse_idx
from termcolor import colored
from lib.params import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import sys, time


class Scene_Graph:
    def __init__(self):
        # All vector and sparse matrix initialized with 'False' boolean
        self._obj_vec = lil_matrix((OBJ_TYPE_NUM, 1), dtype=np.bool)    # binary object occurence vector where index follow 'obj_2_idx_dic.npy'
        self._R_on = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)    # Relationship sparse matrix _R_on[i,j] = True , obj_i on obj_j
        self._R_in = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._R_proximity = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._R_disjoint = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)

    # reset all values in SG instance
    def reset(self):
        # All vector and sparse matrix initialized with 'False' boolean
        self._obj_vec = lil_matrix((OBJ_TYPE_NUM, 1), dtype=np.bool)    # binary object occurence vector where index follow 'obj_2_idx_dic.npy'
        self._R_on = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)    # Relationship sparse matrix _R_on[i,j] = True , obj_i on obj_j
        self._R_in = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._R_proximity = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)
        self._R_disjoint = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool)

    # Update Scene_Graph by object pair (obj_i, obj_j) and their Relationship r_ij.
    # r_ij = 'on', 'in', 'proximity' or 'disjoint'
    # eg: r_ij = 'on' implies obj_i on obj_j
    def update_SG(self, obj_i, obj_j, r_ij):
        self._obj_vec[obj_i, 0] = True
        self._obj_vec[obj_j, 0] = True
        if r_ij == 'on':
            self._R_on[obj_i, obj_j] = True
            self._R_on[obj_j, obj_i] = False     # The Relationship arrow in SG should be directional
            # priority filter in case two object with same objectType have distinct R with another obj:
            # 'on' > 'in' > 'proximity' > 'disjoint'
            # This is also used to rguarantee unique i-j Relationship
            self._R_in[obj_i, obj_j] = False
            self._R_proximity[obj_i, obj_j] = False
            self._R_disjoint[obj_i, obj_j] = False
        elif r_ij == 'in':
            self._R_in[obj_i, obj_j] = True
            self._R_in[obj_j, obj_i] = False     # The Relationship arrow in SG should be directional
            self._R_proximity[obj_i, obj_j] = False # priority filter
            self._R_disjoint[obj_i, obj_j] = False # priority filter

        # 'proximity' and 'disjoint' belong to mutual Relationship
        # r_ij point from small to larger obj: chair proximity to table
        elif r_ij == 'proximity':
            self._R_proximity[obj_i, obj_j] = True
            self._R_proximity[obj_j, obj_i] = False     # The Relationship arrow in SG should be directional
            self._R_disjoint[obj_i, obj_j] = False # priority filter
        elif r_ij == 'disjoint':
            self._R_disjoint[obj_i, obj_j] = True
            self._R_disjoint[obj_j, obj_i] = False     # The Relationship arrow in SG should be directional
        else:
            sys.stderr.write(colored('ERROR: ','red')
                             + "Expect input r_ij = 'on', 'in', 'proximity' or 'disjoint' while get {}\n".format(r_ij))
            sys.exit(1)

    # Visualize Scene Graph
    def visualize_SG(self, comfirmed=None):
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
                edge_labels[(str(i_list[idx]), str(j_list[idx]))] = edge_type[k]

        # Construct Node-Edge graph
        G = nx.DiGraph(directed=True)
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        #fig = plt.figure()
        nx.draw(G, pos, edge_color='black', width=2, linewidths=1,
                node_size=1500, node_color='skyblue', alpha=0.9,
                labels={node:idx_2_obj_list[int(node)] for node in G.nodes()})
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        if comfirmed is None:
            plt.show() # plt.show() is a blocking function...
        else:
            # to continuous changing SG, cannot use blocking function plt.show()
            # plt.savefig('tmp'+str(random.randint(1,100))+'.png')
            # plt.imshow(mpimg.imread("tmp.png"))
            # plt.show(block=False)
            plt.show()
            print(colored('Client: ','green') + 'Receive Data from navigator')
            comfirmed.value = 1

    # Input object data 'event.metadata['objects']'
    def update_from_data(self, objs, visualization_on=False, comfirmed=None):
        # comfirm is not none --> this function is used as a client node
        # Loop through current observation and update SG
        for i in range(len(objs)-1):
            if objs[i]['objectType'] in BAN_TYPE_LIST:  # Ignore non-informative objectType e.g. 'Floor'
                continue
            for j in range(i+1, len(objs)):

                # 1. Ignore non-informative objectType e.g. 'Floor'
                # 2. Rule out the exceptions of two objects belonging to same Type
                # 3. priority filter in case two object with same objectType have distinct R with another obj:
                #   'on' > 'in' > 'proximity' > 'disjoint'
                if objs[j]['objectType'] in BAN_TYPE_LIST or objs[i]['objectType'] == objs[j]['objectType']:
                    continue
                R_on_stored = (self._R_on[obj_2_idx_dic[objs[i]['objectType']],obj_2_idx_dic[objs[j]['objectType']]]
                               or self._R_on[obj_2_idx_dic[objs[j]['objectType']],obj_2_idx_dic[objs[i]['objectType']]])
                R_in_stored = (self._R_in[obj_2_idx_dic[objs[i]['objectType']],obj_2_idx_dic[objs[j]['objectType']]]
                               or self._R_in[obj_2_idx_dic[objs[j]['objectType']],obj_2_idx_dic[objs[i]['objectType']]])
                R_proximity_stored = (self._R_proximity[obj_2_idx_dic[objs[i]['objectType']],obj_2_idx_dic[objs[j]['objectType']]]
                                      or self._R_proximity[obj_2_idx_dic[objs[j]['objectType']],obj_2_idx_dic[objs[i]['objectType']]])
                if R_on_stored:
                    continue

                # First exam the Receptacle Relationship 'on', high priority defined by the simulation system attributes setting
                if objs[i]['parentReceptacles'] is not None and objs[i]['parentReceptacles'][0] == objs[j]['objectId']:
                    self.update_SG(obj_2_idx_dic[objs[i]['objectType']],
                                   obj_2_idx_dic[objs[j]['objectType']], 'on')
                elif objs[j]['parentReceptacles'] is not None and objs[j]['parentReceptacles'][0] == objs[i]['objectId']:
                    self.update_SG(obj_2_idx_dic[objs[j]['objectType']],
                                   obj_2_idx_dic[objs[i]['objectType']], 'on')
                else:
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
                    for point in objs[idx_ij[smaller_obj]]['axisAlignedBoundingBox']['cornerPoints']:
                        point = np.array(point)
                        diff = np.abs(point - ref_center) - ref_size / 2.0
                        if np.max(diff) > 0:
                            is_in = False
                            break

                    if is_in and not R_on_stored:   # priority filter
                        self.update_SG(obj_2_idx_dic[objs[idx_ij[smaller_obj]]['objectType']],
                                       obj_2_idx_dic[objs[idx_ij[larger_obj]]['objectType']], 'in')
                    else:
                    # Exam the 'proximity' Relationship
                        distance_ij = np.linalg.norm(np.array([center_ij[0]['x'], center_ij[0]['y'], center_ij[0]['z']]) -
                                                     np.array([center_ij[1]['x'], center_ij[1]['y'], center_ij[1]['z']]))
                        # Note: z is the forward axis, z is the horizon axis and y is the upward axis
                        is_proximity = (distance_ij < (PROXIMITY_THRESHOLD * np.linalg.norm([size_ij[smaller_obj]['x'],
                                                                                             size_ij[smaller_obj]['y'],
                                                                                             size_ij[smaller_obj]['z']])))
                        if is_proximity and not R_in_stored:   # priority filter
                            self.update_SG(obj_2_idx_dic[objs[idx_ij[smaller_obj]]['objectType']],
                                           obj_2_idx_dic[objs[idx_ij[larger_obj]]['objectType']], 'proximity')
                        elif not R_proximity_stored:  # priority filter
                            self.update_SG(obj_2_idx_dic[objs[idx_ij[smaller_obj]]['objectType']],
                                           obj_2_idx_dic[objs[idx_ij[larger_obj]]['objectType']], 'disjoint')

        # visualize Scene Graph
        if visualization_on:
            self.visualize_SG(comfirmed)
