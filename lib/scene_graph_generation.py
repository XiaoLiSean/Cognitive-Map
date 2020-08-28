# Scene graph Module for SSG
from scipy.sparse import lil_matrix, find
from scipy.sparse import find as find_sparse_idx
from os.path import dirname, abspath
from termcolor import colored
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

INFO_FILE_PATH = dirname(dirname(abspath(__file__))) + '/AI2THOR_info' # File path for info of iTHOR Env.
obj_2_idx_dic = np.load(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', allow_pickle='TRUE').item()
idx_2_obj_list = np.load(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy')
OBJ_TYPE_NUM = len(idx_2_obj_list) # Maximum numbers of objectType in iTHOR Env.
PROXIMITY_THRESHOLD = 3 # distance ratio threshold for proximity determination

class Scene_Graph:
    def __init__(self):
        # All vector and sparse matrix initialized with 'False' boolean
        self._obj_vec = lil_matrix((OBJ_TYPE_NUM, 1), dtype=np.bool) # binary object occurence vector where index follow 'obj_2_idx_dic.npy'
        self._R_on = lil_matrix((OBJ_TYPE_NUM, OBJ_TYPE_NUM), dtype=np.bool) # Relationship sparse matrix _R_on[i,j] = True , obj_i on obj_j
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
        elif r_ij == 'in':
            self._R_in[obj_i, obj_j] = True
        # 'proximity' and 'disjoint' belong to mutual Relationship
        # r_ij point from small to larger obj: chair proximity to table
        elif r_ij == 'proximity':
            self._R_proximity[obj_i, obj_j] = True
        elif r_ij == 'disjoint':
            self._R_disjoint[obj_i, obj_j] = True
        else:
            sys.stderr.write(colored('ERROR: ','red')
                             + "Expect input r_ij = 'on', 'in', 'proximity' or 'disjoint' while get {}\n".format(r_ij))
            sys.exit(1)

    # Visualize Scene Graph
    def visualize_SG(self):
        node_list = find_sparse_idx(self._obj_vec)[0] # objectType/Node index list
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
        plt.figure()
        nx.draw(G, pos, edge_color='black', width=2, linewidths=1,
                node_size=1500, node_color='skyblue', alpha=0.9,
                labels={node:idx_2_obj_list[int(node)] for node in G.nodes()})
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.axis('off')
        plt.show()

    # Input object data 'event.metadata['objects']'
    def update_from_data(self, objs, visualization_on = False):
        # Loop through current observation and update SG
        for i in range(len(objs)-1):
            for j in range(i+1, len(objs)):
                # First exam the Receptacle Relationship 'on'
                if objs[i]['parentReceptacles'] is not None and objs[i]['parentReceptacles'][0] == objs[j]['objectId']:
                    self.update_SG(obj_2_idx_dic[objs[i]['objectType']],
                                   obj_2_idx_dic[objs[j]['objectType']], 'on')
                elif objs[j]['parentReceptacles'] is not None and objs[j]['parentReceptacles'][0] == objs[i]['objectId']:
                    self.update_SG(obj_2_idx_dic[objs[j]['objectType']],
                                   obj_2_idx_dic[objs[i]['objectType']], 'on')
                else:
                # Exam the 'proximity' Relationship
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
                    distance_ij = np.linalg.norm(np.array([center_ij[0]['x'], center_ij[0]['y'], center_ij[0]['z']]) -
                                                 np.array([center_ij[1]['x'], center_ij[1]['y'], center_ij[1]['z']]))
                    is_proximity = (distance_ij < PROXIMITY_THRESHOLD*max([size_ij[int(not from_i_to_j)]['x'], size_ij[int(not from_i_to_j)]['y']]))
                    if is_proximity:
                        self.update_SG(obj_2_idx_dic[objs[idx_ij[int(not from_i_to_j)]]['objectType']],
                                       obj_2_idx_dic[objs[idx_ij[int(from_i_to_j)]]['objectType']], 'proximity')
                    else:
                        self.update_SG(obj_2_idx_dic[objs[idx_ij[int(not from_i_to_j)]]['objectType']],
                                       obj_2_idx_dic[objs[idx_ij[int(from_i_to_j)]]['objectType']], 'disjoint')

        # visualize Scene Graph
        if visualization_on:
            self.visualize_SG()
