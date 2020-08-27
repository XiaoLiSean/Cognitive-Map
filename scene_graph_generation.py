# Scene graph Module for SSG and store object-level info for
from scipy.sparse import lil_matrix, find
from scipy.sparse import find as find_sparse_idx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xlrd
import sys

OBJ_TYPE_NUM = 125 # Maximum numbers of objectType in iTHOR Env.
INFO_FILE_PATH = './AI2THOR_info' # objectType list of iTHOR Env.
PROXIMITY_THRESHOLD = 3 # distance ratio threshold for proximity determination

# Function for construct dictionary of d[objectType] = index
# The index is listed in ai2thor.allenai.org/ithor/documentation/objects/actionable-properties/#table-of-object-actionable-properties
def load_obj_idx_excel():
    FILE_NAME = 'Object Type.xlsx'
    wb = xlrd.open_workbook(INFO_FILE_PATH + '/' + FILE_NAME)
    sh = wb.sheet_by_index(0)
    obj_2_idx_dic = {}
    idx_2_obj_list = []
    for i in range(OBJ_TYPE_NUM):
        obj_2_idx_dic[sh.cell(i,0).value] = i
        idx_2_obj_list.append(sh.cell(i,0).value)
    np.save(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', obj_2_idx_dic) # Save dictionary as .npy
    np.save(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy', idx_2_obj_list) # Save list as .npy


obj_2_idx_dic = np.load(INFO_FILE_PATH + '/' + 'obj_2_idx_dic.npy', allow_pickle='TRUE').item()
idx_2_obj_list = np.load(INFO_FILE_PATH + '/' + 'idx_2_obj_list.npy')

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
            sys.stderr.write("Expect input r_ij = 'on', 'in', 'proximity' or 'disjoint' while get ", r_ij)
            sys.exit(1)

    # Visualize Scene Graph
    def visualize_SG(self):
        node_list = find_sparse_idx(self._obj_vec)[0] # objectType/Node index list
        edges = []
        edge_labels = {}

        # append edges and edge_labels for r_ij = 'on', 'in', 'proximity' & 'disjoint'
        # eg: r_ij = 'on' implies obj_i on obj_j
        edge_data = [self._R_on, self._R_in, self._R_proximity, self._R_disjoint]
        edge_type = ['on', 'in', 'proximity', 'disjoint']
        for k in range(len(edge_data)-1): # Exclude 'disjoint' for visualization
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


if __name__ == '__main__':
    # Reload the excel info into 'obj_2_idx_dic.npy' and 'idx_2_obj_list.npy'
    print("Reloading AI2THOR objectType excel info into 'obj_2_idx_dic.npy' and 'idx_2_obj_list.npy'")
    load_obj_idx_excel()
    print('Done')
