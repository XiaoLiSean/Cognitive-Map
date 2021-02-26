# This module is used to define siamese networks
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models, ops
from lib.params import OBJ_TYPE_NUM
from Network.retrieval_network.params import BBOX_EMDEDDING_VEC_LENGTH, ROI_EMDEDDING_VEC_LENGTH, WORD_EMDEDDING_VEC_LENGTH, OBJ_FEATURE_VEC_LENGTH
from Network.retrieval_network.params import SG_ENCODING_VEC_LENGTH, IMAGE_ENCODING_VEC_LENGTH, SCENE_ENCODING_VEC_LENGTH
from Network.retrieval_network.params import CHECKPOINTS_DIR, BATCH_SIZE, IMAGE_SIZE, GCN_TIER, DROPOUT_RATE
import math
import numpy as np

# ------------------------------------------------------------------------------
# -----------------------------retrieval_network--------------------------------
# ------------------------------------------------------------------------------
class RetrievalTriplet(torch.nn.Module):
    def __init__(self, GCN_dropout_rate=DROPOUT_RATE, GCN_layers=GCN_TIER, GCN_bias=True, self_pretrained_image=True):
        super(RetrievalTriplet, self).__init__()
        self.ModelName = 'RetrievalTriplet'
        '''
        Default: load pre-trained (on self generated triplets dataset) parameters for image branch,
        then, freeze the image branch (both backbone and head) which means we will rely on pretrained
        visuial features (RoI feature and entire image feature). By this, the backward gradient flow
        from the sg branch through the RoI align to image branch is cut-off.
        '''
        self.image_branch = TripletNetImage(enableRoIBridge=True)
        if self_pretrained_image:
            pretrained_model = torch.load(CHECKPOINTS_DIR + 'image_best_fit.pkl')
            model_dict = self.image_branch.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.image_branch.load_state_dict(model_dict)
            # for parameter in self.image_branch.backbone.parameters():
            #     parameter.requires_grad = False
            # for parameter in self.image_branch.head.parameters():
            #     parameter.requires_grad = False

        self.SG_branch = TripletNetSG(dropout_rate=GCN_dropout_rate, layer_structure=GCN_layers, bias=GCN_bias)
        self.fcn = torch.nn.Sequential(
                                torch.nn.Linear(IMAGE_ENCODING_VEC_LENGTH + SG_ENCODING_VEC_LENGTH, 2048, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(2048, SCENE_ENCODING_VEC_LENGTH, bias=True),
                                torch.nn.ReLU(inplace=True)
                                      )

    def forward(self, A_img, P_img, N_img, A_on, P_on, N_on,
                A_in, P_in, N_in, A_prox, P_prox, N_prox,
                A_bbox, P_bbox, N_bbox, A_vec, P_vec, N_vec):

        anchor = self.get_embedding(A_img, A_on, A_in, A_prox, A_bbox, A_vec)
        positive = self.get_embedding(P_img, P_on, P_in, P_prox, P_bbox, P_vec)
        negative = self.get_embedding(N_img, N_on, N_in, N_prox, N_bbox, N_vec)

        return anchor, positive, negative

    def get_embedding(self, img, R_on, R_in, R_prox, bbox, obj_vec):
        img_embedding, X = self.image_branch.get_embedding(img, bbox, obj_vec)
        sg_embedding = self.SG_branch.get_embedding(R_on, R_in, R_prox, X)

        concacenated = torch.cat((img_embedding, sg_embedding), dim=1)
        embedding = self.fcn(concacenated)

        return embedding

# ------------------------------------------------------------------------------
# --------------------------------RoI Bridge------------------------------------
# ------------------------------------------------------------------------------
# Import vector embeding related parameters
from lib.params import idx_2_obj_list, THOR_2_VEC

# size of OBJ_TYPE_NUM * WORD_EMDEDDING_VEC_LENGTH = (256,300)
def get_glove_matrix():
    features = []
    # Normalize feature vectors to unit vector
    for obj_name in idx_2_obj_list:
        features.append(np.true_divide(THOR_2_VEC[obj_name], np.linalg.norm(THOR_2_VEC[obj_name])))

    return torch.FloatTensor(np.asarray(features))

def denseTensor_to_SparseTensor(denseTensor):
    indices = torch.nonzero(denseTensor, as_tuple=False).t()
    if indices.shape[0] == 3:
        values = denseTensor[indices[0], indices[1], indices[2]] # modify this based on dimensionality
    elif indices.shape[0] == 2:
        values = denseTensor[indices[0], indices[1]]
    elif indices.shape[0] == 1:
        values = denseTensor[indices[0]]
    else:
        print("Tensor dim should be smaller than 4")
    sparseTensor = torch.sparse.FloatTensor(indices, values, denseTensor.size())
    return sparseTensor

# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(5000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position=IMAGE_SIZE+1, d_model=int(BBOX_EMDEDDING_VEC_LENGTH/4)):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding[0]


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class RoIBridge(torch.nn.Module):
    def __init__(self):
        super(RoIBridge, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Used to embed (roi bounding box + roi visual feature + glove word) as one object feature vector
        # Note: this layer will automatically initialize the weights and bias using uniform_(-stdv, stdv)
        self.feature_fcn = torch.nn.Sequential(
                                    torch.nn.Linear(BBOX_EMDEDDING_VEC_LENGTH+ROI_EMDEDDING_VEC_LENGTH+WORD_EMDEDDING_VEC_LENGTH, 1024, bias=True),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(1024, OBJ_FEATURE_VEC_LENGTH, bias=True),
                                    torch.nn.ReLU(inplace=True)
                                              )
        # size of OBJ_TYPE_NUM * WORD_EMDEDDING_VEC_LENGTH = (256,300)
        self.word_embedding_matrix = get_glove_matrix()
        # size of (IMAGE_SIZE+1) * (BBOX_EMDEDDING_VEC_LENGTH/4) = (225,64) and PE(i) = self.position_embedding_matrix[i,:]
        self.position_embedding_matrix = positional_encoding()
    # --------------------------------------------------------------------------
    '''
    Feature Matrix = [RoI + Position + Word] concacenated
    get_NonEmpty_RoI_Conv_features: Pass convolutional features of size (BATCH_NUM, 1024, 14, 14) of batch images
        batch_obj_vecs of size (BATCH_NUM, OBJ_TYPE_NUM) which is vector of the occurence of objects
        and batch_fractional_bboxs of [BATCH_NUM, OBJ_TYPE_NUM, 4] where OBJ_TYPE_NUM = 256,
        Get roi conv features of size [K, 1024, 14, 14] where K < BATCH_NUM*OBJ_TYPE_NUM
    fill_Empty_RoI_features: the conv features of size [K, 1024, 14, 14] passed through ResNet head
        get NonEmpty_RoI_Visual_features = [K, 2048] as RoI visual_features
        this function fill RoI visual_features back to [BATCH_NUM*OBJ_TYPE_NUM, 2048] for torch.cat
    get_PoE_features: input batch_fractional_bboxs of [BATCH_NUM, OBJ_TYPE_NUM, 4] and batch_obj_vecs,
        output PoE features of size [BATCH_NUM*OBJ_TYPE_NUM, 64*4];
    get_GlV_features: output repeated GloVe features of size [BATCH_NUM*OBJ_TYPE_NUM, 300];
        since it's a constant feature matrix it would be better to pre-generate and store in Class
    '''
    # --------------------------------------------------------------------------
    # Note embedding information of empty boundng box of [0,0,0,0] is zero vectors thus do no process here
    def get_NonEmpty_RoI_Conv_features(self, batch_conv_features, batch_fractional_bboxs, batch_obj_vecs):

        tensor_size = batch_conv_features.shape[-1]
        # Flatten and filter (get non-empty) the batch data to process RoI to avoid for-loop operation
        # batch_obj_vecs can't be empty since we take that as filter while collecting data
        batch_obj_vecs = torch.flatten(batch_obj_vecs, start_dim=0)
        batch_fractional_bboxs = torch.flatten(batch_fractional_bboxs, start_dim=0, end_dim=1)
        # store non empty RoI whose Bbox is not [0,0,0,0]
        fractional_bboxs_list = [batch_fractional_bboxs[batch_obj_vecs==1]*tensor_size]
        # RoI feature operation
        RoI_Conv_features = ops.roi_align(batch_conv_features, fractional_bboxs_list, output_size=(tensor_size,tensor_size))

        return RoI_Conv_features

    # Fill in zero vector for empty boundng box of [0,0,0,0]
    def fill_Empty_RoI_features(self, NonEmpty_RoI_Vec_features, batch_obj_vecs):
        local_batch_size = batch_obj_vecs.shape[0]
        # Pend the truncated RoI feature to full size visual_features
        Visual_features = torch.zeros(local_batch_size*OBJ_TYPE_NUM, IMAGE_ENCODING_VEC_LENGTH).to(self.device)
        batch_obj_vecs = torch.flatten(batch_obj_vecs, start_dim=0)
        addin_index = torch.squeeze(torch.nonzero(batch_obj_vecs, as_tuple=False))
        Visual_features.index_add_(0, addin_index, NonEmpty_RoI_Vec_features)

        return denseTensor_to_SparseTensor(Visual_features)

    '''
    Functions need to avoid for-loop (this one and above one)
    which will cause intense communication between GPU and CPu
    '''
    '''well i will try it in the future... for get_PoE_features'''
    def get_PoE_features(self, batch_fractional_bboxs, batch_obj_vecs):
        local_batch_size = batch_fractional_bboxs.shape[0]
        PoE_features = torch.zeros(local_batch_size*OBJ_TYPE_NUM, BBOX_EMDEDDING_VEC_LENGTH)
        for i in range(local_batch_size):
            for j in range(OBJ_TYPE_NUM):
                idx = j + OBJ_TYPE_NUM*i
                if batch_obj_vecs[i][j] == 1:
                    bbox_embedding = [self.position_embedding_matrix[int(torch.clamp(frac*IMAGE_SIZE, 0, IMAGE_SIZE)),:] for frac in batch_fractional_bboxs[i][j]]
                    PoE_features[idx,:] = torch.flatten(torch.FloatTensor(bbox_embedding))
                else:
                    pass
        PoE_features = PoE_features.to(self.device)
        return denseTensor_to_SparseTensor(PoE_features)

    def get_GlV_features(self, local_batch_size):
        GlV_features = self.word_embedding_matrix.repeat(local_batch_size, 1)
        GlV_features = denseTensor_to_SparseTensor(torch.Tensor(GlV_features)).to(self.device)
        return GlV_features

# ------------------------------------------------------------------------------
# -------------------------------Image Branch-----------------------------------
# ------------------------------------------------------------------------------
class TripletNetImage(torch.nn.Module):
    def __init__(self, enableRoIBridge=False):
        super(TripletNetImage, self).__init__()
        self.ModelName = 'TripletNetImage'
        # Initialize weight using ImageNet pretrained weights
        model = models.resnet50(pretrained=True)
        # The ResNet50-C4 Backbone
        self.backbone = torch.nn.Sequential(*(list(model.children())[:-3]))
        # ResNet Stage 5 except from the last linear classifier
        self.head = torch.nn.Sequential(*(list(model.children())[-3:-1]))
        self.enableRoIBridge = enableRoIBridge
        if self.enableRoIBridge:
            self.RoIBridge = RoIBridge()
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    '''
    Pass the batch images (BATCH_NUM,3,224,224) into the ResNet50-C4 backbone
    Get convolutional features of size (BATCH_NUM,1024,14,14)
    '''
    # --------------------------------------------------------------------------
    def get_conv_features(self, batch_imgs):
        return self.backbone(batch_imgs)
    # --------------------------------------------------------------------------
    '''
    Pass the batch convolutional features (BATCH_NUM,1024,14,14) into the ResNet50 head
    Get vector features of size (BATCH_NUM, 2048)
    '''
    # --------------------------------------------------------------------------
    def conv_to_vec_feature(self, conv_features):
        vec_feature = torch.squeeze(torch.squeeze(self.head(conv_features), dim=2), dim=2)
        return vec_feature
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    '''Prepare batch feature matrices for GCN as Batch Inputs + Single Img Vector Embedding'''
    # --------------------------------------------------------------------------
    def get_RoI_features(self, batch_conv_features, batch_fractional_bboxs, batch_obj_vecs):
        NonEmpty_RoI_Conv_features = self.RoIBridge.get_NonEmpty_RoI_Conv_features(batch_conv_features, batch_fractional_bboxs, batch_obj_vecs)
        NonEmpty_RoI_Vec_features = self.conv_to_vec_feature(NonEmpty_RoI_Conv_features)
        RoI_features = self.RoIBridge.fill_Empty_RoI_features(NonEmpty_RoI_Vec_features, batch_obj_vecs)
        return RoI_features

    def get_embedding(self, batch_imgs, batch_fractional_bboxs=None, batch_obj_vecs=None):

        batch_conv_features = self.get_conv_features(batch_imgs)
        batch_img_vector_embeddings = self.conv_to_vec_feature(batch_conv_features)

        if self.enableRoIBridge:
            batch_obj_vecs = torch.squeeze(batch_obj_vecs)

            RoI_features = self.get_RoI_features(batch_conv_features, batch_fractional_bboxs, batch_obj_vecs)
            PoE_features = self.RoIBridge.get_PoE_features(batch_fractional_bboxs, batch_obj_vecs)
            GlV_features = self.RoIBridge.get_GlV_features(batch_imgs.shape[0])

            # Detach to double ensure no gradient propagates back to ResNet and RoI except for the feature_fcn
            raw_features = torch.cat((RoI_features, PoE_features, GlV_features), dim=1).detach().clone()
            batch_feature_matrices = self.RoIBridge.feature_fcn(raw_features)
            batch_feature_matrices = torch.stack(torch.split(batch_feature_matrices, OBJ_TYPE_NUM, dim=0), dim=0)

            return batch_img_vector_embeddings, batch_feature_matrices
        else:
            return batch_img_vector_embeddings

    # --------------------------------------------------------------------------
    def forward(self, A_imgs, P_imgs, N_imgs,
                A_fractional_bboxs=None, P_fractional_bboxs=None, N_fractional_bboxs=None,
                A_vecs=None, P_vecs=None, N_vecs=None):

        if self.enableRoIBridge:
            A_vec, A_X = self.get_embedding(A_imgs, A_fractional_bboxs, A_vecs)
            P_vec, P_X = self.get_embedding(P_imgs, P_fractional_bboxs, P_vecs)
            N_vec, N_X = self.get_embedding(N_imgs, N_fractional_bboxs, N_vecs)
            return A_vec, N_vec, P_vec, A_X, P_X, N_X
        else:
            A_vec = self.get_embedding(A_imgs)
            P_vec = self.get_embedding(P_imgs)
            N_vec = self.get_embedding(N_imgs)
            return A_vec, N_vec, P_vec


# ------------------------------------------------PoE_features------------------------------
# ----------------------------Scene Graph Branch--------------------------------
# ------------------------------------------------------------------------------

class TripletNetSG(torch.nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE, layer_structure=GCN_TIER, bias=True):
        super(TripletNetSG, self).__init__()
        self.layer_structure = layer_structure
        self.num_layers = len(layer_structure) - 1
        self.dropout_rate = dropout_rate

        # Build up three pathway gcns
        self.gcn_on = torch.nn.ModuleList()
        self.gcn_in = torch.nn.ModuleList()
        self.gcn_prox = torch.nn.ModuleList()

        for i in range(self.num_layers):
            in_features = self.layer_structure[i]
            out_features = self.layer_structure[i+1]
            # Three braches of GCN for three object relationships
            self.gcn_on.append(GraphConvolution(in_features, out_features, bias=bias))
            self.gcn_in.append(GraphConvolution(in_features, out_features, bias=bias))
            self.gcn_prox.append(GraphConvolution(in_features, out_features, bias=bias))

        self.gcn_paths = torch.nn.ModuleList()
        self.gcn_paths.append(self.gcn_on)
        self.gcn_paths.append(self.gcn_in)
        self.gcn_paths.append(self.gcn_prox)

        self.fcn = torch.nn.Sequential(
                                torch.nn.Linear(OBJ_TYPE_NUM * self.layer_structure[-1] * 3, 4096, bias=bias),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(4096, 2048, bias=bias),
                                torch.nn.ReLU(inplace=True)
                                      )

    def forward(self, A_on, P_on, N_on, A_in, P_in, N_in,
                A_prox, P_prox, N_prox, A_X, P_X, N_X):

        anchor = self.get_embedding(A_on, A_in, A_prox, A_X)
        positive = self.get_embedding(P_on, P_in, P_prox, P_X)
        negative = self.get_embedding(N_on, N_in, N_prox, N_X)

        return anchor, positive, negative

    # Path = [on, in, prox] --> [0,1,2]
    def graph_convolution_pathway(self, X, R, path=None):
        if path != None:
            for i in range(self.num_layers):
                X = self.gcn_paths[path][i](X, R)
                X = F.relu(X, inplace=True)
                X = F.dropout(X, self.dropout_rate, training=self.training)
            return X
        else:
            print('Please specify the GCN path')
            exit(0)

    def get_embedding(self, R_on, R_in, R_prox, X):

        R_on = denseTensor_to_SparseTensor(R_on)
        R_in = denseTensor_to_SparseTensor(R_in)
        R_prox = denseTensor_to_SparseTensor(R_prox)

        X_on = self.graph_convolution_pathway(X, R_on, path=0)
        X_in = self.graph_convolution_pathway(X, R_in, path=1)
        X_prox = self.graph_convolution_pathway(X, R_prox, path=2)
        # concacenated as [X_on, X_in, X_prox]
        concacenated = torch.flatten(torch.cat((X_on, X_in, X_prox), dim=2), start_dim=1)
        embedding = self.fcn(concacenated)

        return embedding

# ------------------------------------------------------------------------------
# 3rd party code from git@github.com:tkipf/pygcn.git
# ------------------------------------------------------------------------------
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    '''
    Taking batch sparse matrice A and Weight matrix W
    A: [BATCH_SIZE, x, y]
    W: [y, z]
    Output: [BATCH_SIZE, x, z]
    '''
    def sparse_bmm(self, batch_A, W):
        local_batch_size = batch_A.shape[0]
        output = [torch.spmm(batch_A[i], W) for i in range(local_batch_size)]
        output = torch.stack(output)
        return output

    '''
    Taking batch sparse matrice A and batch dense matrix S
    A: [BATCH_SIZE, x, y]
    S: [BATCH_SIZE, y, z]
    Output: [BATCH_SIZE, x, z]
    '''
    def sparse_bbmm(self, batch_A, batch_S):
        local_batch_size = batch_A.shape[0]
        output = [torch.spmm(batch_A[i], batch_S[i]) for i in range(local_batch_size)]
        output = torch.stack(output)
        return output

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = self.sparse_bbmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
