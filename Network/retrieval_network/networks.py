# This module is used to define siamese networks
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models
from Network.retrieval_network.params import SG_ENCODING_VEC_LENGTH, IMAGE_ENCODING_VEC_LENGTH, SCENE_ENCODING_VEC_LENGTH, CHECKPOINTS_DIR
import math
import numpy as np

# ------------------------------------------------------------------------------
# -----------------------------retrieval_network--------------------------------
# ------------------------------------------------------------------------------
class RetrievalTriplet(torch.nn.Module):
    def __init__(self, GCN_layers=3, GCN_bias=True, pretrained_image=True):
        super(RetrievalTriplet, self).__init__()
        self.image_branch = TripletNetImage()

        if pretrained_image:
            self.image_branch.load_state_dict(torch.load(CHECKPOINTS_DIR + 'image_best_fit.pkl'))

        self.SG_branch = TripletNetSG(num_layers=GCN_layers, bias=GCN_bias)

        self.fcn = torch.nn.Sequential(
                                torch.nn.Linear(IMAGE_ENCODING_VEC_LENGTH + SG_ENCODING_VEC_LENGTH, 2048),
                                torch.nn.ReLU(),
                                torch.nn.Linear(2048, SCENE_ENCODING_VEC_LENGTH),
                                torch.nn.ReLU()
                                      )

    def forward(self, anchor_img, positive_img, negative_img,
                anchor_on, positive_on, negative_on,
                anchor_in, positive_in, negative_in,
                anchor_prox, positive_prox, negative_prox):

        anchor = self.get_embedding(anchor_img, anchor_on, anchor_in, anchor_prox)
        positive = self.get_embedding(positive_img, positive_on, positive_in, positive_prox)
        negative = self.get_embedding(negative_img, negative_on, negative_in, negative_prox)

        return anchor, positive, negative

    def get_embedding(self, img, A_on, A_in, A_prox, eval=False):
        img_embedding = self.image_branch.get_embedding(img)
        sg_embedding = self.SG_branch.get_embedding(A_on, A_in, A_prox)
        embedding = torch.cat((img_embedding, sg_embedding), 1)

        if eval:
            concacenated = torch.cat((img_embedding, sg_embedding), 0)
        else:
            concacenated = torch.cat((img_embedding, sg_embedding), 1)

        embedding = self.fcn(concacenated)

        return embedding

# ------------------------------------------------------------------------------
# -------------------------------Image Branch-----------------------------------
# ------------------------------------------------------------------------------
class TripletNetImage(torch.nn.Module):
    def __init__(self):
        super(TripletNetImage, self).__init__()
        model = models.resnet50(pretrained=True)
        # The ResNet50-C4 Backbone
        self.backbone = torch.nn.Sequential(*(list(model.children())[:-3]))
        # ResNet Stage 5
        self.head = torch.nn.Sequential(*(list(model.children())[-3:]))

    def forward(self, anchor_img, positive_img, negative_img):
        anchor = self.get_embedding(anchor_img)
        positive = self.get_embedding(positive_img)
        negative = self.get_embedding(negative_img)
        return anchor, positive, negative

    def get_embedding(self, img):
        return torch.squeeze(self.embedding(img))

# ------------------------------------------------------------------------------
# ----------------------------Scene Graph Branch--------------------------------
# ------------------------------------------------------------------------------
# Import vector embeding related parameters
from lib.params import idx_2_obj_list, THOR_2_VEC
# ------------------------------------------------------------------------------
class TripletNetSG(torch.nn.Module):
    def __init__(self, num_layers=3, bias=True):
        super(TripletNetSG, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = num_layers
        self.feature_matrix = self.get_features_matrix()
        vector_d = list(self.feature_matrix.size())[1]
        node_num = list(self.feature_matrix.size())[0]

        self.GCNs_on = torch.nn.ModuleList()
        self.GCNs_in = torch.nn.ModuleList()
        self.GCNs_prox = torch.nn.ModuleList()
        self.fcn = torch.nn.Sequential(
                                torch.nn.Linear(node_num * 3, SG_ENCODING_VEC_LENGTH),
                                torch.nn.ReLU()
                                      )

        for i in range(self.layers):
            in_features = vector_d
            out_features = vector_d
            # Case Last layer: output scala for each node
            if i == self.layers - 1:
                out_features = 1
            # Three braches of GCN for three object relationships
            self.GCNs_on.append(GraphConvolution(in_features, out_features, bias=bias))
            self.GCNs_in.append(GraphConvolution(in_features, out_features, bias=bias))
            self.GCNs_prox.append(GraphConvolution(in_features, out_features, bias=bias))

    def get_features_matrix(self):
        features = []
        # Normalize feature vectors to unit vector
        for obj_name in idx_2_obj_list:
            features.append(np.true_divide(THOR_2_VEC[obj_name], np.linalg.norm(THOR_2_VEC[obj_name])))

        return torch.FloatTensor(np.asarray(features))

    def forward(self, anchor_on, positive_on, negative_on, anchor_in, positive_in, negative_in, anchor_prox, positive_prox, negative_prox):
        anchor = self.get_embedding(anchor_on, anchor_in, anchor_prox)
        positive = self.get_embedding(positive_on, positive_in, positive_prox)
        negative = self.get_embedding(negative_on, negative_in, negative_prox)

        return anchor, positive, negative

    def get_embedding(self, A_on, A_in, A_prox):
        X_on = self.feature_matrix.clone().detach().to(self.device)
        X_in = self.feature_matrix.clone().detach().to(self.device)
        X_prox = self.feature_matrix.clone().detach().to(self.device)
        for i in range(self.layers):
            X_on = F.relu(self.GCNs_on[i](X_on, A_on))
            X_in = F.relu(self.GCNs_in[i](X_in, A_in))
            X_prox = F.relu(self.GCNs_prox[i](X_prox, A_prox))

        concacenated = torch.squeeze(torch.cat((X_on, X_in, X_prox), 1))
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

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
