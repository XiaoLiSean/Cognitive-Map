# This module is used to define siamese networks
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models
import math
import numpy as np

# ------------------------------------------------------------------------------
class SiameseNetImage(torch.nn.Module):
    def __init__(self):
        super(SiameseNetImage, self).__init__()
        model = models.resnet50(pretrained=True)
        # Strip final fc layer: self.embedding output 2048d
        self.embedding = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, img1, img2):
        embedding1 = self.embedding(img1)
        embedding2 = self.embedding(img2)
        return embedding1, embedding2

    def get_embedding(self, img):
        return self.embedding(img)
# ------------------------------------------------------------------------------
class TripletNetImage(torch.nn.Module):
    def __init__(self):
        super(TripletNetImage, self).__init__()
        model = models.resnet50(pretrained=True)
        # Strip final fc layer: self.embedding output 512d
        self.embedding = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, anchor_img, positive_img, negative_img):
        anchor = self.embedding(anchor_img)
        positive = self.embedding(positive_img)
        negative = self.embedding(negative_img)
        return anchor, positive, negative

    def get_embedding(self, img):
        return self.embedding(img)

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
        self.GCNs_on = torch.nn.ModuleList()
        self.GCNs_in = torch.nn.ModuleList()
        self.GCNs_prox = torch.nn.ModuleList()
        for i in range(self.layers):
            self.GCNs_on.append(GraphConvolution(vector_d, vector_d, bias=True))
            self.GCNs_in.append(GraphConvolution(vector_d, vector_d, bias=True))
            self.GCNs_prox.append(GraphConvolution(vector_d, vector_d, bias=True))

    def get_features_matrix(self):
        features = []
        # Normalize feature vectors
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

        X_on = torch.flatten(X_on, start_dim=1)
        X_in = torch.flatten(X_in, start_dim=1)
        X_prox = torch.flatten(X_prox, start_dim=1)
        return torch.cat((X_on, X_in, X_prox), 1)
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
