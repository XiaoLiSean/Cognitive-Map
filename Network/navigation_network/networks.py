'''
Navigation Network, Written by Xiao
For robot localization in a dynamic environment.
'''
import torch
from torch.nn.parameter import Parameter
from torchvision import models, ops
from lib.params import OBJ_TYPE_NUM
from Network.retrieval_network.params import SCENE_ENCODING_VEC_LENGTH
from Network.navigation_network.params import ACTION_CLASSNUM, CHECKPOINTS_DIR
from Network.retrieval_network.networks import RetrievalTriplet, TripletNetImage
import math
import numpy as np

# ------------------------------------------------------------------------------
# -----------------------------navigation_network-------------------------------
# ------------------------------------------------------------------------------
class NavigationNet(torch.nn.Module):
    def __init__(self, only_image_branch=False, self_pretrained_image=True):
        super(NavigationNet, self).__init__()
        if only_image_branch:
            self.naviBackbone = TripletNetImage(enableRoIBridge=False)
        else:
            self.naviBackbone = RetrievalTriplet(self_pretrained_image=False)
            if self_pretrained_image:
                self.RetrievalTriplet.load_self_pretrained_image(CHECKPOINTS_DIR + 'image_best_fit.pkl')

        self.decisionHead = torch.nn.Sequential(
                                        torch.nn.Linear(2*SCENE_ENCODING_VEC_LENGTH, 1024, bias=True),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(1024, 256, bias=True),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(256, ACTION_CLASSNUM, bias=True),
                                        torch.nn.ReLU(inplace=True)
                                                )

    def forward(self, G_img, C_img, G_on, C_on, G_in, C_in, G_prox, C_prox, G_bbox, C_bbox, G_vec, C_vec):

        goal_embedding = self.naviBackbone.get_embedding(G_img, G_on, G_in, G_prox, G_bbox, G_vec)
        current_embedding = self.naviBackbone.get_embedding(C_img, C_on, C_in, C_prox, C_bbox, C_vec)
        concacenated = torch.cat((goal_embedding, current_embedding), dim=1)
        distribution = self.decisionHead(concacenated)

        return distribution
