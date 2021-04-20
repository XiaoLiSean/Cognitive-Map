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

    def forward(self, A_img, B_img, A_on, B_on, A_in, B_in, A_prox, B_prox, A_bbox, B_bbox, A_vec, B_vec):

        current_embedding = self.naviBackbone.get_embedding(A_img, A_on, A_in, A_prox, A_bbox, A_vec)
        goal_embedding = self.naviBackbone.get_embedding(B_img, B_on, B_in, B_prox, B_bbox, B_vec)
        concacenated = torch.cat((current_embedding, goal_embedding), dim=1)
        distribution = self.decisionHead(concacenated)

        return distribution
