'''
Navigation Network, Written by Xiao
For robot localization in a dynamic environment.
'''
import torch
from torch.nn.parameter import Parameter
from torchvision import models, ops
from lib.params import OBJ_TYPE_NUM
from Network.retrieval_network.params import SCENE_ENCODING_VEC_LENGTH, IMAGE_ENCODING_VEC_LENGTH
from Network.navigation_network.params import ACTION_CLASSNUM, CHECKPOINTS_DIR
from Network.retrieval_network.networks import RetrievalTriplet, TripletNetImage
import math
import numpy as np

# ------------------------------------------------------------------------------
# -----------------------------navigation_network-------------------------------
# ------------------------------------------------------------------------------
class NavigationNet(torch.nn.Module):
    def __init__(self, only_image_branch=False):
        super(NavigationNet, self).__init__()
        self.only_image_branch = only_image_branch
        if self.only_image_branch:
            self.naviBackbone = TripletNetImage(enableRoIBridge=False, pretrainedResNet=True)
            feature_embedding_len = IMAGE_ENCODING_VEC_LENGTH
        else:
            self.naviBackbone = RetrievalTriplet(self_pretrained_image=False, pretrainedResNet=True)
            feature_embedding_len = SCENE_ENCODING_VEC_LENGTH

        self.decisionHead = torch.nn.Sequential(
                                        torch.nn.Linear(2*feature_embedding_len, 1024, bias=True),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(1024, 256, bias=True),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(256, 64, bias=True),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(64, 32, bias=True),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(32, ACTION_CLASSNUM, bias=True),
                                        torch.nn.Softmax(dim=1)
                                                )

    def forward(self, A_img, B_img, A_on=None, B_on=None, A_in=None, B_in=None, A_prox=None, B_prox=None, A_bbox=None, B_bbox=None, A_vec=None, B_vec=None):

        if self.only_image_branch:
            current_embedding = self.naviBackbone.get_embedding(A_img)
            goal_embedding = self.naviBackbone.get_embedding(B_img)
        else:
            current_embedding = self.naviBackbone.get_embedding(A_img, A_on, A_in, A_prox, A_bbox, A_vec)
            goal_embedding = self.naviBackbone.get_embedding(B_img, B_on, B_in, B_prox, B_bbox, B_vec)

        concacenated = torch.cat((current_embedding, goal_embedding), dim=1)
        distribution = self.decisionHead(concacenated)


        return distribution
