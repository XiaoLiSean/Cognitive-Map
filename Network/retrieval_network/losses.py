import torch
import torch.nn.functional as F
from termcolor import colored
from Network.retrieval_network.params import *
from Network.retrieval_network.datasets import get_pose_from_name
import sys
from os.path import dirname, abspath
root_folder = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_folder)
from lib.similarity import view_similarity

# ------------------------------------------------------------------------------
# This function is triplet loss with variable margin alpha defined by view cone overlaps
class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    def __init__(self, constant_margin=True, margin=0.0):
        super(TripletLoss, self).__init__()
        self.constant_margin = constant_margin
        self.margin = margin

    def forward(self, anchors, positives, negatives, alphas, device, batch_average_loss=True):

        if self.constant_margin:
            alphas = self.margin*torch.ones(BATCH_SIZE)
            alphas.to(device)
            # Using ReLU instead of max since max is not differentiable
            losses = F.relu(COS(anchors, negatives) - COS(anchors, positives) + alphas)
        else:
            # Using ReLU instead of max since max is not differentiable
            losses = F.relu(COS(anchors, negatives) - COS(anchors, positives) + alphas[0] - alphas[1])
        corrects = (COS(anchors, negatives) < COS(anchors, positives))

        return losses.mean() if batch_average_loss else losses.sum(), torch.sum(corrects.int())
