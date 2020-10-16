import torch
import torch.nn.functional as F
from termcolor import colored
from params import *
from datasets import get_pose_from_name
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

    def forward(self, anchors, positives, negatives, img_names, batch_average_loss=True):
        # Set margins alpha
        alphas = self.margin*torch.ones(BATCH_SIZE)
        if not self.constant_margin:
            for idx in range(len(img_names[0])):
                anchor_pose = get_pose_from_name(img_names[0][idx])
                positive_pose = get_pose_from_name(img_names[1][idx])
                negative_pose = get_pose_from_name(img_names[2][idx])
                alphas[idx] = view_similarity(anchor_pose, positive_pose) - view_similarity(anchor_pose, negative_pose) # margin
                if alphas[idx] <= 0:
                    print('----'*10 + '\n' + colored('Data Error: ','red') + 'triplet({},{},{}) have negative margin'.format(img_names[0][idx], img_names[1][idx], img_names[2][idx]))
                    exit(1)
        # Using ReLU instead of max since max is not differentiable
        losses = F.relu(COS(anchors, negatives) - COS(anchors, positives) + alphas)
        corrects = (COS(anchors, negatives) < COS(anchors, positives))

        return losses.mean() if batch_average_loss else losses.sum(), torch.sum(corrects.int())
