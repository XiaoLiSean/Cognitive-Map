'''
Retrieval Network, Written by Xiao
For robot localization in a dynamic environment.
'''
import torch
import torch.nn.functional as F
from termcolor import colored
from Network.retrieval_network.params import ALPHA_MARGIN

CosineSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-10)

# ------------------------------------------------------------------------------
# This function is triplet loss with variable margin alpha defined by view cone overlaps
class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    def __init__(self, margin=ALPHA_MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, anchors, positives, negatives, batch_average_loss=True):
        alphas = self.margin*torch.ones(anchors.shape[0])
        alphas = alphas.to(self.device)
        # Using ReLU instead of max since max is not differentiable
        losses = F.relu(CosineSimilarity(anchors, negatives) - CosineSimilarity(anchors, positives) + alphas)
        corrects = (CosineSimilarity(anchors, negatives) < CosineSimilarity(anchors, positives))

        return losses.mean() if batch_average_loss else losses.sum(), torch.sum(corrects.int())
