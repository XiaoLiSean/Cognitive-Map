import torch
import torch.nn.functional as F
from termcolor import colored
from Network.retrieval_network.params import BATCH_SIZE, ALPHA_MARGIN

COS = torch.nn.CosineSimilarity(dim=1, eps=1e-10)

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
        alphas = self.margin*torch.ones(BATCH_SIZE)
        alphas = alphas.to(self.device)
        # Using ReLU instead of max since max is not differentiable
        losses = F.relu(COS(anchors, negatives) - COS(anchors, positives) + alphas)
        corrects = (COS(anchors, negatives) < COS(anchors, positives))

        return losses.mean() if batch_average_loss else losses.sum(), torch.sum(corrects.int())
