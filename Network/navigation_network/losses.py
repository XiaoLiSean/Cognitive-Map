'''
Navigation Network
For robot localization in a dynamic environment.
'''
import torch
import torch.nn.functional as F
from termcolor import colored
from Network.retrieval_network.params import ALPHA_MARGIN

# ------------------------------------------------------------------------------
# This function is self defined CrossEntropyLoss
class Cross_Entropy_Loss(torch.nn.Module):
    def __init__(self):
        super(Cross_Entropy_Loss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fcn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets, batch_average_loss=True):
        targets_classes = torch.argmax(targets, dim=1)
        losses = self.loss_fcn(predictions, targets_classes)
        corrects = (targets_classes == torch.argmax(predictions, dim=1))

        return losses.mean() if batch_average_loss else losses.sum(), torch.sum(corrects.int())
