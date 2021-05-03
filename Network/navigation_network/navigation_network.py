'''
Navigation Network, Written by Xiao
For robot navigation in a dynamic environment.
'''
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from Network.retrieval_network.params import IMAGE_SIZE
from Network.navigation_network.params import CHECKPOINTS_DIR
from Network.navigation_network.networks import NavigationNet

class Navigation_network():
    def __init__(self, isResNetLocalization=False):
        # Prepare model and load checkpoint
        self.isResNetLocalization = isResNetLocalization
        if self.isResNetLocalization:
            self.checkpoint = CHECKPOINTS_DIR + 'image_best_fit.pkl'
        else:
            self.checkpoint = CHECKPOINTS_DIR + 'best_fit.pkl'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.get_network()
        self.feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_network(self):
        model = NavigationNet(isResNetLocalization=self.isResNetLocalization)
        model.to(self.device)
        model.load_state_dict(torch.load(self.checkpoint))
        model.eval()
        return model
    # ------------------------------------------------------------------------------
    # This function is used to navigate the robot given features from current and goal pose
    # ------------------------------------------------------------------------------
    # INPUT: current_info, goal_info in tuple format with containt (img, R_on, R_in, R_prox, bbox, obj_vec)
    #        model is the network loaded with pretrained weights
    #        device is CPU or cuda:0
    # ----------------------------------------------------------
    # OUTPUT: True (reached the goal) || False (did not reach the goal)
    # ----------------------------------------------------------
    def toTuple(self, info):
        info[0] = self.feature_transforms(info[0]).clone().detach()
        current_info = tuple(input.clone().detach() if torch.is_tensor(input) else torch.tensor(input) for input in info)
        infoTuple = tuple(torch.unsqueeze(input.clone().detach(), 0).to(self.device) for input in current_info)
        return infoTuple

    def action_prediction(self, current_info, goal_info):
        if self.isResNetLocalization:
            current_info = self.feature_transforms(current_info).unsqueeze(dim=0)
            current_info = current_info.clone().detach().to(self.device)
            goal_info = self.feature_transforms(goal_info).unsqueeze(dim=0)
            goal_info = goal_info.clone().detach().to(self.device)
        else:
            current_info = self.toTuple(current_info)
            goal_info = self.toTuple(goal_info)

        current_embedding = self.model.get_embedding(current_info)
        goal_embedding = self.model.get_embedding(goal_info)
        concacenated = torch.cat((current_embedding, goal_embedding), dim=1)
        distribution = self.decisionHead(concacenated)

        action = torch.argmax(distribution, dim=1)

        return action
