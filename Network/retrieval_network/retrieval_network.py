'''
Retrieval Network, Written by Xiao
For robot localization in a dynamic environment.
'''
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from Network.retrieval_network.params import IMAGE_SIZE, CHECKPOINTS_DIR
from Network.retrieval_network.networks import RetrievalTriplet, TripletNetImage

class Retrieval_network():
    def __init__(self, netName, isImageLocalization=False):
        # Prepare model and load checkpoint
        self.netName = netName
        self.isImageLocalization = isImageLocalization
        self.checkpoint = CHECKPOINTS_DIR + netName + '/' + 'best_fit.pkl'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.get_network()
        self.COS = torch.nn.CosineSimilarity(dim=1, eps=1e-10)
        self.feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_network(self):
        if self.isImageLocalization:
            model = TripletNetImage(enableRoIBridge=False, pretrainedXXXNet=True, XXXNetName=self.netName)
        else:
            model = RetrievalTriplet(self_pretrained_image=False, pretrainedXXXNet=True)
        model.to(self.device)
        model.load_state_dict(torch.load(self.checkpoint))
        model.eval()
        return model
    # ------------------------------------------------------------------------------
    # This function is used to tell if two images are similar to each other
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

    def get_similarity(self, current_info, goal_info):
        if self.isImageLocalization:
            current_info = self.feature_transforms(current_info).unsqueeze(dim=0)
            current_info = current_info.clone().detach().to(self.device)
            goal_info = self.feature_transforms(goal_info).unsqueeze(dim=0)
            goal_info = goal_info.clone().detach().to(self.device)
            similarity = self.COS(self.model.get_embedding(current_info), self.model.get_embedding(goal_info)).item()
        else:
            current_info = self.toTuple(current_info)
            goal_info = self.toTuple(goal_info)
            similarity = self.COS(self.model.get_embedding(*current_info), self.model.get_embedding(*goal_info)).item()

        return similarity

    def is_localized(self, current_info, goal_info):
        similarity = self.get_similarity(current_info, goal_info)
        if self.isImageLocalization:
            '''This threshold is generated by thresholding testing: (mu,signma)=(0.97,0.04564)'''
            threshold = 0.93
        else:
            '''This threshold is generated by thresholding testing: (mu,signma)=(0.953,0.0502)'''
            threshold = 0.9
        return similarity >= threshold
