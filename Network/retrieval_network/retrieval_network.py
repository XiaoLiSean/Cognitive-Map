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
    def __init__(self, isStaticEnv=False):
        # Prepare model and load checkpoint
        self.isStaticEnv = isStaticEnv
        if self.isStaticEnv:
            self.checkpoint = CHECKPOINTS_DIR + 'image_best_fit.pkl'
        else:
            self.checkpoint = CHECKPOINTS_DIR + 'best_fit.pkl'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.get_network()
        self.COS = torch.nn.CosineSimilarity(dim=1, eps=1e-10)
        self.feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_network(self):
        if self.isStaticEnv:
            model = TripletNetImage()
        else:
            model = RetrievalTriplet()
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
    def is_localized(self, current_info, goal_info, threshold=0.94):
        if self.isStaticEnv:
            current_info = self.feature_transforms(current_info).unsqueeze(dim=0)
            current_info = current_info.to(self.device)
            goal_info = self.feature_transforms(goal_info).unsqueeze(dim=0)
            goal_info = goal_info.to(self.device)
            similarity = self.COS(self.model.get_embedding(current_info), self.model.get_embedding(goal_info)).item()
        else:
            current_info[0] = self.feature_transforms(current_info[0])
            current_info = tuple(torch.unsqueeze(torch.tensor(input), 0).to(self.device) for input in current_info)
            goal_info[0] = self.feature_transforms(goal_info[0])
            goal_info = tuple(torch.unsqueeze(torch.tensor(input), 0).to(self.device) for input in goal_info)
            similarity = self.COS(self.model.get_embedding(*current_info), self.model.get_embedding(*goal_info)).item()

        return similarity >= threshold

if __name__ == '__main__':
    # Prepare model and load checkpoint
    checkpoint = CHECKPOINTS_DIR + 'image_siamese_nondynamics_best_fit.pkl'
    model = SiameseNetImage()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model testing on: ", device)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # Generate images for
    array = Image.fromarray(np.zeros([512, 512, 3], dtype=np.uint8))
    print(is_localized_static(model, device, array, array))
