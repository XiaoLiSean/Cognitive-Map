# Import params and similarity from lib module
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from Network.retrieval_network.params import IMAGE_SIZE, COS, CHECKPOINTS_PREFIX
from Network.retrieval_network.networks import TripletNetImage, SiameseNetImage

# ------------------------------------------------------------------------------
# This function is used to tell if two images are similar to each other
# ------------------------------------------------------------------------------
# INPUT: current_img, goal_img in PIL image format
#        model is the network loaded with pretrained weights
#        device is CPU or cuda:0
# ----------------------------------------------------------
# OUTPUT: True (reached the goal) || False (did not reach the goal)
# ----------------------------------------------------------
def is_localized_static(model, device, current_img, goal_img):
    feature_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    current_img = feature_transforms(current_img).unsqueeze(dim=0)
    current_img = current_img.to(device)
    goal_img = feature_transforms(goal_img).unsqueeze(dim=0)
    goal_img = goal_img.to(device)
    similarity = COS(model.get_embedding(current_img), model.get_embedding(goal_img)).item()

    return similarity == 1.0

if __name__ == '__main__':
    # Prepare model and load checkpoint
    checkpoint = CHECKPOINTS_PREFIX + 'image_siamese_nondynamics_best_fit.pkl'
    model = SiameseNetImage()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model testing on: ", device)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # Generate images for 
    array = Image.fromarray(np.zeros([512, 512, 3], dtype=np.uint8))
    print(is_localized_static(model, device, array, array))
