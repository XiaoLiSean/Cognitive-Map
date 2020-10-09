# This module is used to define siamese networks
import torch
from torchvision import models

class SiameseNetImage(torch.nn.Module):
    def __init__(self):
        super(SiameseNetImage, self).__init__()
        model = models.resnet50(pretrained=True)
        # Strip final fc layer: self.embedding output 512d
        self.embedding = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, img1, img2):
        embedding1 = self.embedding(img1)
        embedding2 = self.embedding(img2)
        return embedding1, embedding2

    def get_embedding(self, img):
        return self.embedding(img)

class TripletNetImage(torch.nn.Module):
    def __init__(self):
        super(TripletNetImage, self).__init__()
        model = models.resnet50(pretrained=True)
        # Strip final fc layer: self.embedding output 512d
        self.embedding = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, anchor_img, positive_img, negative_img):
        anchor = self.embedding(anchor_img)
        positive = self.embedding(positive_img)
        negative = self.embedding(negative_img)
        return anchor, positive, negative

    def get_embedding(self, img):
        return self.embedding(img)
