import torch
from torchvision import transforms

import random
from PIL import ImageFilter, ImageOps, ImageFilter


class Episode_Transformations:
    def __init__(self, num_views, zca=False):
        self.num_views = num_views
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if zca: std = [1.0, 1.0, 1.0]
        self.mean = mean
        self.std = std

        # random flip function
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
        
        # function to create first view
        self.create_first_view = transforms.Compose([
                transforms.Resize((224,224)),
                ])
        
        # function to create other views
        self.create_view = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
                ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2)])

        # function to convert to tensor and normalize views
        self.tensor_normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ])
            
    def __call__(self, x):
        views = torch.zeros(self.num_views, 3, 224, 224) # initialize views tensor
        original_image = self.random_flip(x) # randomly flip original image first
        first_view = self.create_first_view(original_image) # create first view
        views[0] = self.tensor_normalize(first_view)
        for i in range(1, self.num_views): # create other views with augmentations
            views[i] = self.tensor_normalize(self.create_view(original_image))
        return views
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img