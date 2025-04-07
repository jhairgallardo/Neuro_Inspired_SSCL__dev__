import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import random
from PIL import ImageFilter, ImageOps, ImageFilter
import math


class Episode_Transformations:
    def __init__(self, 
                 num_views,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 size=224,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 p_color_jitter=0.8,
                 brightness=0.4,
                 contrast = 0.4,
                 saturation = 0.2,
                 hue = 0.1,
                 p_grayscale=0.2,
                 p_gaussian_blur=0.1,
                 p_solarization=0.2,
                 return_actions=False):
        
        self.num_views = num_views
        self.mean = mean
        self.std = std
        self.size = size

        self.scale = scale
        self.ratio = ratio

        self.p_color_jitter = p_color_jitter
        self.brightness = [1-brightness, 1+brightness]
        self.contrast = [1-contrast, 1+contrast]
        self.saturation = [1-saturation, 1+saturation]
        self.hue = [-hue, hue]

        self.p_grayscale = p_grayscale
        self.p_gaussian_blur = p_gaussian_blur
        self.p_solarization = p_solarization

        self.return_actions = return_actions

        # random flip function
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
        # resize to size value function
        self.resize_only = transforms.Resize((self.size,self.size))

        # function to convert to tensor and normalize views
        self.tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def random_resized_crop(self, img, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=InterpolationMode.BILINEAR):
        # RandomResizedCrop
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                img = F.resized_crop(img, i, j, h, w, [size,size], interpolation)
                return img, [i/size, j/size, h/size, w/size]
            
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        img = F.resized_crop(img, i, j, h, w, [size,size])

        if self.return_actions:
            return img, [i / size, j / size, h / size, w / size]
        else:
            return img, None
    
    def color_jitter(self, img, p_color_jitter=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        # ColorJitter  -- use with caution. Values w/o _check_input for range test
        color_jitter_code = [1.,1.,1.,0.] # adjust_brightness; adjust_contrast; adjust_saturation; adjust_hue
        action_order = [0, 1, 2, 3] # first brightness, then contrast, then saturation, then hue
        if torch.rand(1) < p_color_jitter:
            for action in action_order:
                if action == 0:
                    brightness_factor = torch.empty(1).uniform_(brightness[0], brightness[1]).item()
                    img = F.adjust_brightness(img, brightness_factor)
                    color_jitter_code[0] = brightness_factor
                elif action == 1:
                    contrast_factor = torch.empty(1).uniform_(contrast[0], contrast[1]).item()
                    img = F.adjust_contrast(img, contrast_factor)
                    color_jitter_code[1] = contrast_factor
                elif action == 2:
                    saturation_factor = torch.empty(1).uniform_(saturation[0], saturation[1]).item()
                    img = F.adjust_saturation(img, saturation_factor)
                    color_jitter_code[2] = saturation_factor
                elif action == 3:
                    hue_factor = torch.empty(1).uniform_(hue[0], hue[1]).item()
                    img = F.adjust_hue(img, hue_factor)
                    color_jitter_code[3] = hue_factor
        if self.return_actions:
            return img, color_jitter_code
        else:
            return img, None
    
    def apply_random_grayscale(self, img, p_grayscale=0.2):
        # RandomGrayscale
        grayscale_code = 0
        if torch.rand(1) < p_grayscale:
            num_output_channels, _, _ = F.get_dimensions(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            grayscale_code = 1
        if self.return_actions:
            return img, grayscale_code
        else:
            return img, None

    def apply_gaussian_blur(self, img, p_gaussian_blur=0.1):
        # GaussianBlur
        gaussian_blur_code = 0
        if torch.rand(1) < p_gaussian_blur:
            sigma = torch.empty(1).uniform_(0.1, 2.0).item()
            img = img.filter(ImageFilter.GaussianBlur(sigma))
            gaussian_blur_code = sigma
        if self.return_actions:
            return img, gaussian_blur_code
        else:
            return img, None

    def apply_solarization(self, img, p_solarization=0.2):
        # Solarization
        solarization_code = 0
        if torch.rand(1) < p_solarization:
            img = ImageOps.solarize(img)
            solarization_code = 1
        if self.return_actions:
            return img, solarization_code
        else:
            return img, None
            
    def __call__(self, x):

        '''
        Action vector: [4 -> randcrop, 4 -> colorjitter, 1 -> grayscale, 1 -> gaussianblur, 1 -> solarization]
                       4 randcrop: [i, j, h, w] where i,j are the top left corner of the crop, 
                                   and h,w are the height and width of the crop. (divided by size)
                       4 colorjitter: [brightness, contrast, saturation, hue]
                       1 grayscale: [0 -> no grayscale, 1 -> grayscale]
                       1 gaussianblur: [0 -> no blur, 1 -> blur]
                       1 solarization: [0 -> no solarization, 1 -> solarization]
        '''
        
        # initialize views tensor, and actions tensor
        views = torch.zeros(self.num_views, 3, self.size, self.size) 
        if self.return_actions:
            action_vectors = torch.zeros(self.num_views, 11)

        # Randomly flip original image first
        original_image = self.random_flip(x)

        # Get first view and no transfomation action vector
        first_view = self.resize_only(original_image)
        views[0] = self.tensor_normalize(first_view)
        if self.return_actions:
            action_vectors[0] = torch.tensor([0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.])

        # Get other views and their action vectors
        for i in range(1, self.num_views):
            # use first view (already 224x224) to create other views. 
            # Using the original view will cause action_cropbb to not be realtive to 224x224
            view, action_cropbb = self.random_resized_crop(first_view, 
                                                           size = self.size,
                                                           scale = self.scale, 
                                                           ratio = self.ratio)
            # view=first_view # colorjitter only
            view, action_colorjitter = self.color_jitter(view, 
                                                         p_color_jitter = self.p_color_jitter, 
                                                         brightness = self.brightness, 
                                                         contrast = self.contrast, 
                                                         saturation = self.saturation, 
                                                         hue = self.hue)
            view, action_grayscale = self.apply_random_grayscale(view,
                                                                 p_grayscale = self.p_grayscale)
            view, action_gaussianblur = self.apply_gaussian_blur(view,
                                                                 p_gaussian_blur = self.p_gaussian_blur)
            view, action_solarization = self.apply_solarization(view,
                                                                p_solarization = self.p_solarization)
            views[i] = self.tensor_normalize(view)
            if self.return_actions: 
                action_vectors[i] = torch.tensor([action_cropbb[0], action_cropbb[1], action_cropbb[2], action_cropbb[3],
                                                                    action_colorjitter[0], action_colorjitter[1], action_colorjitter[2], action_colorjitter[3],
                                                                    action_grayscale, action_gaussianblur, action_solarization])

                # # Randcrop only
                # action_vectors[i] = torch.tensor([action_cropbb[0], action_cropbb[1], action_cropbb[2], action_cropbb[3], 
                #                                   1., 1., 1., 0., 
                #                                   0., 0., 0.])

                # Colorjitter only
                # action_vectors[i] = torch.tensor([0., 0., 1., 1., 
                #                                  action_colorjitter[0], action_colorjitter[1], action_colorjitter[2], action_colorjitter[3],
                #                                  0., 0., 0.])
        
        if self.return_actions:
            return views, action_vectors
        else:
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