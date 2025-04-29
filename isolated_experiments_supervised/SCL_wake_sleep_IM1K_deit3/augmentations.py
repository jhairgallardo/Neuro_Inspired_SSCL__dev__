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
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225],
                 size = 224,
                 scale = (0.2, 1.0),
                 ratio = (3. / 4., 4. / 3.),
                 p_hflip = 0.5,
                 p_color_jitter = 0.8,
                 brightness = 0.4,
                 contrast = 0.4,
                 saturation = 0.2,
                 hue = 0.1,
                 p_grayscale = 1.0,
                 p_gaussian_blur=1.0,
                 sigma_range=(0.1, 2.0),
                 p_solarization=1.0,
                 ):
        
        self.num_views = num_views
        self.mean = mean
        self.std = std
        self.size = size

        # RandomResizedCrop parameters
        self.scale = scale
        self.ratio = ratio
        # RandomHorizontalFlip parameters
        self.p_hflip = p_hflip
        # ColorJitter parameters
        self.p_color_jitter = p_color_jitter
        self.brightness_range = [1-brightness, 1+brightness]
        self.contrast_range = [1-contrast, 1+contrast]
        self.saturation_range = [1-saturation, 1+saturation]
        self.hue_range = [-hue, hue]
        # RandomGrayscale parameters
        self.p_grayscale = p_grayscale
        # RandomBlur parameters
        self.p_gaussian_blur = p_gaussian_blur
        self.sigma_range = sigma_range
        # RandomSolarization parameters
        self.p_solarization = p_solarization

        # resize to size value function for first view
        self.resize_only = transforms.Resize((self.size,self.size))

        # function to convert to tensor and normalize views
        self.tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.totensor = transforms.ToTensor()

    def random_resized_crop(self, img, size, scale, ratio, interpolation=InterpolationMode.BILINEAR):
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
                return img, torch.tensor([i/size, j/size, h/size, w/size], dtype=torch.float)
            
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
        img = F.resized_crop(img, i, j, h, w, [size,size], interpolation)
        return img, torch.tensor([i/size, j/size, h/size, w/size], dtype=torch.float)
        
    def apply_random_flip(self, img, p_hflip):
        # random_flip
        flip_code = 0
        if torch.rand(1) < p_hflip:
            img = F.hflip(img)
            flip_code = 1
        return img, torch.tensor([flip_code], dtype=torch.float)
    
    def color_jitter(self, img_input, p_color_jitter, brightness_range, contrast_range, saturation_range, hue_range):
        # ColorJitter  -- use with caution. Values w/o _check_input for range test
        color_jitter_code = [1., 1., 1., 0.] # Values for no changes
        # Order: first brightness, then contrast, then saturation, then hue.
        if torch.rand(1) < p_color_jitter:
            brightness_factor = torch.empty(1).uniform_(brightness_range[0], brightness_range[1]).item()
            img = F.adjust_brightness(img_input, brightness_factor)
            color_jitter_code[0] = brightness_factor

            contrast_factor = torch.empty(1).uniform_(contrast_range[0], contrast_range[1]).item()
            img = F.adjust_contrast(img, contrast_factor)
            color_jitter_code[1] = contrast_factor

            saturation_factor = torch.empty(1).uniform_(saturation_range[0], saturation_range[1]).item()
            img = F.adjust_saturation(img, saturation_factor)
            color_jitter_code[2] = saturation_factor

            hue_factor = torch.empty(1).uniform_(hue_range[0], hue_range[1]).item()
            img = F.adjust_hue(img, hue_factor)
            color_jitter_code[3] = hue_factor
        else:
            img = img_input
        
        # Calculate the mean per channel of img_input and img. (img_input is a PIL image)
        # Idea from this paper: https://arxiv.org/abs/2306.06082
        img_input_tensor = self.totensor(img_input)
        mean_input = img_input_tensor.mean(dim=(1, 2))
        img_tensor = self.totensor(img)
        mean_output = img_tensor.mean(dim=(1, 2))
        # get mean diff
        mean_diff = mean_input - mean_output
        # Concatenate the mean diff with the color_jitter_code
        color_jitter_code = torch.tensor(color_jitter_code, dtype=torch.float)
        color_jitter_code = torch.cat((color_jitter_code, mean_diff), dim=0)

        return img, color_jitter_code
    
    def apply_random_grayscale(self, img, p_grayscale):
        # RandomGrayscale
        grayscale_code = 0
        if torch.rand(1) < p_grayscale:
            num_output_channels, _, _ = F.get_dimensions(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            grayscale_code = 1
        return img, torch.tensor([grayscale_code], dtype=torch.float)

    def apply_gaussian_blur(self, img, p_gaussian_blur, sigma_range):
        # GaussianBlur
        gaussian_blur_code = 0.1 # applying 0.1 is basically no blur (output image values are the same as input image values)
        if torch.rand(1) < p_gaussian_blur:
            sigma = torch.empty(1).uniform_(sigma_range[0], sigma_range[1]).item()
            img = img.filter(ImageFilter.GaussianBlur(sigma))
            gaussian_blur_code = sigma
        return img, torch.tensor([gaussian_blur_code], dtype=torch.float)
    
    def apply_solarization(self, img, p_solarization):
        # Solarization
        solarization_code = 0
        if torch.rand(1) < p_solarization:
            # solarization_code = 1
            # solarization code is the ratio of pixel that are solarized to the total number of pixels
            solarization_code = torch.sum(self.totensor(img) > 0.5) / (3 * img.size[0] * img.size[1])
            img = ImageOps.solarize(img, threshold=128)
        return img, torch.tensor([solarization_code], dtype=torch.float)
            
    def __call__(self, original_image):

        '''
        Action vector: [4 -> randcrop, 1-> horizontal flip, 4 -> colorjitter, 1 -> grayscale, 1 -> gaussianblur]
                       4 randcrop: [i, j, h, w] where i,j are the top left corner of the crop, 
                                   and h,w are the height and width of the crop. (divided by size)
                       1 horizontalflip: [0 -> no flip, 1 -> flip]
                       7 colorjitter: [brightness, contrast, saturation, hue, mean_diff_R, mean_diff_G, mean_diff_B]
                       1 grayscale: [0 -> no grayscale, 1 -> grayscale]
                       1 gaussianblur: [0 -> no blur, sigma -> blur] where sigma is the standard deviation of the gaussian blur (0.1 to 2.0)
                       1 solarization: [0 -> no solarization, ratio -> solarization] where ratio is the ratio of pixels that are solarized to the total number of pixels       
        '''
        
        # initialize views tensor, and actions tensor
        views = torch.zeros(self.num_views, 3, self.size, self.size) 
        action_vectors = torch.zeros(self.num_views, 15, dtype=torch.float)

        ########################################################
        #### Get first view and no action vector
        ########################################################
        first_view = self.resize_only(original_image)
        views[0] = self.tensor_normalize(first_view)
        no_action_cropbb = torch.tensor([0., 0., 1., 1.], dtype=torch.float) # RandomCrop no action -> get whole image
        no_action_horizontalflip = torch.tensor([0.], dtype=torch.float) # HorizontalFlip no action -> no flip
        no_action_colorjitter = torch.tensor([1., 1., 1., 0., 0., 0., 0.], dtype=torch.float) # ColorJitter no action -> no change of brightness, contrast, saturation, hue. Diff is 0 for all channels.
        no_action_grayscale = torch.tensor([0.], dtype=torch.float) # RandomGrayscale no action -> no grayscale
        no_action_gaussianblur = torch.tensor([0.1], dtype=torch.float) # GaussianBlur no action -> no blur. for kernel size 23 and sigma 0.069, the image is not blurred. (kernel has a one and a bunch of zeros)
        no_action_solarization = torch.tensor([0.], dtype=torch.float) # Solarization no action -> no solarization
        ## Normalize no_action vectors to be between 0 and 1
        # normalize action_colorjitter (all values between 0 and 1)
        no_action_colorjitter[0] = (no_action_colorjitter[0] - self.brightness_range[0]) / (self.brightness_range[1] - self.brightness_range[0])
        no_action_colorjitter[1] = (no_action_colorjitter[1] - self.contrast_range[0]) / (self.contrast_range[1] - self.contrast_range[0])
        no_action_colorjitter[2] = (no_action_colorjitter[2] - self.saturation_range[0]) / (self.saturation_range[1] - self.saturation_range[0])
        no_action_colorjitter[3] = (no_action_colorjitter[3] - self.hue_range[0]) / (self.hue_range[1] - self.hue_range[0])
        # normalize action_gaussianblur (all values between 0 and 1)
        no_action_gaussianblur[0] = (no_action_gaussianblur[0] - self.sigma_range[0]) / (self.sigma_range[1] - self.sigma_range[0])
        ## No action vector (for first view)
        action_vectors[0] = torch.cat((no_action_cropbb, no_action_horizontalflip, no_action_colorjitter, no_action_grayscale, no_action_gaussianblur, no_action_solarization), dim=0)

        ################################################
        #### Get other views and their action vectors
        ################################################
        for i in range(1, self.num_views):
            ## Random Resized Crop
            view, action_cropbb = self.random_resized_crop(first_view, 
                                                           size = self.size,
                                                           scale = self.scale, 
                                                           ratio = self.ratio)
            ## Random Horizontal Flip
            view, action_horizontalflip = self.apply_random_flip(view, self.p_hflip)
            ## ColorJitter
            view, action_colorjitter = self.color_jitter(view,
                                                        p_color_jitter = self.p_color_jitter, 
                                                        brightness_range = self.brightness_range, 
                                                        contrast_range = self.contrast_range, 
                                                        saturation_range = self.saturation_range, 
                                                        hue_range = self.hue_range)
            
            # ThreeAgument from Deit3
            # Randomly choose between grayscale, gaussian blur, and solarization (uniformly)
            action_grayscale = torch.tensor([0.], dtype=torch.float)
            action_gaussianblur = torch.tensor([0.1], dtype=torch.float)
            action_solarization = torch.tensor([0.], dtype=torch.float)
            choice = random.randint(0, 2)

            ## Random Grayscale
            if choice == 0:
                view, action_grayscale = self.apply_random_grayscale(view, p_grayscale = self.p_grayscale)
            ## Gaussian Blur
            elif choice == 1:
                view, action_gaussianblur = self.apply_gaussian_blur(view, 
                                                                 p_gaussian_blur = self.p_gaussian_blur, 
                                                                 sigma_range = self.sigma_range)
            ## Solarization
            elif choice == 2:
                view, action_solarization = self.apply_solarization(view, p_solarization = self.p_solarization)

            ## Normalize action vectors to be between 0 and 1
            # normalize action_colorjitter (all values between 0 and 1)
            action_colorjitter[0] = (action_colorjitter[0] - self.brightness_range[0]) / (self.brightness_range[1] - self.brightness_range[0])
            action_colorjitter[1] = (action_colorjitter[1] - self.contrast_range[0]) / (self.contrast_range[1] - self.contrast_range[0])
            action_colorjitter[2] = (action_colorjitter[2] - self.saturation_range[0]) / (self.saturation_range[1] - self.saturation_range[0])
            action_colorjitter[3] = (action_colorjitter[3] - self.hue_range[0]) / (self.hue_range[1] - self.hue_range[0])
            # normalize action_gaussianblur (all values between 0 and 1)
            action_gaussianblur[0] = (action_gaussianblur[0] - self.sigma_range[0]) / (self.sigma_range[1] - self.sigma_range[0])
            
            views[i] = self.tensor_normalize(view)
            action_vectors[i] = torch.cat((action_cropbb, action_horizontalflip, action_colorjitter, action_grayscale, action_gaussianblur, action_solarization), dim=0)
    
        return views, action_vectors

# Function to test which sigma creates a no blur kernel (a bunch of zeros and a one) for kernel size 23
# def _get_gaussian_kernel1d(kernel_size: int, sigma: float):
#     ksize_half = (kernel_size - 1) * 0.5

#     x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
#     pdf = torch.exp(-0.5 * (x / sigma).pow(2))
#     kernel1d = pdf / pdf.sum()

#     return kernel1d

# Test the transformations
if __name__ == "__main__":
    # Create an instance of the Episode_Transformations class
    transform = Episode_Transformations(num_views=2)

    # Load an example image
    img = transforms.ToPILImage()(torch.rand(3, 256, 256))

    # Apply the transformations
    views, actions = transform(img)

    # Print the results
    print("Views shape:", views.shape)
    print("Actions shape:", actions.shape)
    




