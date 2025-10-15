import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import random
from PIL import ImageFilter, ImageOps, ImageFilter
import math

import hashlib
from torchvision.datasets import ImageFolder

class DeterministicEpisodes:
    """
    Wraps a stochastic Episode_Transformations to make its randomness
    deterministic per *image path* (key). This ensures a fixed val episode set.
    """
    def __init__(self, base_transform, base_seed: int = 0):
        self.base_transform = base_transform
        self.base_seed = int(base_seed)

    @staticmethod
    def _seed_from_key(key: str, base_seed: int) -> int:
        # Stable 64-bit hash → 31-bit torch seed
        h = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
        s = int.from_bytes(h, 'big') ^ base_seed
        return s % (2**31 - 1)

    def __call__(self, img, *, key: str):
        # Isolate RNG so we don't disturb global state
        py_state = random.getstate()
        try:
            seed = self._seed_from_key(key, self.base_seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                random.seed(seed)
                return self.base_transform(img)  # returns (views, aug_seq)
        finally:
            random.setstate(py_state)


class ImageFolderDetEpisodes(ImageFolder):
    """
    ImageFolder that passes the file path as a 'key' to DeterministicEpisodes.
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # Expect DeterministicEpisodes here
            sample = self.transform(sample, key=path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

def collate_function(batch):
    """
    batch = list of tuples returned by __getitem__
    0) stack *views*  -> Tensor (B, V, 3, H, W)
    1) keep *aug_lists* as pure Python list-of-lists
    """
    views_b     = torch.stack([item[0][0] for item in batch], dim=0)
    aug_lists_b = [item[0][1] for item in batch]   # len = B
    labels_b    = torch.tensor([item[1] for item in batch]) # labels\
    taskid_b   = torch.tensor([item[2] for item in batch]) # taskid
    return (views_b, aug_lists_b), labels_b , taskid_b

def collate_function_notaskid(batch):
    """
    batch = list of tuples returned by __getitem__
    0) stack *views*  -> Tensor (B, V, 3, H, W)
    1) keep *aug_lists* as pure Python list-of-lists
    """
    views_b     = torch.stack([item[0][0] for item in batch], dim=0)
    aug_lists_b = [item[0][1] for item in batch]   # len = B
    labels_b    = torch.tensor([item[1] for item in batch]) # labels\
    return (views_b, aug_lists_b), labels_b

class Episode_Transformations:
    def __init__(self, 
                 num_views,
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225],
                 p_crop = 0.9,
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
                 p_threeaugment=0.9,
                 p_noaugment=0.05,
                 ):
        
        self.num_views = num_views
        self.mean = mean
        self.std = std
        self.size = size

        # RandomResizedCrop parameters
        self.p_crop = p_crop
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

        self.p_threeaugment = p_threeaugment
        self.p_noaugment = p_noaugment

    @staticmethod
    def _minmax01(x, lo, hi):
        """map [lo,hi] → [0,1]"""
        return (x - lo) / (hi - lo)

    @staticmethod
    def _minmax11(x, lo, hi):
        """map [lo,hi] → [-1,1] (centre at 0)"""
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def random_resized_crop(self, img, p_crop, size, scale, ratio, interpolation=InterpolationMode.BILINEAR):
        crop_flag = False
        if torch.rand(1) < p_crop:
            crop_flag = True
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
                    return img, torch.tensor([i/height, j/width, h/height, w/width], dtype=torch.float), crop_flag
                
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
            return img, torch.tensor([i/height, j/width, h/height, w/width], dtype=torch.float), crop_flag
        else:
            crop_flag = False
            # If no crop, return the original image resized to size
            img = F.resize(img, [size, size], interpolation=interpolation)
            return img, torch.tensor([0., 0., 1., 1.], dtype=torch.float), crop_flag
        
    def apply_random_flip(self, img, p_hflip):
        # random_flip
        flip_code = 0
        hflip_flag=False
        if torch.rand(1) < p_hflip:
            img = F.hflip(img)
            flip_code = 1
            hflip_flag=True
        return img, torch.tensor([flip_code], dtype=torch.float), hflip_flag
    
    def color_jitter(self, img_input, p_color_jitter, brightness_range, contrast_range, saturation_range, hue_range):
        # ColorJitter  -- use with caution. Values w/o _check_input for range test
        color_jitter_code = [1., 1., 1., 0.] # Values for no changes
        jitter_flag=False
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
            jitter_flag=True
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
        color_jitter_code = color_jitter_code + [mean_diff[0].item(), mean_diff[1].item(), mean_diff[2].item()]

        return img, torch.tensor(color_jitter_code, dtype=torch.float), jitter_flag
    
    def apply_random_grayscale(self, img, p_grayscale):
        # RandomGrayscale
        grayscale_code = 0
        gray_flag=False
        if torch.rand(1) < p_grayscale:
            num_output_channels, _, _ = F.get_dimensions(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            grayscale_code = 1
            gray_flag=True
        return img, torch.tensor([grayscale_code], dtype=torch.float), gray_flag

    def apply_gaussian_blur(self, img, p_gaussian_blur, sigma_range):
        # GaussianBlur
        gaussian_blur_code = 0.1 # applying 0.1 is basically no blur (output image values are the same as input image values)
        blur_flag=False
        if torch.rand(1) < p_gaussian_blur:
            sigma = torch.empty(1).uniform_(sigma_range[0], sigma_range[1]).item()
            img = img.filter(ImageFilter.GaussianBlur(sigma))
            gaussian_blur_code = sigma
            blur_flag=True
        return img, torch.tensor([gaussian_blur_code], dtype=torch.float), blur_flag
    
    def apply_solarization(self, img, p_solarization):
        # Solarization
        solarization_code = 0
        solar_flag=False
        if torch.rand(1) < p_solarization:
            # solarization_code = 1
            # solarization code is the ratio of pixel that are solarized to the total number of pixels
            solarization_code = torch.sum(self.totensor(img) > 0.5) / (3 * img.size[0] * img.size[1])
            img = ImageOps.solarize(img, threshold=128)
            solar_flag=True
        return img, torch.tensor([solarization_code], dtype=torch.float), solar_flag
            
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
        aug_seq = []

        #################################################################################
        #### Get first view as the original image and pair it with an empty action vector
        #################################################################################
        first_view = self.resize_only(original_image)
        views[0] = self.tensor_normalize(first_view)
        aug_seq.append([]) # First view has no action because it is the original image

        ################################################
        #### Get other views and their action vectors
        ################################################
        for i in range(1, self.num_views):
            tokens = []

            ## Random Resized Crop
            view, action_cropbb, crop_flag = self.random_resized_crop(first_view,
                                                           p_crop = self.p_crop,
                                                           size = self.size,
                                                           scale = self.scale,
                                                           ratio = self.ratio)
            if crop_flag:
                tokens.append(("crop", action_cropbb))

            ## Random Horizontal Flip
            view, action_horizontalflip, hflip_flag = self.apply_random_flip(view, self.p_hflip)
            if hflip_flag:
                tokens.append(("hflip", torch.empty(0))) # param-less. Only needs type_emb later on.

            ## ColorJitter
            view, action_colorjitter, jitter_flag = self.color_jitter(view, p_color_jitter = self.p_color_jitter, 
                                                                    brightness_range = self.brightness_range, 
                                                                    contrast_range = self.contrast_range, 
                                                                    saturation_range = self.saturation_range, 
                                                                    hue_range = self.hue_range)
            if jitter_flag:
                action_colorjitter[0] = self._minmax11(action_colorjitter[0], self.brightness_range[0], self.brightness_range[1])
                action_colorjitter[1] = self._minmax11(action_colorjitter[1], self.contrast_range[0], self.contrast_range[1])
                action_colorjitter[2] = self._minmax11(action_colorjitter[2], self.saturation_range[0], self.saturation_range[1])
                action_colorjitter[3] = self._minmax11(action_colorjitter[3], self.hue_range[0], self.hue_range[1])
                tokens.append(("jitter", action_colorjitter)) 
            

            if torch.rand(1) < self.p_threeaugment: # apply threeaugment with a probability
                # ThreeAgument from Deit3
                # Randomly choose between grayscale, gaussian blur, and solarization (uniformly)
                choice = random.randint(0, 2)
                ## Random Grayscale
                if choice == 0:
                    view, action_grayscale, gray_flag = self.apply_random_grayscale(view, p_grayscale = self.p_grayscale)
                    if gray_flag:
                        tokens.append(("gray", torch.empty(0))) # param-less. Only needs type_emb later on.
                ## Gaussian Blur
                elif choice == 1:
                    view, action_gaussianblur, blur_flag = self.apply_gaussian_blur(view, p_gaussian_blur = self.p_gaussian_blur, 
                                                                                    sigma_range = self.sigma_range)
                    if blur_flag:
                        # action_gaussianblur[0] = self._minmax01(action_gaussianblur[0], self.sigma_range[0], self.sigma_range[1])
                        # tokens.append(("blur", action_gaussianblur))
                        sigma_log  = math.log(action_gaussianblur.item() / self.sigma_range[0]) \
                                    / math.log(self.sigma_range[1] / self.sigma_range[0])
                        sigma_norm = 2.0 * sigma_log - 1.0              # [-1, 1]
                        tokens.append(("blur", torch.tensor([sigma_norm], dtype=torch.float)))
                ## Solarization
                elif choice == 2:
                    view, action_solarization, solar_flag = self.apply_solarization(view, p_solarization = self.p_solarization)
                    if solar_flag:
                        tokens.append(("solar", action_solarization))

            ## Add the view and the action vector to the views and aug_seq
            views[i] = self.tensor_normalize(view)
            aug_seq.append(tokens)        
    
        return views, aug_seq

# Test the transformations
if __name__ == "__main__":

    import os
    from PIL import Image

    NUM_VIEWS=4

    # Create an instance of the Episode_Transformations class
    transform = Episode_Transformations(num_views=NUM_VIEWS)

    # Load an example image from "plots/image.png"
    pil_img = Image.open("plots/image.png").convert("RGB")

        # Generate episode
    views, actions = transform(pil_img)

    # Prepare a grid for visualization (unnormalize for display)
    unnorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(transform.mean, transform.std)],
        std=[1.0 / s for s in transform.std],
    )
    disp_imgs = torch.stack([torch.clamp(unnorm(v), 0, 1) for v in views], dim=0)  # (V,3,H,W)

    # Save grid
    import torchvision
    grid = torchvision.utils.make_grid(disp_imgs, nrow=NUM_VIEWS)
    grid_img = transforms.ToPILImage()(grid.cpu())
    grid_img.save("plots/episode_grid_firstori.png")



    # # Apply the transformations
    # views, actions = transform(img)

    # # Print the results
    # print("Views shape:", views.shape)
    # print("Actions shape:", len(actions))
    # for i, action in enumerate(actions):
    #     print(f"Action View {i}: {len(action)}")
    #     print("Action details:", action)
    




