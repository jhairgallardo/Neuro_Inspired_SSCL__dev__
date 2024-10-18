import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import ImageOps, ImageFilter
import math


class aug_with_action_code:
    def __init__(self,
                 size,
                 # flip
                 p_flip=0.0,
                 # RandomResizedCrop
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 # ColorJitter
                 p_color_jitter=0.8,
                 brightness=0.4,
                 contrast = 0.4,
                 saturation = 0.2,
                 hue = 0.1,
                 # RandomGrayscale
                 p_grayscale=0.2,
                 # GaussianBlur
                 p_gaussian_blur=0.1,
                 # Solarization
                 p_solarization=0.2,
                 # Normalize
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]
                 ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

        self.p_flip = p_flip
        self.p_color_jitter = p_color_jitter
        self.p_grayscale = p_grayscale
        self.p_gaussian_blur = p_gaussian_blur
        self.p_solarization = p_solarization
        # follow original _check_input function to set the range of brightness, contrast, saturation, hue
        # range test not implemented, use with caution
        self.brightness = [1-brightness, 1+brightness]
        self.contrast = [1-contrast, 1+contrast]
        self.saturation = [1-saturation, 1+saturation]
        self.hue = [-hue, hue]

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


    def apply_random_flip(self, img):
        # random_flip
        flip_code = 0
        if torch.rand(1) < self.p_flip:
            img = F.hflip(img)
            flip_code = 1
        return img, flip_code

    def apply_random_resized_crop(self, img):
        # RandomResizedCrop
        _, height, width = F.get_dimensions(img)
        area = height * width
        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                img = F.resized_crop(img, i, j, h, w, (self.size, self.size))
                # return img, [i, j, h, w]
                return img, [i/self.size, j/self.size, h/self.size, w/self.size]


        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        img = F.resized_crop(img, i, j, h, w, (self.size, self.size))
        # return img, [i, j, h, w]
        return img, [i/self.size, j/self.size, h/self.size, w/self.size]

    def apply_color_jitter(self, img):
        # ColorJitter
        color_jitter_code = [1.,1.,1.,0.] # adjust_brightness; adjust_contrast; adjust_saturation; adjust_hue
        action_order = torch.randperm(4).tolist()
        if torch.rand(1) < self.p_color_jitter:
            for action in action_order:
                if action == 0:
                    brightness_factor = torch.empty(1).uniform_(self.brightness[0], self.brightness[1]).item()
                    img = F.adjust_brightness(img, brightness_factor)
                    color_jitter_code[0] = brightness_factor
                elif action == 1:
                    contrast_factor = torch.empty(1).uniform_(self.contrast[0], self.contrast[1]).item()
                    img = F.adjust_contrast(img, contrast_factor)
                    color_jitter_code[1] = contrast_factor
                elif action == 2:
                    saturation_factor = torch.empty(1).uniform_(self.saturation[0], self.saturation[1]).item()
                    img = F.adjust_saturation(img, saturation_factor)
                    color_jitter_code[2] = saturation_factor
                elif action == 3:
                    hue_factor = torch.empty(1).uniform_(self.hue[0], self.hue[1]).item()
                    img = F.adjust_hue(img, hue_factor)
                    color_jitter_code[3] = hue_factor
        return img, color_jitter_code + action_order

    def apply_random_grayscale(self, img):
        # RandomGrayscale
        grayscale_code = 0
        if torch.rand(1) < self.p_grayscale:
            num_output_channels, _, _ = F.get_dimensions(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            grayscale_code = 1
        return img, grayscale_code

    def apply_gaussian_blur(self, img):
        # GaussianBlur
        gaussian_blur_code = 0
        if torch.rand(1) < self.p_gaussian_blur:
            sigma = torch.empty(1).uniform_(0.1, 2.0).item()
            img = img.filter(ImageFilter.GaussianBlur(sigma))
            gaussian_blur_code = sigma
        return img, gaussian_blur_code


    def apply_solarization(self, img):
        # Solarization
        solarization_code = 0
        if torch.rand(1) < self.p_solarization:
            img = ImageOps.solarize(img)
            solarization_code = 1
        return img, solarization_code

    @staticmethod
    def squeeze(actions):
        code = []
        for a in actions:
            if isinstance(a, (list, tuple)):
                code += a
            else:
                code.append(a)
        return code

    def __call__(self, img):
        actions = []
        original_img = self.normalize(img)

        img, flip_code = self.apply_random_flip(img)
        actions.append(flip_code)
        img, crop_code = self.apply_random_resized_crop(img)
        actions.append(crop_code)
        img, color_jitter_code = self.apply_color_jitter(img)
        actions.append(color_jitter_code)
        img, grayscale_code = self.apply_random_grayscale(img)
        actions.append(grayscale_code)
        img, gaussian_blur_code = self.apply_gaussian_blur(img)
        actions.append(gaussian_blur_code)
        img, solarization_code = self.apply_solarization(img)
        actions.append(solarization_code)
        actions = self.squeeze(actions)
        actions = torch.tensor(actions, dtype=torch.float32)
        img = self.normalize(img)
        # (1, 4, 8, 1, 1, 1)

        return original_img, img, actions




class ActionCodeDataset(Dataset):
    def __init__(self, root, action_aug, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.tensor_transform = action_aug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get the image and its label
        img, label = self.dataset[index]
        original_img, img, actions = self.tensor_transform(img)
        return original_img, img, actions, label
