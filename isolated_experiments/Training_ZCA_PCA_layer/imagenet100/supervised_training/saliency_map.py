import json, os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from resnet_gn_mish import *

from torchvision import transforms
from torchvision import datasets

from scipy.ndimage import uniform_filter

from scipy.stats import gennorm
from scipy.special import gamma

import torch.nn.functional as F
from scipy.stats import norm
import cv2

def l2_pooling(feats, kernel_size, stride=1, padding=0):
    squared_feats = feats ** 2
    pooled_feats = torch.nn.functional.avg_pool2d(squared_feats.unsqueeze(0), kernel_size, stride=stride, padding=padding)
    return torch.sqrt(pooled_feats.squeeze(0))

def fit_gaussian_distribution(tensor):
    """
    Fit a Gaussian distribution to the tensor, return the mean and standard deviation.
    """
    mean = tensor.mean().item()
    std = tensor.std().item()
    return mean, std

def compute_saliency_map(features, mean, std):
    saliency_maps = []
    p = 1
    for i, feat in enumerate(features):
        # fit gaussian to each filter response (10 filters in total) --> compute joint prob (product) of all
        p *= norm.pdf(feat.detach().cpu().numpy(), loc=mean, scale=std).reshape(feat.shape)
    # compute 1/P(f) --- adding 1e-5 to avoid division by 0
    saliency_map = 1 / (p + 1e-5)
    saliency_map /= np.sum(saliency_map)
    
    # Apply Gaussian smoothing
    saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)
    return saliency_map

def resize_saliency_map(saliency_map, original_shape):
    saliency_map_tensor = torch.tensor(saliency_map).unsqueeze(0).unsqueeze(0)
    resized_saliency_map = F.interpolate(saliency_map_tensor, size=original_shape, mode='bilinear', align_corners=True)
    resized_saliency_map = resized_saliency_map.squeeze().numpy()/np.sum(resized_saliency_map.squeeze().numpy())
    return resized_saliency_map

# seed everything
seed =0 
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

pretrained_folder = "output/old/resnet18_zca_eps0.0005/"
outchannels = 10
save_dir = os.path.join(pretrained_folder, "saliency_maps")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

### Load args
with open(pretrained_folder+"args.json", "r") as f:
    args = json.load(f)

### Load dataset
data_path = args["data_path"]
print('data_path:', data_path)
mean=[0.485, 0.456, 0.406]
std=[1.0, 1.0, 1.0]
transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

### Load model
model = eval(args["model_name"])(num_classes=args["num_classes"], conv0_flag=True, conv0_outchannels=outchannels)
pretranined_model = args["model_name"] + '_best.pth'
model_state_dict = torch.load(os.path.join(pretrained_folder,pretranined_model))
model.load_state_dict(model_state_dict)
model.to('cuda')

### Get one image
idx=5
batch_image, batch_label = next(iter(train_loader))
image = batch_image[0+idx:1+idx]

### Get unorm image, plot and save
unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
unnorm_image = unnorm_image.squeeze().numpy()
unnorm_image = np.moveaxis(unnorm_image, 0, -1)

plt.figure()
plt.imshow(unnorm_image)
plt.axis('off')
plt.savefig(os.path.join(save_dir,f"image_{idx}.png"), bbox_inches='tight')
plt.close()

### Get feature maps after conv0 layer (absolute values)
model.eval()
feats = model.conv0(image.to('cuda')).squeeze()
abs_feats = feats.abs()

### Plot all feature maps
num_feats = feats.shape[0]
figsize=(3*(num_feats/2), 5)
plt.figure(figsize=figsize)
for j in range(feats.shape[0]):
    plt.subplot(1,num_feats,j+1)
    plt.imshow(feats[j].cpu().detach().numpy())
    plt.axis('off')
plt.savefig(os.path.join(save_dir,f"image_{idx}_conv0_feature_maps.png"), bbox_inches='tight')
plt.close()

### L2 pooling
pooled_l2_feats = l2_pooling(abs_feats, kernel_size=3)

### Fit gaussian distribution
mean_l2, std_l2 = fit_gaussian_distribution(pooled_l2_feats)

### Compute saliency map
saliency_map_l2 = compute_saliency_map(pooled_l2_feats, mean_l2, std_l2)

### Resize saliency map to original image size
original_shape = (224, 224)
resized_saliency_map_l2 = resize_saliency_map(saliency_map_l2, original_shape)
print('l2 pooled map: ', resized_saliency_map_l2.shape)

### Plot saliency map
plt.figure()
plt.imshow(unnorm_image.mean(2), cmap='gray')
plt.imshow(resized_saliency_map_l2, cmap='jet', alpha=0.5, interpolation='nearest')
plt.axis('off')
plt.savefig(os.path.join(save_dir,f"image_{idx}_saliency_map_l2_gauss.png"), bbox_inches='tight')
plt.close()


