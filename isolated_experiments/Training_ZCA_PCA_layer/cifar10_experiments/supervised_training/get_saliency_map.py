import json, os
import numpy as np
import matplotlib.pyplot as plt

import torch
from vgg import *

from torchvision import transforms
from torchvision import datasets

from scipy.ndimage import uniform_filter

pretrained_folder = "output/vgg11_bn_cifar10_zca_eps0.0005/"
pca_run=False
zca_run=False
if 'pca' in pretrained_folder:
    outchannels = 54
    pca_run = True
elif 'zca' in pretrained_folder:
    outchannels = 10
    zca_run = True

### Load args
with open(pretrained_folder+"args.json", "r") as f:
    args = json.load(f)

### Load model
model = eval(args["model_name"])(num_classes=args["num_classes"], conv0_flag=True, conv0_outchannels=outchannels)
pretranined_model = args["model_name"] + '_cifar10_best.pth'
model_state_dict = torch.load(os.path.join(pretrained_folder,pretranined_model))
model.load_state_dict(model_state_dict)

### Load dataset
data_path = args["data_path"]
resolution = args["resolution"]
mean=[0.4914, 0.4822, 0.4465]
std=[0.2470, 0.2435, 0.2615]
if zca_run: std=[1.0, 1.0, 1.0]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std),
     ])

val_dataset = datasets.CIFAR10(data_path, train=False, transform=transform, download=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

### Get one image
i=1
batch_image, batch_label = next(iter(val_loader))
image = batch_image[0+i:1+i]

### Plot image
unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
unnorm_image = unnorm_image.squeeze().numpy()
unnorm_image = np.moveaxis(unnorm_image, 0, -1)

### Get feature maps after conv0 layer
model.eval()
feats = model.conv0(image).squeeze()
# mean of the absolute values
mean_feats = feats.abs().mean(0)

### Plot all feature maps and the sum
num_filters = feats.shape[0]
if pca_run:
    figsize=(3*(num_filters/2), 5)
    plt.figure(figsize=figsize)
for j in range(feats.shape[0]):
    if zca_run: plt.subplot(1,10,j+1)
    elif pca_run: plt.subplot(2,27,j+1)
    plt.imshow(feats[j].detach().numpy())
    plt.axis('off')
plt.savefig(os.path.join(pretrained_folder,f"conv0_feature_maps.png"), bbox_inches='tight')
plt.close()

# ### Get saliency map on mean_feats
# window_size = 5  # You can adjust this size based on your specific needs
# smoothed_feats = uniform_filter(mean_feats.detach().numpy(), size=window_size)
# probability_map_feats = smoothed_feats / np.sum(smoothed_feats)

### Create a convolutional layer for mean filtering in PyTorch
window_size = 5 # You can adjust this size based on your specific needs
# Prepare the kernel
kernel = torch.ones((1, 1, window_size, window_size)) / (window_size ** 2)
kernel = kernel.to(mean_feats.device)  # Move kernel to the correct device
# Apply the convolution
mean_feats_aux = mean_feats.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
smoothed_feats = torch.nn.functional.conv2d(mean_feats_aux, kernel, padding=window_size//2)
smoothed_feats = smoothed_feats.squeeze()  # Remove unnecessary dimensions
probability_map_feats = (smoothed_feats / torch.sum(smoothed_feats)).cpu().detach().numpy()

plt.figure(figsize=(24, 6))
plt.subplot(1, 3, 1)
plt.imshow(unnorm_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mean_feats.detach().numpy())
plt.title('Mean abs feats')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(probability_map_feats, cmap='hot')
plt.title('Probability Heatmap')
plt.axis('off')
plt.colorbar()

plt.savefig(os.path.join(pretrained_folder,"conv0_saliency_map.png"), bbox_inches='tight')

print('END')