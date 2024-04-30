import json, os
import numpy as np
import matplotlib.pyplot as plt

import torch
from resnet import *

from torchvision import transforms
from torchvision import datasets

from scipy.ndimage import uniform_filter

pretrained_folder = "output/resnet18_zca_eps0.0005/"
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
pretranined_model = args["model_name"] + '_best.pth'
model_state_dict = torch.load(os.path.join(pretrained_folder,pretranined_model))
model.load_state_dict(model_state_dict)

### Load dataset
data_path = args["data_path"]
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
if zca_run: std=[1.0, 1.0, 1.0]
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])

val_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

### Get one image
i=7 #3 #7 #100 #110
batch_image, batch_label = next(iter(val_loader))
image = batch_image[0+i:1+i]

### Plot image
unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
unnorm_image = unnorm_image.squeeze().numpy()
unnorm_image = np.moveaxis(unnorm_image, 0, -1)

### Get feature maps after conv0 layer
model.to('cuda')
model.eval()
feats = model.conv0(image.to('cuda')).squeeze()
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
    plt.imshow(feats[j].cpu().detach().numpy())
    plt.axis('off')
plt.savefig(os.path.join(pretrained_folder,f"conv0_feature_maps.png"), bbox_inches='tight')
plt.close()

### Create a convolutional layer for mean filtering in PyTorch
window_size = 32 # You can adjust this size based on your specific needs ##################################################
# The window size will be the crop size once I implement smart crops
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
plt.imshow(mean_feats.cpu().detach().numpy())
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