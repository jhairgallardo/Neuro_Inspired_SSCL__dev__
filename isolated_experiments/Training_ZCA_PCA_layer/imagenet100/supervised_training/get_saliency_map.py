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

# seed everything
seed =0 
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

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

### Load dataset
data_path = args["data_path"]
print('data_path:', data_path)
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
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=False)

### Load model
model = eval(args["model_name"])(num_classes=args["num_classes"], conv0_flag=True, conv0_outchannels=outchannels)
pretranined_model = args["model_name"] + '_best.pth'
model_state_dict = torch.load(os.path.join(pretrained_folder,pretranined_model))
model.load_state_dict(model_state_dict)

### Get one image
idx=389  #1830 #1010 #723 #389
batch_image, batch_label = next(iter(val_loader))
image = batch_image[0+idx:1+idx]

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

### Create a convolutional layer for mean filtering
window_size = 16 # 32, 64 # The window size will be the crop size once I implement smart crops.
# Prepare the kernel
kernel = torch.ones((1, 1, window_size, window_size)) / (window_size ** 2)
kernel = kernel.to(mean_feats.device)  # Move kernel to the correct device
# Apply the convolution
mean_feats_aux = mean_feats.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
mean_feats_aux = torch.nn.functional.pad(mean_feats_aux, (window_size//2, window_size//2, window_size//2, window_size//2), mode='constant', value=0)
smoothed_feats = torch.nn.functional.conv2d(mean_feats_aux, kernel) # Saliency map
smoothed_feats = smoothed_feats.squeeze()  # Remove unnecessary dimensions
probability_map_feats = (smoothed_feats / torch.sum(smoothed_feats)).cpu().detach().numpy() # Probability map

plt.figure(figsize=(24, 6))
plt.subplot(1, 3, 1)
plt.imshow(unnorm_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mean_feats.cpu().detach().numpy())
plt.title('Mean abs feats')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(unnorm_image.mean(2), cmap='gray')
max_idx = np.unravel_index(probability_map_feats.argmax(), probability_map_feats.shape)

plt.imshow(probability_map_feats, cmap='jet', alpha=0.4)
plt.title(f'Probability Heatmap (Crop size: {window_size})')
plt.axis('off')

plt.savefig(os.path.join(pretrained_folder,f"conv0_saliency_map_idx{idx}_window{window_size}.png"), bbox_inches='tight')
plt.close()

print('END')