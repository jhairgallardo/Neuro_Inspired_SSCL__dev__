import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from resnet_gn_mish import *

from torchvision import transforms
from torchvision import datasets

from scipy.ndimage import uniform_filter

from scipy.stats import gennorm
from scipy.special import gamma, gammainc

import torch.nn.functional as F
from scipy.stats import norm
import cv2
from multiprocessing import Pool, cpu_count

import json, os
import numpy as np
import matplotlib.pyplot as plt

def fit_gennorm(feature_map):
    return gennorm.fit(feature_map)
    
def fit_gennorm_to_batch(feats):
    batch_size, num_filters, height, width = feats.shape
    
    # Flatten the feature maps for all filters in the batch
    flattened_feats = feats.view(batch_size * height * width, num_filters).cpu().numpy()
    # print(flattened_feats.shape)   # (..., 10)

    with Pool(processes=cpu_count()) as pool:
        ggd_params = pool.map(fit_gennorm, [flattened_feats[:, j] for j in range(num_filters)])

    return ggd_params

def compute_saliency_map_ggd_batch(features, ggd_params, weighted=False):
    features = features.to(torch.float64)
    batch_size, num_filters, height, width = features.shape

    if weighted:
        # Perform parametric activation on f
        improved_feats = torch.zeros_like(features, dtype=torch.float64)
        
        for i in range(num_filters):
            theta, loc, sigma = ggd_params[i]
            theta_inv = 1.0 / theta
            
            # Calculate the incomplete gamma function for the current dimension
            gamma_incomplete = gammainc(theta_inv, (torch.abs(features[:, i, :, :]) ** theta) * (sigma ** -theta))
            
            # Calculate the Gamma function for θ_i^(-1)
            gamma_func = gamma(theta_inv)
            
            # Calculate the improved features for the current dimension
            improved_feats[:, i, :, :] = gamma_incomplete / gamma_func

    # Compute the joint probability of all filters for each image in the batch
    saliency_maps = []
    for b in range(batch_size):
        p = 1
        for i in range(num_filters):
            theta, loc, sigma = ggd_params[i]
            if weighted:
                p *= gennorm.pdf(improved_feats[b, i, :, :].flatten(), theta, loc, sigma).reshape(height, width)
            else:
                p *= gennorm.pdf(features[b, i, :, :].flatten(), theta, loc, sigma).reshape(height, width)
        saliency_map = 1 / (p + 1e-5)
        saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)
        saliency_map /= np.sum(saliency_map)
        saliency_maps.append(saliency_map)
    saliency_maps = np.array(saliency_maps)
    return torch.tensor(saliency_maps)

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

pretrained_folder = "output/resnet18_barlowtwins_zca6filters_kernel7_eps0.01/"
zca_outchannels = 6
zca_kernel_size = 3
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

### Get batch of images
batch_image, batch_label = next(iter(train_loader))


### Load model
model = eval(args["model_name"])(num_classes=args["num_classes"], conv0_flag=True, conv0_outchannels=zca_outchannels, conv0_kernel_size=zca_kernel_size)
pretranined_model = args["model_name"] + '_best.pth'
model_state_dict = torch.load(os.path.join(pretrained_folder,pretranined_model))
model.load_state_dict(model_state_dict)
model.to('cuda')


### Get feature maps after conv0 layer (absolute values)
batch_feats = model.conv0(batch_image.to('cuda')).squeeze()
batch_abs_feats = batch_feats.abs()


### Get ggd params for the batch
batch_ggd_params = fit_gennorm_to_batch(batch_abs_feats.cpu().detach())

### Compute saliency maps for the batch
weighted_batch_saliency_maps = compute_saliency_map_ggd_batch(batch_abs_feats.detach().cpu(), batch_ggd_params, weighted=True)
batch_saliency_maps = compute_saliency_map_ggd_batch(batch_abs_feats.detach().cpu(), batch_ggd_params, weighted=False)

### Plot image and saliency map for one image in the batch
idx = 5 # 5
image = batch_image[0+idx:1+idx]

# Get unorm image, saliency map for that image, plot and save
unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
unnorm_image = unnorm_image.squeeze().numpy()
unnorm_image = np.moveaxis(unnorm_image, 0, -1)

weighted_saliencymap_image = weighted_batch_saliency_maps[idx].cpu().numpy()
batch_saliency_maps = batch_saliency_maps[idx].cpu().numpy()


# plot
plt.figure(figsize=(30, 10))

plt.subplot(1,4,1)
plt.imshow(unnorm_image)
plt.title('Original', fontsize=12)
plt.axis('off')

# plot mean of batch_abs of idx image
plt.subplot(1,4,2)
plt.imshow(batch_abs_feats[idx].mean(0).cpu().detach().numpy())
plt.title(f'Mean Abs feat', fontsize=12)
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(unnorm_image.mean(2), cmap='gray')
plt.imshow(weighted_saliencymap_image, cmap='jet', alpha=0.5, interpolation='nearest')
plt.title(f'Weighted Batch GGD', fontsize=12)
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(unnorm_image.mean(2), cmap='gray')
plt.imshow(batch_saliency_maps, cmap='jet', alpha=0.5, interpolation='nearest')
plt.title(f'Batch GGD', fontsize=12)
plt.axis('off')

plt.savefig(os.path.join(save_dir,f"image_{idx}_saliency.png"), bbox_inches='tight', dpi=300)
plt.close()


# Plot all feature maps of idx
feats = batch_feats[idx]
num_feats = feats.shape[0]
figsize=(6*(num_feats/2), 10)
plt.figure(figsize=figsize)
for j in range(feats.shape[0]):
    plt.subplot(1,num_feats,j+1)
    plt.imshow(feats[j].cpu().detach().numpy())
    plt.axis('off')
plt.savefig(os.path.join(save_dir,f"image_{idx}_zcafeats.png"), bbox_inches='tight', dpi=300)
plt.close()









# ### Get feature maps after conv0 layer (absolute values)
# model.eval()
# feats = model.conv0(image.to('cuda')).squeeze()
# abs_feats = feats.abs()

# ### Plot all feature maps
# num_feats = feats.shape[0]
# figsize=(3*(num_feats/2), 5)
# plt.figure(figsize=figsize)
# for j in range(feats.shape[0]):
#     plt.subplot(1,num_feats,j+1)
#     plt.imshow(feats[j].cpu().detach().numpy())
#     plt.axis('off')
# plt.savefig(os.path.join(save_dir,f"image_{idx}_conv0_feature_maps.png"), bbox_inches='tight')
# plt.close()

# ### L2 pooling
# pooled_l2_feats = l2_pooling(abs_feats, kernel_size=3)

# ### Fit gaussian distribution
# mean_l2, std_l2 = fit_gaussian_distribution(pooled_l2_feats)

# ### Compute saliency map
# saliency_map_l2 = compute_saliency_map(pooled_l2_feats, mean_l2, std_l2)

# ### Resize saliency map to original image size
# original_shape = (224, 224)
# resized_saliency_map_l2 = resize_saliency_map(saliency_map_l2, original_shape)
# print('l2 pooled map: ', resized_saliency_map_l2.shape)

# ### Plot saliency map
# plt.figure()
# plt.imshow(unnorm_image.mean(2), cmap='gray')
# plt.imshow(resized_saliency_map_l2, cmap='jet', alpha=0.5, interpolation='nearest')
# plt.axis('off')
# plt.savefig(os.path.join(save_dir,f"image_{idx}_saliency_map_l2_gauss.png"), bbox_inches='tight')
# plt.close()


