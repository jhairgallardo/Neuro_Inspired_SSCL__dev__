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

    with Pool(processes=int(cpu_count()/2)) as pool:
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
    saliency_maps = torch.ones((batch_size, height, width), dtype=torch.float64)
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
        saliency_maps[b] = torch.tensor(saliency_map)
    saliency_maps = saliency_maps.to(torch.float32)
    return torch.tensor(saliency_maps)

def mean_pooling_batch(feats, kernel_size, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    pooled_feats = F.avg_pool2d(feats, kernel_size, stride=stride, padding=padding)
    return pooled_feats

def l2_pooling_batch(feats, kernel_size, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    squared_feats = feats ** 2
    pooled_feats = F.avg_pool2d(squared_feats, kernel_size, stride=stride, padding=padding)#, divisor_override=1) # divisor_override=1 to do sum pooling
    # when using divisor_override=1, it causes the saliency map to flip probabilities for some reason. numerical issue
    return torch.sqrt(pooled_feats)

def resize_saliency_map_batch(saliency_maps, original_shape=(224,224)):
    # saliency_maps: tensor of shape (batch_size, height, width)
    saliency_maps_tensor = saliency_maps.unsqueeze(1)  # Add channel dimension
    resized_saliency_maps = F.interpolate(saliency_maps_tensor, size=original_shape, mode='bilinear', align_corners=True)
    resized_saliency_maps = resized_saliency_maps.squeeze(1)  # Remove channel dimension

    # Normalize each saliency map in the batch
    batch_sums = resized_saliency_maps.view(resized_saliency_maps.shape[0], -1).sum(dim=1)  # Compute sum of each saliency map
    resized_saliency_maps /= batch_sums.view(-1, 1, 1)  # Normalize using broadcasting
    return resized_saliency_maps

# seed everything
seed =0 
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

pretrained_folder = "output/resnet18_barlowtwins_zca6filters_kernel3_eps0.01/"
zca_outchannels = 6
zca_kernel_size = 3
pool_kernel_size = 32
stride = 1
weighted = True
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


### Get feature maps after conv0 layer (absolute values) (also the mean and L2 pool versions)
model.eval()
with torch.no_grad():
    batch_feats = model.conv0(batch_image.to('cuda')).squeeze()
batch_abs_feats = batch_feats.abs().detach().cpu()
batch_meanpool_feats = mean_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)
batch_l2pool_feats = l2_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)


### Get ggd params for the batch
ggd_params = fit_gennorm_to_batch(batch_abs_feats)
meanpool_ggd_params = fit_gennorm_to_batch(batch_meanpool_feats)
l2pool_ggd_params = fit_gennorm_to_batch(batch_l2pool_feats)


### Compute saliency maps for the batch
batch_saliency_maps = compute_saliency_map_ggd_batch(batch_abs_feats, ggd_params, weighted=weighted)
batch_meanpool_saliency_maps = compute_saliency_map_ggd_batch(batch_meanpool_feats, meanpool_ggd_params, weighted=weighted)
batch_l2pool_saliency_maps = compute_saliency_map_ggd_batch(batch_l2pool_feats, l2pool_ggd_params, weighted=weighted)

### Resize saliency maps to original image size (only meanpool and L2pool need this)
original_shape = (224, 224)
batch_meanpool_saliency_maps = resize_saliency_map_batch(batch_meanpool_saliency_maps, original_shape)
batch_l2pool_saliency_maps = resize_saliency_map_batch(batch_l2pool_saliency_maps, original_shape)


### Plot image and saliency map for one image in the batch
for idx in range(50): # [1,8,12,13,14,16,20]
# idx = 5 # 5 (bad) 8 (good) 23 (bad) 24 (nice)
    image = batch_image[0+idx:1+idx]

    # Get unorm image, saliency map for that image, plot and save
    unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
    unnorm_image = unnorm_image.squeeze().numpy()
    unnorm_image = np.moveaxis(unnorm_image, 0, -1)

    saliencymap_image = batch_saliency_maps[idx].numpy()
    saliencymap_meanpool_image = batch_meanpool_saliency_maps[idx].numpy()
    saliencymap_l2pool_image = batch_l2pool_saliency_maps[idx].numpy()



    # plot
    plt.figure(figsize=(40, 8))

    # plot original image
    plt.subplot(1,5,1)
    plt.imshow(unnorm_image)
    plt.title('Original', fontsize=20)
    plt.axis('off')

    # plot mean of batch_abs of idx image
    plt.subplot(1,5,2)
    plt.imshow(batch_abs_feats[idx].mean(0).cpu().detach().numpy())
    plt.title(f'Mean Abs feat', fontsize=20)
    plt.axis('off')

    # plot saliency map
    plt.subplot(1,5,3)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_image, cmap='jet', alpha=0.5, interpolation='nearest')
    if weighted:
        plt.title(f'Weighted GGD', fontsize=20)
    else:
        plt.title(f'GGD', fontsize=20)
    plt.axis('off')

    # plot saliency map (mean pool)
    plt.subplot(1,5,4)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_meanpool_image, cmap='jet', alpha=0.5, interpolation='nearest')
    if weighted:
        plt.title(f'Weighted GGD (mean pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (mean pool)', fontsize=20)
    plt.axis('off')

    # plot saliency map (L2 pool)
    plt.subplot(1,5,5)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_l2pool_image, cmap='jet', alpha=0.5, interpolation='nearest')
    if weighted:
        plt.title(f'Weighted GGD (L2 pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (L2 pool)', fontsize=20)
    plt.axis('off')

    # save plot
    if weighted:
        plt.savefig(os.path.join(save_dir,f"image_{idx}_saliency_weighted_pool_kernel_{pool_kernel_size}.png"), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(save_dir,f"image_{idx}_saliency_pool_kernel_{pool_kernel_size}.png"), bbox_inches='tight', dpi=300)
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



