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

import json, os, math
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

# def compute_saliency_map_ggd_batch(features, ggd_params, weighted=False):
#     features = features.to(torch.float64)
#     batch_size, num_filters, height, width = features.shape

#     # Compute the joint probability of all filters for each image in the batch
#     saliency_maps = torch.ones((batch_size, height, width), dtype=torch.float64)
#     for i in range(num_filters):
#         theta, loc, sigma = ggd_params[i]
#         if weighted:
#             theta_inv = 1.0 / theta
#             gamma_incomplete = gammainc(theta_inv, (torch.abs(features[:, i, :, :]) ** theta) * (sigma ** -theta))
#             gamma_func = gamma(theta_inv)
#             improved_feats = gamma_incomplete / gamma_func
#             p = gennorm.pdf(improved_feats.flatten(), theta, loc, sigma)
#         else:
#             p = gennorm.pdf(features[:, i, :, :].flatten(), theta, loc, sigma)
#         p = p.reshape(batch_size, height, width)
#         saliency_maps *= torch.tensor(p, dtype=torch.float64)
#     saliency_maps = 1 / (saliency_maps + 1e-5)
#     saliency_maps /= saliency_maps.sum(dim=(1, 2), keepdim=True)
#     saliency_maps = saliency_maps.to(torch.float32)
#     return saliency_maps




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
    return saliency_maps



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

def smart_crop_batch(feature_maps, scale = [0.08, 1.0], ratio = [3.0/4.0, 4.0/3.0], crop_size=None):
    batch_size, height, width = feature_maps.shape
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio))

    all_crops = torch.zeros(batch_size, 4, dtype=torch.int32)
    
    for b in range(batch_size):
        saliency_map = feature_maps[b].cpu().numpy()

        # Get w_crop and h_crop (size of crop)
        for i in range(10):
            if crop_size is None:
                target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
                aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
                h_crop = int(round(math.sqrt(target_area / aspect_ratio)))
                w_crop = int(round(math.sqrt(target_area * aspect_ratio)))
            else:
                h_crop, w_crop = crop_size
            
            if 0 < h_crop <= height and 0 < w_crop <= width:
                break
            elif i == 9: # if it fails 10 times, then just take the whole image
                h_crop = height
                w_crop = width

        if  h_crop == height and w_crop == width: # if the whole image is the crop, save time by not sampling position
            all_crops[b] = torch.tensor([0, 0, h_crop, w_crop])
            continue

        # Get idx_x and idx_y (top left corner of crop). Use saliency map as probability distribution
        for i in range(10):
            idx = np.random.choice(np.arange(len(saliency_map.flatten())), p=saliency_map.flatten())
            idx_cy, idx_cx = np.unravel_index(idx, saliency_map.shape) # center of crop
            
            # idx_cy, idx_cx = np.unravel_index(np.argmax(saliency_map.flatten()), saliency_map.shape) # center of crop (get position with highest saliency)
            
            # if part of the crop falls outside the image, then move the center of the crop.
            # It makes sure that the sampled center is within the image (not necesarily the center)
            if idx_cy + h_crop // 2 > height:
                diff_cy = idx_cy + h_crop // 2 - height
                idx_cy -= diff_cy
            if idx_cy - h_crop // 2 < 0:
                diff_cy = h_crop // 2 - idx_cy
                idx_cy += diff_cy
            if idx_cx + w_crop // 2 > width:
                diff_cx = idx_cx + w_crop // 2 - width
                idx_cx -= diff_cx
            if idx_cx - w_crop // 2 < 0:
                diff_cx = w_crop // 2 - idx_cx
                idx_cx += diff_cx
            
            idx_y = idx_cy - h_crop // 2
            idx_x = idx_cx - w_crop // 2


            # make sure the complete crop is within the image (take into accounr the top left corner and the size of the crop)
            if 0 <= idx_x and idx_x + w_crop <= width and 0 <= idx_y and idx_y + h_crop <= height:
                break
            elif i == 9: # if it fails 10 times, then just take the center of the image
                idx_cx = width // 2
                idx_cy = height // 2
                idx_x = idx_cx - w_crop // 2
                idx_y = idx_cy - h_crop // 2
        
        all_crops[b] = torch.tensor([idx_x, idx_y, w_crop, h_crop])
    
    return all_crops 



# seed everything
seed = 0
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
pool_kernel_size = 8
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
zca_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
zca_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=512, shuffle=True)

### Get batch of images
batch_zca_image, _ = next(iter(zca_loader))

### Load model
model = eval(args["model_name"])(num_classes=args["num_classes"], conv0_flag=True, conv0_outchannels=zca_outchannels, conv0_kernel_size=zca_kernel_size)
pretranined_model = args["model_name"] + '_best.pth'
model_state_dict = torch.load(os.path.join(pretrained_folder,pretranined_model))
model.load_state_dict(model_state_dict)
model.to('cuda')
model.eval()

### Get GGD params for a batch_zca_image (no pooling, meanpool and L2 pool)
with torch.no_grad():
    batch_zca_feats = model.conv0(batch_zca_image.to('cuda')).squeeze()
batch_abs_zca_feats = batch_zca_feats.abs().detach().cpu()
batch_meanpool_zca_feats = mean_pooling_batch(batch_abs_zca_feats, kernel_size=pool_kernel_size, stride=stride)
batch_l2pool_zca_feats = l2_pooling_batch(batch_abs_zca_feats, kernel_size=pool_kernel_size, stride=stride)

ggd_params = fit_gennorm_to_batch(batch_abs_zca_feats)
meanpool_ggd_params = fit_gennorm_to_batch(batch_meanpool_zca_feats)
l2pool_ggd_params = fit_gennorm_to_batch(batch_l2pool_zca_feats)





# ### Load a training batch
# trainig_transform = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)
#             ])
# training_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=trainig_transform)
# training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
# batch_image, _ = next(iter(training_loader))
batch_image = batch_zca_image

### Get features for the batch
with torch.no_grad():
    batch_feats = model.conv0(batch_image.to('cuda')).squeeze()
batch_abs_feats = batch_feats.abs().detach().cpu()
batch_meanpool_feats = mean_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)
batch_l2pool_feats = l2_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)

### Compute saliency maps for the batch
batch_saliency_maps = compute_saliency_map_ggd_batch(batch_abs_feats, ggd_params, weighted=weighted)
batch_meanpool_saliency_maps = compute_saliency_map_ggd_batch(batch_meanpool_feats, meanpool_ggd_params, weighted=weighted)
batch_l2pool_saliency_maps = compute_saliency_map_ggd_batch(batch_l2pool_feats, l2pool_ggd_params, weighted=weighted)

### Resize saliency maps to original image size (only meanpool and L2pool need this)
original_shape = (224, 224)
batch_meanpool_saliency_maps = resize_saliency_map_batch(batch_meanpool_saliency_maps, original_shape)
batch_l2pool_saliency_maps = resize_saliency_map_batch(batch_l2pool_saliency_maps, original_shape)

### Get crops for the batch
batch_crops = smart_crop_batch(batch_saliency_maps)#, scale=[0.08, 0.09])
batch_meanpool_crops = smart_crop_batch(batch_meanpool_saliency_maps)#, scale=[0.08, 0.09])
batch_l2pool_crops = smart_crop_batch(batch_l2pool_saliency_maps)#, scale=[0.08, 0.09])

# TODO: plot resized crops



### Plot image and saliency map for one image in the batch
for idx in range(50): # [1,8,12,13,14,16,20]
# idx = 5 # 5 (bad) 8 (good) 23 (bad) 24 (nice)
    image = batch_image[0+idx:1+idx]

    unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
    unnorm_image = unnorm_image.squeeze().numpy()
    unnorm_image = np.moveaxis(unnorm_image, 0, -1)

    zca_image = batch_feats[idx,:3,:,:].cpu().detach().numpy()
    zca_image = (zca_image - zca_image.min()) / (zca_image.max() - zca_image.min())
    zca_image = np.moveaxis(zca_image, 0, -1)

    saliencymap_image = batch_saliency_maps[idx].numpy()
    saliencymap_meanpool_image = batch_meanpool_saliency_maps[idx].numpy()
    saliencymap_l2pool_image = batch_l2pool_saliency_maps[idx].numpy()

    # plot
    plt.figure(figsize=(24, 24))

    # plot original image
    plt.subplot(3,3,1)
    plt.imshow(unnorm_image)
    plt.title('Original', fontsize=20)
    plt.axis('off')

    # plot zca version
    plt.subplot(3,3,2)
    plt.imshow(zca_image)
    plt.title('ZCA (3 first channels)', fontsize=20)
    plt.axis('off')

    # plot mean of batch_abs of idx image
    plt.subplot(3,3,3)
    plt.imshow(batch_abs_feats[idx].mean(0).cpu().detach().numpy())
    plt.title(f'Mean Abs feat', fontsize=20)
    plt.axis('off')

    # plot saliency map
    plt.subplot(3,3,4)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_image, cmap='jet', alpha=0.5, interpolation='nearest')
    if weighted:
        plt.title(f'Weighted GGD', fontsize=20)
    else:
        plt.title(f'GGD', fontsize=20)
    plt.axis('off')

    # plot saliency map (mean pool)
    plt.subplot(3,3,5)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_meanpool_image, cmap='jet', alpha=0.5, interpolation='nearest')
    if weighted:
        plt.title(f'Weighted GGD (mean pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (mean pool)', fontsize=20)
    plt.axis('off')

    # plot saliency map (L2 pool)
    plt.subplot(3,3,6)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_l2pool_image, cmap='jet', alpha=0.5, interpolation='nearest')
    if weighted:
        plt.title(f'Weighted GGD (L2 pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (L2 pool)', fontsize=20)
    plt.axis('off')

    # plot image, saliency map, and a bounding box for the crop
    plt.subplot(3,3,7)
    plt.imshow(unnorm_image)
    plt.imshow(saliencymap_image, cmap='jet', alpha=0.5, interpolation='nearest')
    plt.gca().add_patch(plt.Rectangle((batch_crops[idx][0], batch_crops[idx][1]), batch_crops[idx][2], batch_crops[idx][3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.title('Crop', fontsize=20)
    plt.axis('off')

    # plot image, saliency map, and a bounding box for the crop (mean pool)
    plt.subplot(3,3,8)
    plt.imshow(unnorm_image)
    plt.imshow(saliencymap_meanpool_image, cmap='jet', alpha=0.5, interpolation='nearest')
    plt.gca().add_patch(plt.Rectangle((batch_meanpool_crops[idx][0], batch_meanpool_crops[idx][1]), batch_meanpool_crops[idx][2], batch_meanpool_crops[idx][3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.title('Crop (mean pool)', fontsize=20)
    plt.axis('off')

    # plot image, saliency map, and a bounding box for the crop (L2 pool)
    plt.subplot(3,3,9)
    plt.imshow(unnorm_image)
    plt.imshow(saliencymap_l2pool_image, cmap='jet', alpha=0.5, interpolation='nearest')
    plt.gca().add_patch(plt.Rectangle((batch_l2pool_crops[idx][0], batch_l2pool_crops[idx][1]), batch_l2pool_crops[idx][2], batch_l2pool_crops[idx][3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.title('Crop (L2 pool)', fontsize=20)
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



