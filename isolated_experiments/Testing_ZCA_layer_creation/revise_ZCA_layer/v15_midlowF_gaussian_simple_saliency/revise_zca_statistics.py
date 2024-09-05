import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import transforms
from torchvision import datasets

import torch.backends.cudnn as cudnn
import numpy as np
import random
import einops
from copy import deepcopy

import matplotlib.pyplot as plt
from PIL import ImageOps, ImageFilter
from scipy.stats import gennorm
import math
from multiprocessing import Pool, cpu_count

def calculate_ZCA_conv0_weights(imgs, kernel_size=3, zca_epsilon=1e-6, save_dir=None):

    # save imgs channel mean in json format
    mean = imgs.mean(dim=(0,2,3)).tolist()
    with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f:
        json.dump(mean, f)
    
    # extract Patches 
    patches = extract_patches(imgs, kernel_size, step=kernel_size)

    # get weight with ZCA
    weight = get_filters(patches, zca_epsilon=zca_epsilon)

    return weight

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, zca_epsilon=1e-6):
    _, n_channels, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # data is a d x N
    data = data.double()

    # Compute ZCA
    W = ZCA(data, epsilon = zca_epsilon).to(data.device) # shape: k (c h w)

    # Reshape filters
    W = einops.rearrange(W, 'k (c h w) -> k c h w', c=n_channels, h=filt_size, w=filt_size)

    # Pick centered filters only (filter is in the middle of the patch)
    W_center = torch.zeros((n_channels, n_channels, filt_size, filt_size), dtype=W.dtype)
    for i in range(n_channels):
        index = int( (filt_size*filt_size)*(i) + (filt_size*filt_size-1)/2 ) # index of the filter that is in the center of the patch
        W_center[i, :] = W[index, :, :, :]

    # back to single precision
    W_center = W_center.float()

    return W_center

def ZCA(data, epsilon=1e-6):
    # data is a d x N matrix, where d is the dimensionality and N is the number of samples
    C = (data @ data.T) / data.size(1)
    D, V = torch.linalg.eigh(C)
    sorted_indices = torch.argsort(D, descending=True)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    Di = (D + epsilon)**(-0.5)
    zca_matrix = V @ torch.diag(Di) @ V.T
    return zca_matrix

def scaled_filters(zca_layer, imgs, prob_threshold=0.99999, name='maxscaled'):
    # Maxscale the filters
    weight = zca_layer.weight
    weight = weight / torch.amax(weight, keepdims=True)
    zca_layer.weight = torch.nn.Parameter(weight)
    zca_layer.weight.requires_grad = False

    # Get the output of the zca layer
    zca_layer.eval()
    zca_layer_output = zca_layer(imgs)

    # Get CDF on the absolute values of the output
    zca_layer_output = zca_layer_output.abs().flatten().numpy()
    count, bins_count = np.histogram(zca_layer_output, bins=100) 
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    # Find threshold, multiplier, and scale the filters
    threshold = bins_count[1:][np.argmax(cdf>prob_threshold)]
    multiplier = np.arctanh(0.999)/threshold # find value where after a tanh function, everythinng after the threshold will be near 1 (0.999)
    print(f'threshold: {threshold}')
    print(f'multiplier: {multiplier}')
    weight = weight * multiplier

    return weight

def plot_filters_and_hist(filters, name, save_dir):
    # plot filters
    num_filters = filters.shape[0]
    plt.figure(figsize=(5*num_filters,5))
    for i in range(num_filters):
        filter_m = deepcopy(filters[i])
        filter_m = (filter_m - filter_m.min()) / (filter_m.max() - filter_m.min())
        filter_m = filter_m.cpu().numpy().transpose(1,2,0)
        plt.subplot(1,num_filters,i+1)
        plt.imshow(filter_m, vmax=1, vmin=0)
        plt.axis('off')
    plt.savefig(os.path.join(save_dir,f'{name}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    # plot hist of filters
    plt.figure(figsize=(5,5*num_filters))
    for i in range(num_filters):
        filter_m = filters[i]
        plt.subplot(num_filters,1,i+1)
        plt.hist(filter_m.flatten(), label=f'filter {i}')
        plt.legend()
    plt.savefig(os.path.join(save_dir,f'{name}_hist.jpg'), bbox_inches='tight')
    plt.close()

    return None

def plot_zca_layer_output_hist(zca_layer, imgs, name, save_dir):
    zca_layer.eval()
    zca_layer_output = zca_layer(imgs)
    zca_layer_output = zca_layer_output.flatten().cpu().numpy()
    plt.figure(figsize=(18,6))
    plt.subplot(2,1,1)
    plt.hist(zca_layer_output, bins=100)
    plt.ylabel('frequency')
    plt.title(name)
    plt.subplot(2,1,2)
    plt.hist(zca_layer_output, bins=100)
    plt.yscale('log')
    plt.ylabel('log scale frequency')
    plt.savefig(f'{save_dir}/{name}.png', bbox_inches='tight')
    plt.close()
    return None

def plot_channel_cov(zca_layer, imgs, name, save_dir):
    zca_layer.eval()
    zca_layer_output = zca_layer(imgs)
    zca_layer_output = zca_layer_output.permute(0,2,3,1).reshape(-1, zca_layer_output.shape[1])
    cov = torch.cov(zca_layer_output.T)
    plt.imshow(cov)
    plt.colorbar()
    plt.savefig(f'{save_dir}/{name}.png')
    plt.close()
    return None

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
def mean_pooling_batch(feats, kernel_size, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    pooled_feats = F.avg_pool2d(feats, kernel_size, stride=stride, padding=padding)
    return pooled_feats

def l2_pooling_batch(feats, kernel_size, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    squared_feats = feats ** 2
    pooled_feats = F.avg_pool2d(squared_feats, kernel_size, stride=stride, padding=padding, divisor_override=1)
    return torch.sqrt(pooled_feats)

def resize_saliency_map_batch(saliency_maps, original_shape=(224,224)):
    # saliency_maps: tensor of shape (batch_size, height, width)
    saliency_maps_tensor = saliency_maps.unsqueeze(1)  # Add channel dimension
    resized_saliency_maps = F.interpolate(saliency_maps_tensor, size=original_shape, mode='bilinear', align_corners=True)
    resized_saliency_maps = resized_saliency_maps.squeeze(1)  # Remove channel dimension
    return resized_saliency_maps

def smart_crop_batch(saliency_maps, num_crops = 1, scale = [0.08, 1.0], ratio = [3.0/4.0, 4.0/3.0], top_percentage=0.1):
    batch_size, height, width = saliency_maps.shape
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio))
    all_crops = torch.zeros(batch_size, num_crops, 4, dtype=torch.int32)
    # saliency_maps = saliency_maps.cpu().numpy()
    
    for b in range(batch_size):
        saliency_map = saliency_maps[b]
        for k in range(num_crops):
            # Get w_crop and h_crop (size of crop)
            for i in range(10):
                target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
                aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
                h_crop = int(round(math.sqrt(target_area / aspect_ratio)))
                w_crop = int(round(math.sqrt(target_area * aspect_ratio)))
                
                if 0 < h_crop <= height and 0 < w_crop <= width:
                    break
                elif i == 9: # if it fails 10 times, then just take the whole image
                    h_crop = height
                    w_crop = width

            if h_crop == height and w_crop == width: # if the whole image is the crop, save time by not sampling position
                all_crops[b,k] = torch.tensor([0, 0, h_crop, w_crop])
                continue

            # Get idx_x and idx_y (top left corner of crop). Use saliency map as probability distribution
            for i in range(10):
                probabilities = saliency_map.flatten()
                top_number = int(len(probabilities)*top_percentage)
                top_pixels = torch.topk(probabilities, top_number).values
                top_idxs = torch.topk(probabilities, top_number).indices
                inner_idx = top_pixels.multinomial(1).item()
                idx = top_idxs[inner_idx]
                idx_cy, idx_cx = np.unravel_index(idx, saliency_map.shape) # center of crop
                
                # sanity check line: get position with highest saliency
                # idx_cy, idx_cx = np.unravel_index(np.argmax(saliency_map.flatten()), saliency_map.shape) 
                
                # if part of the crop falls outside the image, then move the center of the crop.
                # It makes sure that the sampled center is within the crop (not necesarily the center, but inside the crop)
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

                # make sure the complete crop is within the image (safety check)
                if 0 <= idx_x and idx_x + w_crop <= width and 0 <= idx_y and idx_y + h_crop <= height:
                    break
                elif i == 9: # if it fails 10 times, then just take the center of the image
                    idx_cx = width // 2
                    idx_cy = height // 2
                    idx_x = idx_cx - w_crop // 2
                    idx_y = idx_cy - h_crop // 2
        
            all_crops[b,k] = torch.tensor([idx_x, idx_y, w_crop, h_crop])
    
    return all_crops


def gaussian_blur(weight, kernel_size, sigma):
    def _get_gaussian_kernel1d(kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        return kernel1d


    def _get_gaussian_kernel2d(kernel_size, sigma, dtype, device):
        kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
        kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
        return kernel2d
    
    dtype = weight.dtype if torch.is_floating_point(weight) else torch.float32
    kernel = _get_gaussian_kernel2d([kernel_size,kernel_size], [sigma,sigma], dtype=dtype, device=weight.device)
    kernel = kernel.expand(weight.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    weight_pad = F.pad(weight, padding, mode="replicate")
    weight_gauss_lf = F.conv2d(weight_pad, kernel, groups=weight.shape[-3])

    return weight_gauss_lf


### Seed Everything
seed = 0 
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

### Parameters
batch_aug_type = 'none' # 'none' 'colorjitter' 'grayscale' 'gaussianblur' 'solarization' 'barlowtwins'
conv0_outchannels=18
conv0_kernel_size=7
nimg = 10000 #10000
zca_epsilon = 1e-4 # 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
dataset_name='ImageNet-10'
activation = 'mishtanh' # noact, tanh, mish, softplustanh, relutanh, mishtanh
zca_scale_filter = True
save_dir = f'output'
num_batch_plot=16

save_dir += f'/{batch_aug_type}aug_{conv0_kernel_size}kernerlsize_{conv0_outchannels}channels_{zca_epsilon}eps_{activation}'
if zca_scale_filter:
    save_dir += f'_scaled_replicatepad'
os.makedirs(save_dir, exist_ok=True)

### Data to calculate ZCA layer
mean=[0.485, 0.456, 0.406]
std=[1.0, 1.0, 1.0]

### Calculate ZCA layer main 3 filters (high frequency)
zca_layer = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
zca_dataset = datasets.ImageFolder(root=f"/data/datasets/{dataset_name}/train", transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=nimg, shuffle=True)
zca_input_imgs,_ = next(iter(zca_loader))
weight = calculate_ZCA_conv0_weights(imgs=zca_input_imgs, kernel_size=conv0_kernel_size, zca_epsilon=zca_epsilon, save_dir=save_dir)

### Calculate low frequency filters
weight_lf = gaussian_blur(weight, kernel_size=9, sigma=2.0)
plot_filters_and_hist(weight_lf, name='zca_lowF_filters', save_dir=save_dir)

### Calculate mid frequency filters
weight_mf = gaussian_blur(weight, kernel_size=9, sigma=1.0)
plot_filters_and_hist(weight_mf, name='zca_midF_filters', save_dir=save_dir)
                                

### Scale down filters
if zca_scale_filter:
    # scale down high frequency filters
    aux_conv0 = torch.nn.Conv2d(3, weight.shape[0], kernel_size=conv0_kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
    aux_conv0.weight = torch.nn.Parameter(weight)
    aux_conv0.weight.requires_grad = False
    weight = scaled_filters(aux_conv0, imgs=zca_input_imgs)
    del aux_conv0

    # scale down low frequency filters
    aux_conv0 = torch.nn.Conv2d(3, weight_lf.shape[0], kernel_size=conv0_kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
    aux_conv0.weight = torch.nn.Parameter(weight_lf)
    aux_conv0.weight.requires_grad = False
    weight_lf = scaled_filters(aux_conv0, imgs=zca_input_imgs)
    del aux_conv0

    # scale down mid frequency filters
    aux_conv0 = torch.nn.Conv2d(3, weight_mf.shape[0], kernel_size=conv0_kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
    aux_conv0.weight = torch.nn.Parameter(weight_mf)
    aux_conv0.weight.requires_grad = False
    weight_mf = scaled_filters(aux_conv0, imgs=zca_input_imgs)
    del aux_conv0

### Concatenate high and low frequency filters
weight = torch.cat((weight, weight_mf, weight_lf), dim=0)

### Add negative filters
if conv0_outchannels>=18:
    weight = torch.cat((weight, -weight), dim=0)

# Define activation function
if activation == 'tanh':
    act = nn.Tanh()
elif activation == 'mish':
    act = nn.Mish()
elif activation == 'softplustanh':
    act = nn.Sequential(nn.Softplus(), nn.Tanh())
elif activation == 'relutanh':
    act = nn.Sequential(nn.ReLU(), nn.Tanh())
elif activation == 'mishtanh':
    act = nn.Sequential(nn.Mish(), nn.Tanh())
elif activation == 'noact':
    act = nn.Identity()

### Define final ZCA layer 
zca_layer = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', padding_mode='replicate', bias=False)
zca_layer.weight = torch.nn.Parameter(weight)
zca_layer.weight.requires_grad = False

### Plot final ZCA layer stats
plot_filters_and_hist(zca_layer.weight, name='zca_filters', save_dir=save_dir)
plot_channel_cov(zca_layer, zca_input_imgs, name='zcainput_channelcov_zca_out', save_dir=save_dir)
plot_zca_layer_output_hist(zca_layer, zca_input_imgs, name='zcainput_zca_out_hist', save_dir=save_dir)

### Define final ZCA layer with activation (plot stats after activation)
zca_layer_act = nn.Sequential(zca_layer, act)
plot_channel_cov(zca_layer_act, zca_input_imgs, name='zcainput_channelcov_zca_act_out', save_dir=save_dir)
plot_zca_layer_output_hist(zca_layer_act, zca_input_imgs, name='zcainput_zca_act_out_hist', save_dir=save_dir)

# put ZCA layer in gpu
zca_layer = zca_layer.cuda()

### Load a batch of data (could have augmentations)
if batch_aug_type == 'colorjitter':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                    saturation=0.2, hue=0.1)],
                            p=1.0
                            ),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
elif batch_aug_type == 'grayscale':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomGrayscale(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
elif batch_aug_type == 'gaussianblur':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        GaussianBlur(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
elif batch_aug_type == 'solarization':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        Solarization(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
elif batch_aug_type == 'none':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
elif batch_aug_type == 'barlowtwins':
    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                    saturation=0.2, hue=0.1)],
                            p=0.8
                            ),
                        transforms.RandomGrayscale(p=0.2),
                        GaussianBlur(p=0.1),
                        Solarization(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
else:
    raise ValueError(f'Augmentation type {batch_aug_type} not recognized')
train_dataset = datasets.ImageFolder(root=f"/data/datasets/{dataset_name}/train", transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
batch_train_images,batch_train_label = next(iter(train_loader))

### Pass batch of data through zca layer (forward pass using cuda)
with torch.no_grad():
    batch_zca_out = zca_layer(batch_train_images.cuda())
    batch_zca_act_out = act(batch_zca_out).cpu().detach()
    batch_zca_out = batch_zca_out.cpu().detach()

### Save input images
imgs = deepcopy(batch_train_images[:num_batch_plot])
imgs = imgs * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
grid = torchvision.utils.make_grid(imgs, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title(f'{batch_aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_imgs.png', bbox_inches='tight')
plt.close()

### Plot ZCA transformed images
batch_zca_out_aux = deepcopy(batch_zca_out[:num_batch_plot, :3, :, :]) # * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
for i in range(num_batch_plot):
    batch_zca_out_aux[i] = ( batch_zca_out_aux[i] - batch_zca_out_aux[i].min() )/ (batch_zca_out_aux[i].max() - batch_zca_out_aux[i].min())
grid = torchvision.utils.make_grid(batch_zca_out_aux, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title(f'{batch_aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_zca_out_imgs.png', bbox_inches='tight')
plt.close()

### Plot the mean of the absolute values of batch_zca_out
batch_zca_out_abs = deepcopy(batch_zca_out[:num_batch_plot].abs())
batch_zca_out_abs_mean = batch_zca_out_abs.mean(1)
batch_zca_out_abs_mean = batch_zca_out_abs_mean.unsqueeze(1)
grid = torchvision.utils.make_grid(batch_zca_out_abs_mean, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid[0])
plt.axis('off')
plt.title(f'{batch_aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_zca_out_mean_abs.png', bbox_inches='tight')
plt.close()

### Plot the mean of the absolute values of the batch_zca_act_out
batch_zca_act_out_abs = deepcopy(batch_zca_act_out[:num_batch_plot].abs())
batch_zca_act_out_abs_mean = batch_zca_act_out_abs.mean(1)
batch_zca_act_out_abs_mean = batch_zca_act_out_abs_mean.unsqueeze(1)
grid = torchvision.utils.make_grid(batch_zca_act_out_abs_mean, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid[0])
plt.axis('off')
plt.title(f'{batch_aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_zca_act_out_mean_abs.png', bbox_inches='tight')
plt.close()

### Get channel covariance matrix of batch
batch_train_images_aux = batch_train_images.permute(0,2,3,1).reshape(-1, 3)
cov = torch.cov(batch_train_images_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/batch_channelcov_matrix.png')
plt.close()

### Get channel covariance matrix of batch zca out
batch_zca_out_aux = batch_zca_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(batch_zca_out_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/batch_channelcov_matrix_zca_out.png')
plt.close()

### Get channel covariance matrix of batch zca act out
batch_zca_act_out_aux = batch_zca_act_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(batch_zca_act_out_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/batch_channelcov_matrix_zca_act_out.png')
plt.close()

### Violin plots of zca act out
fig, ax = plt.subplots()
ax.violinplot(batch_zca_act_out.view(-1).numpy(), showmeans=False, showmedians=True)
ax.set_title('ZCA act out')
ax.set_xlabel(f'{batch_aug_type}')
ax.set_ylabel('Values')
plt.savefig(f'{save_dir}/batch_violin_zca_act_out.png')
plt.close()

save_dir_max_values = f'{save_dir}/max_values_analysis'
os.makedirs(save_dir_max_values, exist_ok=True)

# Plot max values of each image per channel after activation
for channel in range(batch_zca_act_out.shape[1]):
    # apply max pooling to reduce 224x224 to 1x1
    max_values, positions = torch.nn.functional.max_pool2d(batch_zca_act_out[:,channel], kernel_size=(224,224), return_indices=True)
    max_values = max_values.squeeze()
    positions = positions.squeeze()
    # unravel positions to get the position in 2D
    positions = torch.stack((positions // 224, positions % 224), dim=1)
    # assert for one image that the max value is the correponding one from positions
    for i in range(len(max_values)):
        assert batch_zca_act_out[i,channel,positions[i,0],positions[i,1]] == max_values[i]

    # plot images index vs max values
    plt.figure()
    plt.scatter(range(len(max_values)), max_values, marker='o', label=f'channel {channel+1}')
    plt.title(f'Channel {channel+1} max values\nAfter activation function')
    plt.xlabel('Image index')
    plt.ylabel('Max value')
    plt.legend()
    # plt.ylim(0,35)
    plt.savefig(f'{save_dir_max_values}/channel_batch_zca_act_out_channel{channel+1}_max_values.png')
    plt.close()

    top5_max, top5_max_index = torch.topk(max_values, 5)
    top5_max_positions = positions[top5_max_index]

    for i in range(len(top5_max)):
        ## Subplot to show image, image with a kernel size rectangle where the max value is (zoomed in a kernelsize*10)
        plt.figure(figsize=(10,30))

        # plot image and plot a big cross the goes across the image on position of the max value
        plt.subplot(1,4,1)
        img = deepcopy(batch_train_images[top5_max_index[i]])
        img = img * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
        img_plot = img.permute(1,2,0).cpu().numpy()
        plt.imshow(img_plot)
        plt.axvline(top5_max_positions[i,1], color='gray', linewidth=2)
        plt.axhline(top5_max_positions[i,0], color='gray', linewidth=2)
        plt.title(f'image {top5_max_index[i]}')
        plt.axis('off')

        # plot a zoomed in version of the image where the max value is
        plt.subplot(1,4,2)
        zoomd_size = max(conv0_kernel_size, 16)
        img_patch = img[:,
                        max(0,top5_max_positions[i,0]-zoomd_size):min(224,top5_max_positions[i,0]+zoomd_size+1), 
                        max(0,top5_max_positions[i,1]-zoomd_size):min(224,top5_max_positions[i,1]+zoomd_size+1)]
        plt.imshow(img_patch.permute(1,2,0).cpu().numpy())
        plt.title('Zoomed in')
        plt.axis('off')

        # plot patch of the image where the max value is. It should be same size as kernel size
        plt.subplot(1,4,3)
        img_patch = img[:,
                        max(0,top5_max_positions[i,0]-conv0_kernel_size//2):min(224,top5_max_positions[i,0]+conv0_kernel_size//2+1), 
                        max(0,top5_max_positions[i,1]-conv0_kernel_size//2):min(224,top5_max_positions[i,1]+conv0_kernel_size//2+1)]
        plt.imshow(img_patch.permute(1,2,0).cpu().numpy())
        plt.title(f'Max patch max value {top5_max[i]:.2f}\nchannel {channel+1}')
        plt.axis('off')

        # Kernel for channel
        plt.subplot(1,4,4)
        kernel = zca_layer.weight[channel]
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
        kernel = kernel.cpu().numpy().transpose(1,2,0)
        plt.imshow(kernel)
        plt.title(f'Kernel\nchannel {channel+1}')
        plt.axis('off')

        # suptitle and save image      
        plt.savefig(f'{save_dir_max_values}/channel_batch_zca_act_out_channel{channel+1}_image{top5_max_index[i]}_max_value{top5_max[i]:.2f}.png', bbox_inches='tight')
        plt.close()

        # plot scatter of all batch zca act out values for this image
        img_zca_act_out = batch_zca_act_out[top5_max_index[i],channel].flatten()
        plt.figure()
        plt.scatter(range(len(img_zca_act_out)), img_zca_act_out, s=1, alpha=0.2)
        plt.title(f'image {top5_max_index[i]} zca act out\nchannel {channel+1}')
        plt.xlabel('Flattened index')
        plt.ylabel('value')
        plt.savefig(f'{save_dir_max_values}/channel_batch_zca_act_out_channel{channel+1}_image{top5_max_index[i]}_zca_act_out.png')
        plt.close()

### Get and plot saliency maps
zca_kernel_size = conv0_kernel_size
pool_kernel_size = 8
stride=1
ggd_param_nimg = 500
inner_batchsize = 100
num_crops = 12
top_percentage=0.1
crop_scale = [0.08, 0.08]

save_dir_saliency = f'{save_dir}/saliency_maps'
os.makedirs(save_dir_saliency, exist_ok=True)

with torch.no_grad():
    batch_zca_out = zca_layer(batch_train_images.cuda())
    batch_zca_act_out = act(batch_zca_out).cpu().detach()
    batch_zca_out = batch_zca_out.cpu().detach()

batch_feats = batch_zca_out

batch_abs_feats = batch_feats.abs().mean(dim=1).detach().cpu()
batch_abs_meanpool_feats = mean_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)
batch_abs_l2pool_feats = l2_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)

### Resize saliency maps to original image size (only meanpool and L2pool need this)
original_shape = (224, 224)
batch_abs_saliency_maps = deepcopy(batch_abs_feats)
batch_abs_meanpool_saliency_maps = resize_saliency_map_batch(batch_abs_meanpool_feats, original_shape)
batch_abs_l2pool_saliency_maps = resize_saliency_map_batch(batch_abs_l2pool_feats, original_shape)

### I am not dividing by the sum or using softmax. We can use the values directly in unravel_index as weights in the smart_crop_batch function
### To make the values more distant from each other, I will use the square of the values
batch_abs_saliency_maps = batch_abs_saliency_maps ** 2
batch_abs_meanpool_saliency_maps = batch_abs_meanpool_saliency_maps ** 2
batch_abs_l2pool_saliency_maps = batch_abs_l2pool_saliency_maps ** 2

### Get crops for the batch
batch_crops = smart_crop_batch(batch_abs_saliency_maps, num_crops=num_crops, scale=crop_scale, top_percentage=top_percentage)
batch_meanpool_crops = smart_crop_batch(batch_abs_meanpool_saliency_maps, num_crops=num_crops, scale=crop_scale, top_percentage=top_percentage)
batch_l2pool_crops = smart_crop_batch(batch_abs_l2pool_saliency_maps, num_crops=num_crops, scale=crop_scale, top_percentage=top_percentage)

### Plot image and saliency map for one image in the batch
for idx in range(50):
# idx = 5 # 5 (bad) 8 (good) 23 (bad) 24 (nice)
    image = batch_train_images[0+idx:1+idx]

    unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
    unnorm_image = unnorm_image.squeeze().numpy()
    unnorm_image = np.moveaxis(unnorm_image, 0, -1)

    zca_image = batch_feats[idx,:3,:,:].cpu().detach().numpy()
    zca_image = (zca_image - zca_image.min()) / (zca_image.max() - zca_image.min())
    zca_image = np.moveaxis(zca_image, 0, -1)

    saliencymap_image = batch_abs_saliency_maps[idx].numpy()
    saliencymap_meanpool_image = batch_abs_meanpool_saliency_maps[idx].numpy()
    saliencymap_l2pool_image = batch_abs_l2pool_saliency_maps[idx].numpy()

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
    plt.imshow(batch_abs_feats[idx].cpu().detach().numpy())
    plt.title(f'Mean Abs feat', fontsize=20)
    plt.axis('off')

    # plot saliency map
    plt.subplot(3,3,4)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_image, cmap='jet', alpha=0.5)
    plt.title(f'Raw', fontsize=20)
    plt.axis('off')

    # plot saliency map (mean pool)
    plt.subplot(3,3,5)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_meanpool_image, cmap='jet', alpha=0.5)
    plt.title(f'Meanpool, kernel: {pool_kernel_size})', fontsize=20)
    plt.axis('off')

    # plot saliency map (L2 pool)
    plt.subplot(3,3,6)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_l2pool_image, cmap='jet', alpha=0.5)
    plt.title(f'L2 pool, kernel: {pool_kernel_size})', fontsize=20)
    plt.axis('off')

    # plot image, saliency map, and a bounding box for the crop
    plt.subplot(3,3,7)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_image, cmap='jet', alpha=0.5)
    for n_crop in range(num_crops):
        plt.gca().add_patch(plt.Rectangle((batch_crops[idx][n_crop][0], batch_crops[idx][n_crop][1]), batch_crops[idx][n_crop][2], batch_crops[idx][n_crop][3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.title('Crop', fontsize=20)
    plt.axis('off')

    # plot image, saliency map, and a bounding box for the crop (mean pool)
    plt.subplot(3,3,8)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_meanpool_image, cmap='jet', alpha=0.5)
    for n_crop in range(num_crops):
        plt.gca().add_patch(plt.Rectangle((batch_meanpool_crops[idx][n_crop][0], batch_meanpool_crops[idx][n_crop][1]), batch_meanpool_crops[idx][n_crop][2], batch_meanpool_crops[idx][n_crop][3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.title('Crop (mean pool)', fontsize=20)
    plt.axis('off')

    # plot image, saliency map, and a bounding box for the crop (L2 pool)
    plt.subplot(3,3,9)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_l2pool_image, cmap='jet', alpha=0.5)
    for n_crop in range(num_crops):
        plt.gca().add_patch(plt.Rectangle((batch_l2pool_crops[idx][n_crop][0], batch_l2pool_crops[idx][n_crop][1]), batch_l2pool_crops[idx][n_crop][2], batch_l2pool_crops[idx][n_crop][3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.title('Crop (L2 pool)', fontsize=20)
    plt.axis('off')

    # save plot
    plt.savefig(os.path.join(save_dir_saliency,f"image_{idx}_saliency_pool_kernel_{pool_kernel_size}.png"), bbox_inches='tight', dpi=300)
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
    plt.savefig(os.path.join(save_dir_saliency,f"image_{idx}_zcafeats.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # plot Saliency maps distributions
    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.scatter(range(len(saliencymap_image.flatten())), saliencymap_image.flatten(), s=1, alpha=0.2)
    plt.title(f'Raw')
    plt.xlabel('Flattened index')
    plt.ylabel('value')

    plt.subplot(1,3,2)
    plt.scatter(range(len(saliencymap_meanpool_image.flatten())), saliencymap_meanpool_image.flatten(), s=1, alpha=0.2)
    plt.title(f'Meanpool')
    plt.xlabel('Flattened index')
    plt.ylabel('value')

    plt.subplot(1,3,3)
    plt.scatter(range(len(saliencymap_l2pool_image.flatten())), saliencymap_l2pool_image.flatten(), s=1, alpha=0.2)
    plt.title(f'L2pool')
    plt.xlabel('Flattened index')
    plt.ylabel('value')

    plt.savefig(os.path.join(save_dir_saliency,f"image_{idx}_saliency_pool_kernel_{pool_kernel_size}_distributions.png"), bbox_inches='tight', dpi=300)
    plt.close()



