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
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
from scipy.stats import gennorm
from scipy.special import gamma, gammainc
import math
from multiprocessing import Pool, cpu_count

def calculate_ZCA_conv0_weights(zca_layer, imgs, zca_epsilon=1e-6, save_dir=None):

    # create clone of conv0
    conv0 = deepcopy(zca_layer)

    # save imgs channel mean in json format
    mean = imgs.mean(dim=(0,2,3)).tolist()
    with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f:
        json.dump(mean, f)
    
    # extract Patches 
    kernel_size = conv0.kernel_size[0]
    patches = extract_patches(imgs, kernel_size, step=kernel_size)

    # get weight with ZCA
    num_channels_out = conv0.weight.shape[0]
    weight = get_filters(patches, zca_epsilon=zca_epsilon, zca_num_channels=num_channels_out, save_dir=save_dir)

    return weight

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, zca_epsilon=1e-6, zca_num_channels=6, save_dir=None):
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

    # Expand filters by having negative version
    if zca_num_channels >=6:
        W_center = torch.cat([W_center, -W_center], dim=0)

    # back to single precision
    W_center = W_center.float()

    # Plot raw all filters
    num_filters = W.shape[0]
    num_rows = int(num_filters / (filt_size**2))
    num_columns = int(filt_size**2)
    plt.figure(figsize=(num_columns*3,num_rows*3))
    for i in range(num_filters):
        filter_m = W[i]
        plt.subplot(num_rows,num_columns,i+1)
        plt.hist(filter_m.flatten(), label=f'filter {i}')
        plt.legend()
    plt.savefig(os.path.join(save_dir,'zca_filters_all_raw_hist.jpg'), bbox_inches='tight')
    plt.close()

    # Plot raw all filters hist
    plt.figure(figsize=(num_columns*3,num_rows*3))
    for i in range(num_filters):
        filter_m = W[i]
        filter_m = (filter_m - filter_m.min()) / (filter_m.max() - filter_m.min())
        filter_m = filter_m.cpu().numpy().transpose(1,2,0)
        plt.subplot(num_rows,num_columns,i+1)
        plt.imshow(filter_m)
        plt.axis('off')
    plt.savefig(os.path.join(save_dir,'zca_filters_all_raw.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot raw all filters output channel covariance matrix
    cov = torch.cov(einops.rearrange(W, 'k c h w -> k (c h w)') @ data)
    plt.figure()
    plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,'zcainput_cov_zca_all_raw_out.jpg'), bbox_inches='tight')
    plt.close()

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

def scaled_filters(zca_layer, imgs, per_channel=False, save_dir=None, prob_threshold=0.99999):
    # Maxscale the filters
    weight = zca_layer.weight
    weight = weight / torch.amax(weight, keepdims=True)
    zca_layer.weight = torch.nn.Parameter(weight)
    zca_layer.weight.requires_grad = False

    plot_filters_and_hist(zca_layer.weight, name='zca_filters_maxscaled', save_dir=save_dir)
    plot_channel_cov(zca_layer, imgs, name='zcainput_channelcov_zca_maxscaled_out', save_dir=save_dir)
    plot_zca_layer_output_hist(zca_layer, imgs, name='zcainput_zca_maxscaled_out_hist', save_dir=save_dir)

    # Get the output of the zca layer
    zca_layer.eval()
    zca_layer_output = zca_layer(imgs)

    if per_channel:
        for channel in range(zca_layer_output.shape[1]):
            # Get CDF on the absolute values of the output
            zca_layer_output_channel = zca_layer_output[:,channel,:,:]
            zca_layer_output_channel = zca_layer_output_channel[zca_layer_output_channel>=0] # ignore negative values
            zca_layer_output_channel = zca_layer_output_channel.flatten().numpy()
            count, bins_count = np.histogram(zca_layer_output_channel, bins=100) 
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)

            # Find threshold, multiplier, and scale the filters
            threshold = bins_count[1:][np.argmax(cdf>prob_threshold)]
            multiplier = np.arctanh(0.999)/threshold # find value where after a tanh function, everythinng after the threshold will be near 1 (0.999)
            print('Channel:', channel)
            print(f'threshold: {threshold}')
            print(f'multiplier: {multiplier}')
            weight[channel] = weight[channel] * multiplier

            plot_cdf_with_threshold(cdf, bins_count, threshold, prob_threshold, multiplier, save_dir, name=f'zcainput_zca_maxscaled_out_cdf_channel{channel}')

    else:
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

        plot_cdf_with_threshold(cdf, bins_count, threshold, prob_threshold, multiplier, save_dir, name='zcainput_zca_maxscaled_out_cdf')

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

def plot_cdf_with_threshold(cdf, bins_count, threshold, prob_threshold, multiplier, save_dir, name):
    plt.figure()
    plt.plot(bins_count[1:], cdf, label='CDF', linewidth=3)
    plt.axhline(prob_threshold, color='green', label=f'prob_threshold: {prob_threshold}')
    plt.axvline(threshold, color='red', label=f'threshold: {threshold}')
    plt.title(f'CDF (Threshold at {prob_threshold*100:.3f}%)\nThreshold: {threshold:.3f} Multiplier: {multiplier:.3f}')
    plt.xlabel('Values')
    plt.ylabel('Probability (at or below value)')
    plt.savefig(f'{save_dir}/{name}.png', bbox_inches='tight')
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

def calculate_GGD_params(dataset, layer, nimg=1000, pool_mode=None, inner_batchsize=100, device='cuda'):
    # get loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=inner_batchsize, shuffle=True)
    # model to eval
    layer.eval()
    # get nimg feature outputs from model.conv0
    with torch.no_grad():
        feats = []
        for i, (batch_images, _) in enumerate(loader):
            batch_feats = layer(batch_images.to(device)).detach().cpu()
            feats.append(batch_feats)
            if (i+1) * inner_batchsize >= nimg:
                break
        feats = torch.cat(feats, dim=0)
    # get abs feats
    abs_feats = feats.abs()
    # get pool feats if activated
    if pool_mode == 'l2pool':
        pool_abs_feats = l2_pooling_batch(abs_feats, kernel_size=8, stride=1)
    elif pool_mode == 'meanpool':
        pool_abs_feats = mean_pooling_batch(abs_feats, kernel_size=8, stride=1)
    else:
        pool_abs_feats = abs_feats
    # get ggd params
    ggd_params = fit_gennorm_to_batch(pool_abs_feats)
    return ggd_params

def gennorm_pdf(x, theta, loc, sigma):
    # https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1
    # https://github.com/scipy/scipy/blob/87c46641a8b3b5b47b81de44c07b840468f7ebe7/scipy/stats/_continuous_distns.py#L11153
    theta = torch.tensor(theta)
    loc = torch.tensor(loc)
    sigma = torch.tensor(sigma)
    return torch.exp( torch.log(0.5*theta) - torch.log(sigma) - torch.lgamma(1.0/theta) - (torch.abs(x-loc)/sigma)**theta )

def compute_saliency_map_ggd_batch(features, ggd_params, weighted=False):
    features = features.to(torch.float64)
    batch_size, num_filters, height, width = features.shape

    # Compute the joint probability of all filters for each image in the batch
    saliency_maps = torch.ones((batch_size, height, width), dtype=torch.float64).to(features.device)
    for i in range(num_filters):
        theta, loc, sigma = ggd_params[i]
        if weighted:
            theta_inv = torch.tensor(1.0 / theta)
            gamma_func = gamma(theta_inv)
            theta_inv = theta_inv.to(features.device)
            gamma_incomplete = torch.special.gammainc(theta_inv, (torch.abs(features[:, i, :, :]) ** theta) * (sigma ** -theta))
            improved_feats = gamma_incomplete / gamma_func
            p = gennorm_pdf(improved_feats.flatten(), theta, loc, sigma)
        else:
            p = gennorm_pdf(features[:, i, :, :].flatten(), theta, loc, sigma)
        p = p.reshape(batch_size, height, width)
        saliency_maps *= p
    saliency_maps = 1 / (saliency_maps + 1e-18)
    saliency_maps = saliency_maps.to(torch.float32)
    return saliency_maps

def resize_saliency_map_batch(saliency_maps, original_shape=(224,224)):
    # saliency_maps: tensor of shape (batch_size, height, width)
    saliency_maps_tensor = saliency_maps.unsqueeze(1)  # Add channel dimension
    resized_saliency_maps = F.interpolate(saliency_maps_tensor, size=original_shape, mode='bilinear', align_corners=True)
    resized_saliency_maps = resized_saliency_maps.squeeze(1)  # Remove channel dimension
    return resized_saliency_maps

def smart_crop_batch(saliency_maps, num_crops = 1, scale = [0.08, 1.0], ratio = [3.0/4.0, 4.0/3.0]):
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
                idx = probabilities.multinomial(1).item()
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
conv0_outchannels=6
conv0_kernel_size=3
nimg = 10000 #10000
zca_epsilon = 1e-4 # 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
dataset_name='ImageNet-10'
activation = 'mishtanh' # noact, tanh, mish, softplustanh, relutanh, mishtanh
zca_scale_filter = True
zca_scale_filter_mode = 'per_channel' # per_channel, all
save_dir = f'output'
num_batch_plot=16

save_dir += f'/{batch_aug_type}aug_{conv0_kernel_size}kernerlsize_{conv0_outchannels}channels_{zca_epsilon}eps_{activation}'
if zca_scale_filter:
    save_dir += f'_scaled_{zca_scale_filter_mode}'
os.makedirs(save_dir, exist_ok=True)

### Data to calculate ZCA layer
mean=[0.485, 0.456, 0.406]
std=[1.0, 1.0, 1.0]

### Calculate ZCA layer
zca_layer = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', bias=False)
zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
zca_dataset = datasets.ImageFolder(root=f"/data/datasets/{dataset_name}/train", transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=nimg, shuffle=True)
zca_input_imgs,_ = next(iter(zca_loader))

weight = calculate_ZCA_conv0_weights(zca_layer=zca_layer, imgs=zca_input_imgs,
                                      zca_epsilon=zca_epsilon, save_dir=save_dir)
zca_layer.weight = torch.nn.Parameter(weight)
zca_layer.weight.requires_grad = False

plot_filters_and_hist(zca_layer.weight, name='zca_filters_raw', save_dir=save_dir)
plot_channel_cov(zca_layer, zca_input_imgs, name='zcainput_channelcov_zca_raw_out', save_dir=save_dir)
plot_zca_layer_output_hist(zca_layer, zca_input_imgs, name='zcainput_zca_raw_out_hist', save_dir=save_dir)

if zca_scale_filter:
    if zca_scale_filter_mode == 'all':
        weight = scaled_filters(zca_layer, imgs=zca_input_imgs, save_dir=save_dir)
    elif zca_scale_filter_mode == 'per_channel':
        weight = scaled_filters(zca_layer, imgs=zca_input_imgs, save_dir=save_dir, per_channel=True)
    zca_layer.weight = torch.nn.Parameter(weight)
    zca_layer.weight.requires_grad = False

plot_filters_and_hist(zca_layer.weight, name='zca_filters', save_dir=save_dir)
plot_channel_cov(zca_layer, zca_input_imgs, name='zcainput_channelcov_zca_out', save_dir=save_dir)
plot_zca_layer_output_hist(zca_layer, zca_input_imgs, name='zcainput_zca_out_hist', save_dir=save_dir)

# Define activation function
if activation == 'tanh':
    act = nn.Tanh()
elif activation == 'mish':
    act = nn.Mish()
elif activation == 'softplustanh':
    act = nn.Sequential(nn.Softplus(),
                        nn.Tanh())
elif activation == 'relutanh':
    act = nn.Sequential(nn.ReLU(),
                        nn.Tanh())
elif activation == 'mishtanh':
    act = nn.Sequential(nn.Mish(),
                        nn.Tanh())
elif activation == 'noact':
    act = nn.Identity()

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
batch_zca_out_aux = deepcopy(batch_zca_out[:num_batch_plot]) # * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
for i in range(num_batch_plot):
    batch_zca_out_aux[i] = ( batch_zca_out_aux[i] - batch_zca_out_aux[i].min() )/ (batch_zca_out_aux[i].max() - batch_zca_out_aux[i].min())
grid = torchvision.utils.make_grid(batch_zca_out_aux[:,:3,:,:], nrow=4)
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
### Get GGD params for a batch_zca_image (no pooling, meanpool and L2 pool)
zca_kernel_size = conv0_kernel_size
pool_kernel_size = 8
stride=1
ggd_param_nimg = 500
inner_batchsize = 100
weighted=True
act_output=False
num_crops = 12
crop_scale = [0.08, 0.08]

save_dir_saliency = f'{save_dir}/saliency_maps'
if weighted:
    save_dir_saliency += '_weighted'
if act_output:
    save_dir_saliency += '_act_output'
os.makedirs(save_dir_saliency, exist_ok=True)

ggd_params = calculate_GGD_params(zca_dataset, zca_layer, nimg=ggd_param_nimg, pool_mode=None, inner_batchsize=inner_batchsize, device=zca_layer.weight.device)
meanpool_ggd_params = calculate_GGD_params(zca_dataset, zca_layer, nimg=ggd_param_nimg, pool_mode='meanpool', inner_batchsize=inner_batchsize, device=zca_layer.weight.device)
l2pool_ggd_params = calculate_GGD_params(zca_dataset, zca_layer, nimg=ggd_param_nimg, pool_mode='l2pool', inner_batchsize=inner_batchsize, device=zca_layer.weight.device)

with torch.no_grad():
    batch_zca_out = zca_layer(batch_train_images.cuda())
    batch_zca_act_out = act(batch_zca_out).cpu().detach()
    batch_zca_out = batch_zca_out.cpu().detach()

if act_output:
    batch_feats = batch_zca_act_out
else:
    batch_feats = batch_zca_out

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

# # make saliency maps a probability distribution using softmax (not working currently. I get an error that it probs don't sum to one. Possibly numerical instability)
# batch_saliency_maps = F.softmax(batch_saliency_maps.view(batch_saliency_maps.shape[0], -1), dim=1).view(batch_saliency_maps.shape)
# batch_meanpool_saliency_maps = F.softmax(batch_meanpool_saliency_maps.view(batch_meanpool_saliency_maps.shape[0], -1), dim=1).view(batch_meanpool_saliency_maps.shape)
# batch_l2pool_saliency_maps = F.softmax(batch_l2pool_saliency_maps.view(batch_l2pool_saliency_maps.shape[0], -1), dim=1).view(batch_l2pool_saliency_maps.shape)

# make saliency maps a probability by diving by the sum
batch_saliency_maps/=batch_saliency_maps.sum(dim=(1, 2), keepdim=True)
batch_meanpool_saliency_maps/=batch_meanpool_saliency_maps.sum(dim=(1, 2), keepdim=True)
batch_l2pool_saliency_maps/=batch_l2pool_saliency_maps.sum(dim=(1, 2), keepdim=True)

### Get crops for the batch
batch_crops = smart_crop_batch(batch_saliency_maps, num_crops=num_crops, scale=crop_scale)
batch_meanpool_crops = smart_crop_batch(batch_meanpool_saliency_maps, num_crops=num_crops, scale=crop_scale)
batch_l2pool_crops = smart_crop_batch(batch_l2pool_saliency_maps, num_crops=num_crops, scale=crop_scale)

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
    plt.imshow(saliencymap_image, cmap='jet', alpha=0.5)
    if weighted:
        plt.title(f'Weighted GGD', fontsize=20)
    else:
        plt.title(f'GGD', fontsize=20)
    plt.axis('off')

    # plot saliency map (mean pool)
    plt.subplot(3,3,5)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_meanpool_image, cmap='jet', alpha=0.5)
    if weighted:
        plt.title(f'Weighted GGD (mean pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (mean pool)', fontsize=20)
    plt.axis('off')

    # plot saliency map (L2 pool)
    plt.subplot(3,3,6)
    plt.imshow(unnorm_image.mean(2), cmap='gray')
    plt.imshow(saliencymap_l2pool_image, cmap='jet', alpha=0.5)
    if weighted:
        plt.title(f'Weighted GGD (L2 pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (L2 pool)', fontsize=20)
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
    if weighted:
        plt.savefig(os.path.join(save_dir_saliency,f"image_{idx}_saliency_weighted_pool_kernel_{pool_kernel_size}.png"), bbox_inches='tight', dpi=300)
    else:
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



