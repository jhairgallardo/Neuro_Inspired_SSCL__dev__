import torch
import torch.nn.functional as F
import einops

from scipy.stats import gennorm
from scipy.special import gamma


from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
import math
import os

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
    saliency_maps = 1 / (saliency_maps + 1e-5)
    saliency_maps /= saliency_maps.sum(dim=(1, 2), keepdim=True)
    saliency_maps = saliency_maps.to(torch.float32)
    return saliency_maps

def resize_saliency_map_batch(saliency_maps, original_shape=(224,224)):
    # saliency_maps: tensor of shape (batch_size, height, width)
    saliency_maps_tensor = saliency_maps.unsqueeze(1)  # Add channel dimension
    resized_saliency_maps = F.interpolate(saliency_maps_tensor, size=original_shape, mode='bilinear', align_corners=True)
    resized_saliency_maps = resized_saliency_maps.squeeze(1)  # Remove channel dimension

    # Normalize each saliency map in the batch
    batch_sums = resized_saliency_maps.view(resized_saliency_maps.shape[0], -1).sum(dim=1)  # Compute sum of each saliency map
    resized_saliency_maps /= batch_sums.view(-1, 1, 1)  # Normalize using broadcasting
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
                # idx_cy, idx_cx = np.unravel_index(torch.argmax(saliency_map.flatten()).item(), saliency_map.shape)

                # if part of the crop falls outside the image, then move the center of the crop.
                # It makes sure that the sampled center is within the crop (not necesarily the center, but inside the crop)
                # It also makes sure that the crop is within the image
                if idx_cy + (h_crop // 2) >= height:
                    diff_cy = idx_cy + (h_crop // 2) - height + (h_crop % 2)
                    idx_cy -= diff_cy
                if idx_cy - (h_crop // 2) <= 0:
                    diff_cy = (h_crop // 2) - idx_cy 
                    idx_cy += diff_cy
                if idx_cx + (w_crop // 2) >= width:
                    diff_cx = idx_cx + (w_crop // 2) - width + (w_crop % 2)
                    idx_cx -= diff_cx
                if idx_cx - (w_crop // 2) <= 0:
                    diff_cx = (w_crop // 2) - idx_cx
                    idx_cx += diff_cx
                idx_y = idx_cy - (h_crop // 2)
                idx_x = idx_cx - (w_crop // 2)

                # make sure the complete crop is within the image (safety check)
                if 0 <= idx_x and idx_x + w_crop <= width and 0 <= idx_y and idx_y + h_crop <= height:
                    break
                elif i == 9: # if it fails 10 times, then just take the center of the image (this shouldn't happen)
                    print('Warning: crop is out of bounds. Taking center of image instead.')
                    idx_cx = width // 2
                    idx_cy = height // 2
                    idx_x = idx_cx - w_crop // 2
                    idx_y = idx_cy - h_crop // 2
            all_crops[b,k] = torch.tensor([idx_x, idx_y, w_crop, h_crop])
    return all_crops 

def crop_and_resize(episodes_imgs, episodes_crops):
    batch_size, num_views, _, height, width = episodes_imgs.shape
    for b in range(batch_size):
        for v in range(num_views):
            idx_x, idx_y, w_crop, h_crop = episodes_crops[b, v]
            crop = episodes_imgs[b, v, :, idx_y:idx_y+h_crop, idx_x:idx_x+w_crop].unsqueeze(1)
            episodes_imgs[b, v] = F.interpolate(crop, size=(height, width), mode='bilinear', align_corners=True).squeeze(1)
    return episodes_imgs

def apply_guided_crops(episodes_imgs, 
                       layer, 
                       ggd_params, 
                       scale = [0.08, 1.0], 
                       ratio = [3.0/4.0, 4.0/3.0], 
                       weighted=False, 
                       pool_mode=None, 
                       return_others=False):
    num_views = episodes_imgs.shape[1]

    # crops are calculated using the first views only
    imgs = episodes_imgs[:, 0, :, :, :] # shape b,c,w,h
    # model to eval
    layer.eval()

    # get feature outputs
    with torch.no_grad():
        feats = layer(imgs).detach()
    # get abs feats
    abs_feats = feats.abs()

    # get pool feats if activated
    if pool_mode == 'l2pool':
        pool_abs_feats = l2_pooling_batch(abs_feats, kernel_size=8, stride=1)
    elif pool_mode == 'meanpool':
        pool_abs_feats = mean_pooling_batch(abs_feats, kernel_size=8, stride=1)
    else:
        pool_abs_feats = abs_feats

    # get saliency map
    saliency_maps = compute_saliency_map_ggd_batch(pool_abs_feats, ggd_params, weighted=weighted)

    # resize saliency map if pool was used
    if pool_mode is not None:
        saliency_maps = resize_saliency_map_batch(saliency_maps, original_shape=(imgs.shape[-1], imgs.shape[-2]))
    
    # get crops parameters
    episodes_crops = smart_crop_batch(saliency_maps, num_crops=num_views, scale=scale, ratio=ratio) # shape b, num_crops, 4

    # apply parameters and get final crops. Remeber, fist view should not be cropped
    episodes_imgs[:, 1:, :, :, :] = crop_and_resize(episodes_imgs[:, 1:, :, :, :], episodes_crops)

    if return_others:
        return episodes_imgs, abs_feats, saliency_maps, episodes_crops
    else:
        return episodes_imgs



            





    


