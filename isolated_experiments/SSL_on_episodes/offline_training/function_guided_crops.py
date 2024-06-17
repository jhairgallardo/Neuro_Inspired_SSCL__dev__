import torch


import einops
import numpy as np

def l2_pooling(feats, kernel_size, stride, padding):
    squared_feats = feats ** 2
    pooled_feats = torch.nn.functional.avg_pool2d(squared_feats, kernel_size, stride=stride, padding=padding)
    return torch.sqrt(pooled_feats.squeeze(0))

def fit_gaussian_distribution(tensor):
    """
    Fit a Gaussian distribution to the tensor, return the mean and standard deviation.
    """
    mean = tensor.mean().item()
    std = tensor.std().item()
    return mean, std

def apply_guided_crops(episodes_batch, zca_layer, scale=(0.08,1.0), ratio = (3/4, 4/3)):
    zca_layer.to(episodes_batch.device)
    zca_layer.eval()
    b = episodes_batch.size(0)

    # Get features of all episodes in the batch
    images_batch = einops.rearrange(episodes_batch, 'b v c h w -> (b v) c h w').contiguous()
    absfeats_batch = zca_layer(images_batch).abs()

    # L2 pooling
    absfeats_batch_L2pooled = l2_pooling(absfeats_batch, kernel_size=3, stride=1, padding=0)

    return None



    


