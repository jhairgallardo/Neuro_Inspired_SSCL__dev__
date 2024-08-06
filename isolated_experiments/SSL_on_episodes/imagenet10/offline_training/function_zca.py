import torch
import torchvision
import matplotlib.pyplot as plt
import einops

import os, json
import numpy as np
from copy import deepcopy

def calculate_ZCA_conv0_weights(model, imgs, zca_epsilon=1e-6):

    # create clone of conv0
    conv0 = deepcopy(model.conv0)
    
    # extract Patches 
    kernel_size = conv0.kernel_size[0]
    patches = extract_patches(imgs, kernel_size, step=kernel_size)

    # get weight with ZCA
    num_channels_out = conv0.weight.shape[0]
    weight = get_filters(patches, zca_epsilon=zca_epsilon, zca_num_channels=num_channels_out)

    return weight

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, zca_epsilon=1e-6, zca_num_channels=6):
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

def compute_error(cov_matrix):
    # Compute the Frobenius norm of the difference between the covariance matrix and the identity matrix
    identity = torch.eye(cov_matrix.size(0)).to(cov_matrix.device)
    error = torch.norm(identity - cov_matrix, p='fro')
    return error

def scaled_filters(zca_layer, imgs, per_channel=False, prob_threshold=0.99999):
    # Maxscale the filters
    weight = zca_layer.weight
    weight = weight / torch.amax(weight, keepdims=True)
    zca_layer.weight = torch.nn.Parameter(weight)
    zca_layer.weight.requires_grad = False

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

    return weight