import os
import numpy as np
from copy import deepcopy

import torch
import matplotlib.pyplot as plt
import einops


def calculate_PCA_conv0_weights(model, dataset, save_dir, nimg = 10000, epsilon=1e-6):

    # create clone of conv0
    conv0 = deepcopy(model.conv0)

    # get images
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=nimg, shuffle=True, num_workers=8)
    imgs,labels = next(iter(train_loader))
    
    # extract Patches 
    kernel_size = conv0.kernel_size[0]
    patches = extract_patches(imgs, kernel_size, step=kernel_size)

    # get weight and bias with ZCA
    weight, bias = get_filters(patches,
                               epsilon = epsilon, 
                               save_dir=save_dir)
    
    # plot final filters
    num_filters = weight.shape[0]
    plt.figure(figsize=(3*(num_filters/2), 5))
    for i in range(num_filters):
        if i >= weight.shape[0]:
            break
        filter_m = weight[i]
        f_min, f_max = filter_m.min(), filter_m.max()
        if f_max != f_min:
            filter_m = (filter_m - f_min) / (f_max - f_min) # make them from 0 to 1
        filter_m = filter_m.cpu().numpy().transpose(1,2,0) # put channels last
        plt.subplot(2,int(np.ceil(num_filters/2)),i+1)
        plt.imshow(filter_m, vmax=1, vmin=0)          
        plt.axis('off')
        i +=1
    plt.savefig(f'{save_dir}/PCA_filters.jpg',dpi=300,bbox_inches='tight')
    plt.close()

    # plot filters values histogram
    plt.figure(figsize=(3*(num_filters/2), 5))
    for i in range(num_filters):
        if i >= weight.shape[0]:
            break
        filter_m = weight[i]
        plt.subplot(2,int(np.ceil(num_filters/2)),i+1)
        plt.hist(filter_m.flatten(), label=f'filter {i}')
        plt.legend()
        i +=1
    plt.savefig(f'{save_dir}/PCA_filters_hist.jpg',bbox_inches='tight')
    plt.close()

    # # Plot channel covariance matrix of feature maps
    # # load conv0 with weight and bias
    # conv0.weight = torch.nn.Parameter(weight)
    # conv0.bias = torch.nn.Parameter(bias)
    # # get covariance
    # feat_map = conv0(imgs)
    # channel_cov_matrix = compute_channel_covariance(feat_map.detach())
    # # plot
    # error = compute_error(channel_cov_matrix)
    # plt.figure()
    # plt.imshow(channel_cov_matrix.detach().cpu().numpy(), interpolation='nearest')
    # plt.title(f'Covariance matrix (Channel) for conv0 feature maps (Error: {error:.5e})')
    # plt.colorbar()
    # plt.savefig(f'{save_dir}/channel_covariance.jpg', bbox_inches='tight')
    # plt.close()

    return weight, bias

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, epsilon=5e-4, save_dir=None):
    n_patches, num_colors, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # data is a d x N

    # use double presicion
    data = data.double()

    # Compute ZCA
    W = PCA(data, epsilon = epsilon).to(data.device)

    # Expand filters by having negative version
    W = torch.cat([W, -W], dim=0)

    # # Add filters with all 1s and all -1s
    # W = torch.cat([W, torch.ones_like(W[:1]), -torch.ones_like(W[:1])], dim=0)

    # # L1 normalize filters
    # W = W / W.abs().sum(dim=1, keepdim=True)
    
    # Plot covariance matrix
    cov = torch.cov(W @ data)
    cov_error = compute_error(cov).item()
    plt.figure()
    plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
    plt.title(f'Covariance on patches (Error: {cov_error:.5e})')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,'covariance_on_patches.jpg'), bbox_inches='tight')
    plt.close()

    # Get bias
    bias = -(W @ data).mean(dim=1)

    # reshape filters
    W = einops.rearrange(W, 'k (c h w) -> k c h w', c=3, h=filt_size, w=filt_size)

    # back to single precision
    W = W.float()
    bias = bias.float()

    return W, bias

def PCA(data, epsilon=5e-4):

    C = (data @ data.T) / data.size(1)
    
    D, V = torch.linalg.eigh(C)
    sorted_indices = torch.argsort(D, descending=True)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    D = torch.clamp(D, min=0)  # Replace negative values with zero
    Di = (D + epsilon)**(-0.5)
    Di[~torch.isfinite(Di)] = 0
    pca_trans = torch.diag(Di) @ V.T

    return pca_trans

def compute_channel_covariance(feature_map):
    # feature_map --> (batch_size, num_channels, height, width)
    # Verify that the feature map is a 4D tensor
    if feature_map.dim() != 4:
        raise ValueError("The feature map must be a 4D tensor")
    # Reshape the feature map to shape (batch_size * height * width, num_channels)
    reshaped_features = einops.rearrange(feature_map, "b c h w -> (b h w) c").t()
    cov_matrix = torch.cov(reshaped_features)
    return cov_matrix

def compute_covariance(embeddings):
    # Center the embeddings
    embeddings = embeddings - torch.mean(embeddings, dim=0, keepdim=True)
    # Compute the covariance matrix
    cov_matrix = torch.mm(embeddings.t(), embeddings) / (embeddings.size(0) - 1)
    return cov_matrix

def compute_error(cov_matrix):
    # Compute the Frobenius norm of the difference between the covariance matrix and the identity matrix
    identity = torch.eye(cov_matrix.size(0)).to(cov_matrix.device)
    error = torch.norm(identity - cov_matrix, p='fro')
    return error