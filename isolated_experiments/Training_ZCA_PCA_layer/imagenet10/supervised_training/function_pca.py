import torch
import matplotlib.pyplot as plt
import einops

import os, json
import numpy as np
from copy import deepcopy

def get_patches(x, patch_shape, step=1):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,step).unfold(3,w,step).transpose(1,3).reshape(-1,c,h,w)

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def calculate_PCA_conv0_weights(model, dataset, save_dir, nimg = 10000, epsilon=1e-6):

    # create clone of conv0
    conv0 = deepcopy(model.conv0)
    kernel_size = conv0.kernel_size[0]

    # get images
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=nimg, shuffle=True)
    imgs,labels = next(iter(train_loader))

    # save imgs channel mean in json format
    mean = imgs.mean(dim=(0,2,3)).tolist()
    with open(f'{save_dir}/mean_imgs_input_for_pca.json', 'w') as f:
        json.dump(mean, f)

    # extract Patches 
    patches = extract_patches(imgs, kernel_size, step=kernel_size)
    # get weight with ZCA
    weight, bias = get_filters(patches, epsilon = epsilon)

    # # Their code: https://arxiv.org/pdf/2404.00498 
    # # https://github.com/KellerJordan/cifar10-airbench/blob/e16b886f53ca617017c0e5f9799632a721428f65/airbench94.py#L237
    # patch_shape = (kernel_size, kernel_size)
    # patches = get_patches(imgs, patch_shape, step=kernel_size).double()
    # eigenvalues, eigenvectors = get_whitening_parameters(patches)
    # eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + epsilon)
    # weight = torch.cat((eigenvectors_scaled, -eigenvectors_scaled)).float()
    
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
        plt.imshow(filter_m)          
        plt.axis('off')
        i +=1
    plt.savefig(f'{save_dir}/PCA_filters.jpg', bbox_inches='tight')
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

    return weight, bias

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, epsilon=5e-4):
    n_patches, num_colors, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # data is a d x N

    # use double presicion
    data = data.double()

    # Compute ZCA
    W = PCA(data, epsilon = epsilon).to(data.device)

    # Expand filters by having negative version
    W = torch.cat([W, -W], dim=0)

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
    # D = torch.clamp(D, min=0)  # Replace negative values with zero
    Di = (D + epsilon)**(-0.5)
    # Di[~torch.isfinite(Di)] = 0
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