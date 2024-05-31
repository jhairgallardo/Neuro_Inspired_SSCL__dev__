import os
import numpy as np
from copy import deepcopy

import torch
import matplotlib.pyplot as plt
import einops


def calculate_ZCA_conv0_weights(model, dataset, addgray, save_dir, nimg = 10000, zca_epsilon=1e-6):

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
                               real_cov=False,
                               add_gray=addgray,
                               zca_epsilon = zca_epsilon, 
                               save_dir=save_dir)
    
    # plot final filters
    plt.figure()
    num_filters = weight.shape[0]
    for i in range(num_filters):
        if i >= weight.shape[0]:
            break
        filter_m = weight[i]
        f_min, f_max = filter_m.min(), filter_m.max()
        if f_max != f_min:
            filter_m = (filter_m - f_min) / (f_max - f_min) # make them from 0 to 1
        filter_m = filter_m.cpu().numpy().transpose(1,2,0) # put channels last
        plt.subplot(1,num_filters,i+1)
        plt.imshow(filter_m, vmax=1, vmin=0)          
        plt.axis('off')
        i +=1
    plt.savefig(f'{save_dir}/ZCA_filters.jpg',dpi=300,bbox_inches='tight')
    plt.close()

    # plot filters values histogram
    plt.figure(figsize=(5,5*num_filters))
    for i in range(num_filters):
        if i >= weight.shape[0]:
            break
        filter_m = weight[i]
        plt.subplot(num_filters,1,i+1)
        plt.hist(filter_m.flatten(), label=f'filter {i}')
        plt.legend()
        i +=1
    plt.savefig(f'{save_dir}/ZCA_filters_hist.jpg',bbox_inches='tight')
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

def get_filters(patches_data, real_cov=False, add_gray=True, zca_epsilon=1e-6, save_dir=None):
    n_patches, num_colors, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # data is a d x N

    d_size = data.size(0)
    enforce_symmetry = True

    # use double presicion
    data = data.double()
    
    # Calculate gray filter
    oData = data.clone()
    C_Trans = None
    if add_gray:
        data, C_Trans = add_gray_channels(data, num_colors, filt_size)
        num_colors += 1

    # Compute ZCA
    W = ZCA(data, real_cov, tiny = zca_epsilon).to(data.device)
    W_before_merge = deepcopy(W)
    W = mergeFilters(W, filt_size, num_colors, d_size, C_Trans, enforce_symmetry, plot_all=False, save_dir=save_dir).to(data.device) # num_filters x d

    # Renormalize filter responses
    aZ = ZCA(W @ oData, False, tiny=zca_epsilon)
    W = aZ @ W

    # # L1 normalize filters
    # W = W / W.abs().sum(dim=1, keepdim=True)

    # Expand filters by having negative version
    W = torch.cat([W, -W], dim=0)

    # Add filters with all 1s and all -1s
    ones_filter = torch.ones_like(W[:1])
    ones_filter = ones_filter / ones_filter.abs().sum(dim=1, keepdim=True)
    W = torch.cat([W, ones_filter, -ones_filter], dim=0)

    # Plot covariance matrix before merge
    cov = torch.cov(W_before_merge @ data)
    cov_error = compute_error(cov).item()
    plt.figure()
    plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
    plt.title(f'Covariance on patches before merge (Error: {cov_error:.5e})')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,'covariance_on_patches_before_merge.jpg'), bbox_inches='tight')
    plt.close()
    
    # Plot covariance matrix after merge
    cov = torch.cov(W @ oData)
    cov_error = compute_error(cov).item()
    plt.figure()
    plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
    plt.title(f'Covariance on patches after merge (Error: {cov_error:.5e})')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,'covariance_on_patches_after_merge.jpg'), bbox_inches='tight')
    plt.close()

    # Get bias
    bias = -(W @ oData).mean(dim=1)

    # reshape filters
    W = einops.rearrange(W, 'k (c h w) -> k c h w', c=3, h=filt_size, w=filt_size)

    # back to single precision
    W = W.float()
    bias = bias.float()

    return W, bias

def add_gray_channels(data, num_colors, filt_size):
    gray = einops.rearrange(data, '(c h w) n -> c (h w) n', c=num_colors, h=filt_size, w=filt_size).mean(dim=0)
    data_with_gray = torch.cat([data, gray], dim=0)
    # Inverse transform for getting grayscale filters
    C_Trans = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=data.dtype)
    C_Trans[-1, :] /= 3
    C_Trans = torch.pinverse(C_Trans)
    C_Trans = C_Trans.to(data.device)

    return data_with_gray, C_Trans

def ZCA(data, real_cov, tiny=1e-6):
    if real_cov:
        C = torch.cov(data)
    else:
        C = (data @ data.T) / data.size(1)
    
    D, V = torch.linalg.eigh(C)
    sorted_indices = torch.argsort(D, descending=True)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    D = torch.clamp(D, min=0)  # Replace negative values with zero
    Di = (D + tiny)**(-0.5)
    Di[~torch.isfinite(Di)] = 0
    zca_trans = V @ torch.diag(Di) @ V.T

    return zca_trans

def mergeFilters(W, filt_size, num_colors, d_size, C_Trans, enforce_symmetry, plot_all=False, save_dir=None):
    center = np.ceil(filt_size / 2).astype(int) - 1 # minus 1 because python indexing starts at 0 
    npca = W.size(0)
    F = W.view(npca, num_colors, filt_size, filt_size)
    F = einops.rearrange(F, 'k c h w -> k h w c') # put channels last
    F_centered = torch.zeros((npca, d_size), dtype=W.dtype)
    
    all_centroidX = []
    all_centroidY = []
    for k in range(npca):
        # get centroids
        F_k = F[k]
        a_k = F_k.abs().sum(dim=2)
        centroidX = a_k.sum(dim=0).argmax()
        centroidY = a_k.sum(dim=1).argmax()
        all_centroidX.append(centroidX.item())
        all_centroidY.append(centroidY.item())
        # get shifts
        shiftX = center - centroidX.item()
        shiftY = center - centroidY.item()
        # center filters
        F_k_c = torch.roll(F_k, shifts=(shiftY, shiftX), dims=(0, 1)) 
        # enforce symmetry
        if enforce_symmetry:
            for j in range(num_colors):
                F_k_c[:, :, j] = (F_k_c[:, :, j] + F_k_c[:, :, j].T) / 2
        # Convert grayscale filters back to RGB
        if C_Trans is not None:
            temp = torch.zeros((filt_size, filt_size, 3), dtype=W.dtype)
            for j in range(3):
                temp[:, :, j] = torch.sum(C_Trans[j].view(1, 1, -1) * F_k_c, dim=2)
            F_k_c = temp
        # back to channels first
        F_k_c = einops.rearrange(F_k_c, 'h w c -> c h w')
        # flaten and accumulate
        F_centered[k, :] = F_k_c.reshape(-1)

    # normalize
    F_norm = F_centered / torch.sqrt((F_centered**2).sum(dim=1, keepdim=True))

    # take filters that are not on the edge
    all_centroidX = torch.tensor(all_centroidX)
    all_centroidY = torch.tensor(all_centroidY)
    F_norm_no_edge= F_norm[(all_centroidY != 0) & (all_centroidY != filt_size-1) & (all_centroidX != 0) & (all_centroidX != filt_size-1), :]

    if F_norm_no_edge.shape[0] == num_colors:
        filters = F_norm_no_edge
    else:
        # rise error
        assert F_norm_no_edge.shape[0] == num_colors, 'Need clustering (or handcraft merge) to merge filters. Not implemented yet'

    # plot all filters in F_norm
    if plot_all:
        assert save_dir is not None; 'save_dir must be provided to plot_all'
        plt.figure(figsize=(10,10))
        aux = int(np.sqrt(F_norm.shape[0])) +1
        for i in range(aux*aux):
            if i >= F_norm.shape[0]:
                break
            filter_m = F_norm[i].reshape(3, filt_size, filt_size)
            f_min, f_max = filter_m.min(), filter_m.max()
            filter_m = (filter_m - f_min) / (f_max - f_min) # make them from 0 to 1
            filter_m = filter_m.cpu().numpy().transpose(1,2,0) # put channels last
            plt.subplot(aux,aux,i+1)
            plt.imshow(filter_m)          
            plt.axis('off')
            i +=1
        plt.savefig(os.path.join(save_dir,'ZCA_filters_all_before_merge.jpg'),dpi=300,bbox_inches='tight')
        plt.close()

    return filters

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