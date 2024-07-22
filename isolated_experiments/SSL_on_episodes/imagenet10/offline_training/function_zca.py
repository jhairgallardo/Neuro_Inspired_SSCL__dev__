import torch
import torchvision
import matplotlib.pyplot as plt
import einops

import os, json
import numpy as np
from copy import deepcopy

def calculate_ZCA_conv0_weights(model, dataset, nimg = 10000, zca_epsilon=1e-6, save_dir=None):

    # create clone of conv0
    conv0 = deepcopy(model.conv0)

    # get images
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=nimg, shuffle=True)
    imgs,_ = next(iter(train_loader))

    # save imgs channel mean in json format
    mean = imgs.mean(dim=(0,2,3)).tolist()
    with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f:
        json.dump(mean, f)
    
    # extract Patches 
    kernel_size = conv0.kernel_size[0]
    patches = extract_patches(imgs, kernel_size, step=kernel_size)

    # get weight with ZCA
    weight = get_filters(patches, zca_epsilon=zca_epsilon, save_dir=save_dir)
    
    # plots
    if save_dir is not None:

        # plot final filters
        num_filters = weight.shape[0]
        plt.figure(figsize=(5*num_filters,5))
        for i in range(num_filters):
            filter_m = weight[i]
            filter_m = (filter_m - filter_m.min()) / (filter_m.max() - filter_m.min()) # make them from 0 to 1
            filter_m = filter_m.cpu().numpy().transpose(1,2,0) # put channels last
            plt.subplot(1,num_filters,i+1)
            plt.imshow(filter_m)          
            plt.axis('off')
        plt.savefig(f'{save_dir}/ZCA_filters.jpg',dpi=300,bbox_inches='tight')
        plt.close()

        # plot final filters histogram
        plt.figure(figsize=(5,5*num_filters))
        for i in range(num_filters):
            filter_m = weight[i]
            plt.subplot(num_filters,1,i+1)
            plt.hist(filter_m.flatten(), label=f'filter {i}')
            plt.legend()
        plt.savefig(f'{save_dir}/ZCA_filters_hist.jpg',bbox_inches='tight')
        plt.close()

        # plot 16 images
        mean=[0.485, 0.456, 0.406]
        std = [1.0, 1.0, 1.0]
        imgs_aux = imgs[:16]
        imgs_aux = imgs_aux * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
        grid = torchvision.utils.make_grid(imgs_aux, nrow=4)
        plt.figure(figsize=(20,20))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'{save_dir}/imgs.png', bbox_inches='tight')
        plt.close()

        # plot the zca transformed version of the 16 images
        conv0.weight.data = weight
        imgs_aux = conv0(imgs_aux)
        for i in range(imgs_aux.shape[0]):
            imgs_aux[i] = ( imgs_aux[i] - imgs_aux[i].min() )/ (imgs_aux[i].max() - imgs_aux[i].min())
        grid = torchvision.utils.make_grid(imgs_aux[:,:3,:,:], nrow=4)
        plt.figure(figsize=(20,20))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'{save_dir}/imgs_zca.png', bbox_inches='tight')
        plt.close()

        # plot channel covariance matrix of zca transformed images
        imgs_aux = conv0(imgs)
        imgs_aux = einops.rearrange(imgs_aux, 'b c h w -> c (b h w)')
        cov = torch.cov(imgs_aux)
        error = compute_error(cov)
        plt.figure()
        plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
        plt.title(f'Error: {error:.5e}')
        plt.colorbar()
        plt.savefig(f'{save_dir}/channel_covariance_matrix_zca_images.jpg', bbox_inches='tight')
        plt.close()

    return weight

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, zca_epsilon=1e-6, save_dir=None):
    _, n_channels, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # data is a d x N
    data = data.double()

    # Plot covariance matrix of patches
    if save_dir is not None:
        cov = torch.cov(data)
        error = compute_error(cov)
        plt.figure()
        plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
        plt.title(f'Error: {error:.5e}')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir,'covariance_matrix_original_patches.jpg'), bbox_inches='tight')
        plt.close()

    # Compute ZCA
    W = ZCA(data, epsilon = zca_epsilon).to(data.device) # shape: k (c h w)

    # Plot covariance matrix of zca transformed patches using all filters
    if save_dir is not None:
        cov = torch.cov(W @ data)
        error = compute_error(cov)
        plt.figure()
        plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
        plt.title(f'Error: {error:.5e}')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir,'covariance_matrix_raw_zca_filters_on_zca_patches.jpg'), bbox_inches='tight')
        plt.close()

    # Reshape filters
    W = einops.rearrange(W, 'k (c h w) -> k c h w', c=n_channels, h=filt_size, w=filt_size)

    # Plot all raw filters
    if save_dir is not None:
        num_filters = W.shape[0]
        num_rows = int(num_filters / (filt_size**2))
        num_columns = int(filt_size**2)
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

    # Plot histogram of all raw filters
    if save_dir is not None:
        plt.figure(figsize=(num_columns*3,num_rows*3))
        for i in range(num_filters):
            filter_m = W[i]
            plt.subplot(num_rows,num_columns,i+1)
            plt.hist(filter_m.flatten(), label=f'filter {i}')
            plt.legend()
        plt.savefig(os.path.join(save_dir,'zca_filters_all_raw_hist.jpg'), bbox_inches='tight')
        plt.close()

    # Pick centered filters only (filter is in the middle of the patch)
    W_center = torch.zeros((n_channels, n_channels, filt_size, filt_size), dtype=W.dtype)
    for i in range(n_channels):
        index = int( (filt_size*filt_size)*(i) + (filt_size*filt_size-1)/2 ) # index of the filter that is in the center of the patch
        W_center[i, :] = W[index, :, :, :]

    # Expand filters by having negative version
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