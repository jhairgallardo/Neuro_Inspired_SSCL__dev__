import os
import json

import torch
import torch.nn as nn
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

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, zca_epsilon=1e-6, norm=False, save_dir=None):
    _, n_channels, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # shape: d N
    data = data.double()

    # Plot covariance matrix of patches
    if save_dir is not None:
        cov = torch.cov(data)
        plt.figure()
        plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir,'covariance_matrix_original_patches.jpg'), bbox_inches='tight')
        plt.close()

    # Compute ZCA
    W = ZCA(data, epsilon = zca_epsilon).to(data.device) # shape: k (c h w)

    # Normalize filters
    if norm:
        W = W / torch.sqrt((W**2).sum(dim=1, keepdim=True))

    # Plot covariance matrix on all raw zca filters
    if save_dir is not None:
        cov = torch.cov(W @ data)
        plt.figure()
        plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
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

    # add negative version of filters
    W_center = torch.cat((W_center, -W_center), dim=0)

    # Get bias
    W_center = einops.rearrange(W_center, 'k c h w -> k (c h w)')
    bias = -(W_center @ data).mean(dim=1)
    # reshape filters
    W_center = einops.rearrange(W_center, 'k (c h w) -> k c h w', c=3, h=filt_size, w=filt_size)

    # back to single precision
    W_center = W_center.float()
    bias = bias.float()

    return W_center, bias

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
aug_type = 'none' # 'none' 'colorjitter' 'grayscale' 'gaussianblur' 'solarization' 'barlowtwins'
conv0_outchannels=6
conv0_kernel_size=3
nimg = 10000 # 10000 1000
zca_epsilon = 1e-3
init_bias = False
normalized_filters = False
save_dir = f'output/{conv0_outchannels}channels/{aug_type}aug'
if init_bias:
    save_dir += '_initbias'
save_dir += f'_{conv0_kernel_size}kernerlsize_{conv0_outchannels}channels_{zca_epsilon}eps'
if normalized_filters:
    save_dir += '_normalizedfilters'
os.makedirs(save_dir, exist_ok=True)
num_batch_plot = 16





### Load data
mean=[0.485, 0.456, 0.406]
std=[1.0, 1.0, 1.0]

# color jittering
if aug_type == 'colorjitter':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                    saturation=0.2, hue=0.1)],
                            p=1.0
                            ),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
# Gray scale
elif aug_type == 'grayscale':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomGrayscale(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
# GaussianBlur
elif aug_type == 'gaussianblur':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        GaussianBlur(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
# Solarization
elif aug_type == 'solarization':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        Solarization(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
# No augmentations
elif aug_type == 'none':
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
elif aug_type == 'barlowtwins':
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
    raise ValueError(f'Augmentation type {aug_type} not recognized')
train_dataset = datasets.ImageFolder(root="/data/datasets/ImageNet-10/train", transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
train_imgs,train_labels = next(iter(train_loader))
train_imgs = train_imgs.cuda()




### Calculate ZCA layer
zca_layer = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', bias=True)
act = nn.Mish()
zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
zca_dataset = datasets.ImageFolder(root="/data/datasets/ImageNet-100/train", transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=nimg, shuffle=True)
zca_input_imgs,_ = next(iter(zca_loader))
# save imgs channel mean in json format
zca_input_mean = zca_input_imgs.mean(dim=(0,2,3)).tolist()
with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f:
    json.dump(zca_input_mean, f)
patches = extract_patches(zca_input_imgs, conv0_kernel_size, step=conv0_kernel_size)
# get weight and bias with ZCA
weight, bias = get_filters(patches,
                            zca_epsilon = zca_epsilon,
                            norm=normalized_filters,
                            save_dir=save_dir)
zca_layer.weight = torch.nn.Parameter(weight)
if init_bias:
    zca_layer.bias = torch.nn.Parameter(bias)
else: # bias as zeros
    zca_layer.bias = torch.nn.Parameter(torch.zeros_like(bias))
zca_layer = zca_layer.cuda()




### Plot final filters
num_filters = weight.shape[0]
plt.figure(figsize=(5*num_filters,5))
for i in range(num_filters):
    filter_m = weight[i]
    filter_m = (filter_m - filter_m.min()) / (filter_m.max() - filter_m.min()) # make them from 0 to 1
    filter_m = filter_m.cpu().numpy().transpose(1,2,0) # put channels last
    plt.subplot(1,num_filters,i+1)
    plt.imshow(filter_m, vmax=1, vmin=0)          
    plt.axis('off')
plt.savefig(f'{save_dir}/zca_filters.jpg',dpi=300,bbox_inches='tight')
plt.close()




### Plot final filters values histogram
plt.figure(figsize=(5,5*num_filters))
for i in range(num_filters):
    filter_m = weight[i]
    plt.subplot(num_filters,1,i+1)
    plt.hist(filter_m.flatten(), label=f'filter {i}')
    plt.legend()
plt.savefig(f'{save_dir}/zca_filters_hist.jpg',bbox_inches='tight')
plt.close()




### Plot channel covariance matrix on zca input imgs
zca_input_imgs_aux = zca_input_imgs.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(zca_input_imgs_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/channel_covariance_matrix_zca_input_imgs.png')
plt.close()




### Plot channel covariance matrix on zca out of zca input imgs
zca_input_imgs_aux = zca_input_imgs.cuda()
zca_out_conv_aux = zca_layer(zca_input_imgs_aux)
zca_out_conv_aux = zca_out_conv_aux.cpu().detach()
zca_out_conv_aux = zca_out_conv_aux.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(zca_out_conv_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/channel_covariance_matrix_zca_out_of_zca_input_imgs.png')
plt.close()




### Pass batch of data through zca layer (forward pass using cuda)
zca_layer.eval()
with torch.no_grad():
    batch_zca_out = zca_layer(train_imgs)
    batch_zca_act_out = act(batch_zca_out)
    batch_zca_act_out = batch_zca_act_out.cpu()
    batch_zca_out = batch_zca_out.cpu()
# save zca_out_act
np.save(f'{save_dir}/batch_zca_act_out_raw_values.npy', batch_zca_act_out.numpy())





### Save input images
imgs = train_imgs[:num_batch_plot].cpu().detach()
imgs = imgs * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
grid = torchvision.utils.make_grid(imgs, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title(f'{aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_imgs.png', bbox_inches='tight')
plt.close()

### Plot ZCA transformed images
batch_zca_out_aux = batch_zca_out[:num_batch_plot] # * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
for i in range(num_batch_plot):
    batch_zca_out_aux[i] = ( batch_zca_out_aux[i] - batch_zca_out_aux[i].min() )/ (batch_zca_out_aux[i].max() - batch_zca_out_aux[i].min())
grid = torchvision.utils.make_grid(batch_zca_out_aux[:,:3,:,:], nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title(f'{aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_zca_out_imgs.png', bbox_inches='tight')
plt.close()


### Save the mean of the absolute values of the batch_zca_act_out
batch_zca_act_out_abs = batch_zca_act_out[:num_batch_plot].abs()
batch_zca_act_out_abs_mean = batch_zca_act_out_abs.mean(1)
batch_zca_act_out_abs_mean = batch_zca_act_out_abs_mean.unsqueeze(1)
grid = torchvision.utils.make_grid(batch_zca_act_out_abs_mean, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid[0])
plt.axis('off')
plt.title(f'{aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_zca_act_out_mean_abs.png', bbox_inches='tight')
plt.close()


### Save total statictics (mean, std, max, min)
batch_zca_act_out_mean = batch_zca_act_out.mean().item()
batch_zca_act_out_std = batch_zca_act_out.std().item()
batch_zca_act_out_max = batch_zca_act_out.max().item()
batch_zca_act_out_min = batch_zca_act_out.min().item()
with open(f'{save_dir}/total_statistics_batch_zca_act_out.json', 'w') as f:
    json.dump({'mean': batch_zca_act_out_mean, 
               'std': batch_zca_act_out_std, 
               'max': batch_zca_act_out_max, 
               'min': batch_zca_act_out_min}, f)


### Violin plots of zca
fig, ax = plt.subplots()
ax.violinplot(batch_zca_act_out.view(-1).numpy(), showmeans=False, showmedians=True)
ax.set_title('ZCA act out')
ax.set_xlabel(f'{aug_type}')
ax.set_ylabel('Values')
plt.savefig(f'{save_dir}/violin_batch_zca_act_out.png')
plt.close()


### Get channel covariance matrix of batch
train_imgs_aux = train_imgs.permute(0,2,3,1).reshape(-1, 3).cpu()
cov = torch.cov(train_imgs_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/channel_covariance_matrix_batch.png')
plt.close()


### Get covariance matrix and plot it on current batch
batch_zca_out_aux = batch_zca_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(batch_zca_out_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/channel_covariance_matrix_zca_out_of_batch.png')
plt.close()

