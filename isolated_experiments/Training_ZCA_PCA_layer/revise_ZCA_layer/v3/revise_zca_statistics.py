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

def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def get_filters(patches_data, zca_epsilon=1e-6, save_dir=None):
    _, n_channels, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # shape: d N
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

    # add negative version of filters
    W_center = torch.cat((W_center, -W_center), dim=0)

    # back to single precision
    W_center = W_center.float()

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

    # Plot centered filters
    if save_dir is not None:
        num_filters = W_center.shape[0]
        plt.figure(figsize=(5*num_filters,5))
        for i in range(num_filters):
            filter_m = W_center[i]
            filter_m = (filter_m - filter_m.min()) / (filter_m.max() - filter_m.min())
            filter_m = filter_m.cpu().numpy().transpose(1,2,0)
            plt.subplot(1,num_filters,i+1)
            plt.imshow(filter_m, vmax=1, vmin=0)
            plt.axis('off')
        plt.savefig(os.path.join(save_dir,'zca_filters.jpg'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot histogram of centered filters
    if save_dir is not None:
        plt.figure(figsize=(5,5*num_filters))
        for i in range(num_filters):
            filter_m = W_center[i]
            plt.subplot(num_filters,1,i+1)
            plt.hist(filter_m.flatten(), label=f'filter {i}')
            plt.legend()
        plt.savefig(os.path.join(save_dir,'zca_filters_hist.jpg'), bbox_inches='tight')
        plt.close()

    # Plot covariance matrix on all raw zca filters
    if save_dir is not None:
        cov = torch.cov(einops.rearrange(W, 'k c h w -> k (c h w)') @ data)
        plt.figure()
        plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir,'covariance_matrix_raw_zca_filters_on_zca_input_patches.jpg'), bbox_inches='tight')
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
batch_aug_type = 'none' # 'none' 'colorjitter' 'grayscale' 'gaussianblur' 'solarization' 'barlowtwins'
conv0_outchannels=6
conv0_kernel_size=7
nimg = 10000
zca_epsilon = 1e-4 # 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
dataset_name='ImageNet-10'
activation = 'mish' # mish, hardtanh02, hardtanh22, None
scale_zca_out = False
save_dir = f'output/{conv0_outchannels}channels'

if activation is not None:
    activation_folder = f'{activation}_act'
else:
    activation_folder = 'no_act'

if scale_zca_out:
    activation_folder += '_scaled'

save_dir += f'/{activation_folder}/{batch_aug_type}aug'
save_dir += f'_{conv0_kernel_size}kernerlsize_{conv0_outchannels}channels_{zca_epsilon}eps'
os.makedirs(save_dir, exist_ok=True)
num_batch_plot = 16

### Data to calculate ZCA layer
mean=[0.485, 0.456, 0.406]
std=[1.0, 1.0, 1.0]

### Calculate ZCA layer
zca_layer = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', bias=False)
if activation == 'hardtanh02':
    act = nn.Hardtanh(min_val=0, max_val=2)
elif activation == 'hardtanh22':
    act = nn.Hardtanh(min_val=-2, max_val=2)
elif activation == 'mish':
    act = nn.Mish()
elif activation is None:
    act = nn.Identity()
zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
zca_dataset = datasets.ImageFolder(root=f"/data/datasets/{dataset_name}/train", transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=nimg, shuffle=True)
zca_input_imgs,_ = next(iter(zca_loader))
# save imgs channel mean in json format
zca_input_mean = zca_input_imgs.mean(dim=(0,2,3)).tolist()
with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f:
    json.dump(zca_input_mean, f)
patches = extract_patches(zca_input_imgs, conv0_kernel_size, step=conv0_kernel_size)
# get weight and bias with ZCA
weight = get_filters(patches,
                    zca_epsilon = zca_epsilon,
                    save_dir=save_dir)
# if there is bias in the zca layer, make it zero
if zca_layer.bias is not None:
    zca_layer.bias = torch.nn.Parameter(torch.zeros_like(zca_layer.bias))
    zca_layer.bias.requires_grad = False
zca_layer.weight = torch.nn.Parameter(weight)
zca_layer.weight.requires_grad = False
zca_layer = zca_layer.cuda()
zca_layer.eval()

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
    if scale_zca_out:
        batch_zca_out = 2*batch_zca_out / torch.amax(batch_zca_out, dim=(1,2,3), keepdims=True)
    batch_zca_act_out = act(batch_zca_out).cpu().detach()
    batch_zca_out = batch_zca_out.cpu().detach()
# save zca_out_act
np.save(f'{save_dir}/batch_zca_act_out_raw_values.npy', batch_zca_act_out.numpy())

### Save input images
imgs = batch_train_images[:num_batch_plot]
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

### Plot the mean of the absolute values of the batch_zca_act_out
batch_zca_act_out_abs = batch_zca_act_out[:num_batch_plot].abs()
batch_zca_act_out_abs_mean = batch_zca_act_out_abs.mean(1)
batch_zca_act_out_abs_mean = batch_zca_act_out_abs_mean.unsqueeze(1)
grid = torchvision.utils.make_grid(batch_zca_act_out_abs_mean, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid[0])
plt.axis('off')
plt.title(f'{batch_aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_zca_act_out_mean_abs.png', bbox_inches='tight')
plt.close()

### Plot the mean of the absolute values of batch_zca_out
batch_zca_out_abs = batch_zca_out[:num_batch_plot].abs()
batch_zca_out_abs_mean = batch_zca_out_abs.mean(1)
batch_zca_out_abs_mean = batch_zca_out_abs_mean.unsqueeze(1)
grid = torchvision.utils.make_grid(batch_zca_out_abs_mean, nrow=4)
plt.figure(figsize=(20,20))
plt.imshow(grid[0])
plt.axis('off')
plt.title(f'{batch_aug_type}', fontsize=30)
plt.savefig(f'{save_dir}/batch_zca_out_mean_abs.png', bbox_inches='tight')
plt.close()

### Get channel covariance matrix of batch
batch_train_images_aux = batch_train_images.permute(0,2,3,1).reshape(-1, 3)
cov = torch.cov(batch_train_images_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/channel_covariance_matrix_batch.png')
plt.close()

### Get channel covariance matrix of batch zca out
batch_zca_out_aux = batch_zca_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(batch_zca_out_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/channel_covariance_matrix_batch_zca_out.png')
plt.close()

### Get channel covariance matrix of batch zca act out
batch_zca_act_out_aux = batch_zca_act_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(batch_zca_act_out_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/channel_covariance_matrix_batch_zca_act_out.png')
plt.close()

### Violin plots of zca act out
fig, ax = plt.subplots()
ax.violinplot(batch_zca_act_out.view(-1).numpy(), showmeans=False, showmedians=True)
ax.set_title('ZCA act out')
ax.set_xlabel(f'{batch_aug_type}')
ax.set_ylabel('Values')
plt.savefig(f'{save_dir}/violin_batch_zca_act_out.png')
plt.close()

# ### Violin plots of zca act out per channel
# batch_zca_act_out_aux = batch_zca_act_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
# fig, ax = plt.subplots()
# ax.violinplot(batch_zca_act_out_aux.numpy(), showmeans=False, showmedians=True)
# ax.set_title('ZCA act out per channel')
# ax.set_xlabel(f'{batch_aug_type}')
# ax.set_xticks([1, 2, 3, 4, 5, 6])
# ax.set_ylabel('Values')
# plt.savefig(f'{save_dir}/violin_batch_zca_act_out_per_channel.png')
# plt.close()

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
        img = batch_train_images[top5_max_index[i]]
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
        plt.scatter(range(len(img_zca_act_out)), img_zca_act_out, marker='o')
        plt.title(f'image {top5_max_index[i]} zca act out\nchannel {channel+1}')
        plt.xlabel('Flattened index')
        plt.ylabel('value')
        plt.savefig(f'{save_dir_max_values}/channel_batch_zca_act_out_channel{channel+1}_image{top5_max_index[i]}_zca_act_out.png')
        plt.close()








