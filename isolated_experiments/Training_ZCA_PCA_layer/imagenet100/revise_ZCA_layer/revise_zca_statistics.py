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
    W = ZCA(data, real_cov, epsilon = zca_epsilon).to(data.device)
    W = mergeFilters(W, filt_size, num_colors, d_size, C_Trans, enforce_symmetry, plot_all=False, save_dir=save_dir).to(data.device) # num_filters x d
    # Renormalize filter responses
    aZ = ZCA(W @ oData, False, epsilon=zca_epsilon)
    W = aZ @ W
    # Expand filters by having negative version
    W = torch.cat([W, -W], dim=0)
    # Add filters with all 1s and all -1s
    ones_filter = torch.ones_like(W[:1])
    ones_filter = ones_filter / ones_filter.abs().sum(dim=1, keepdim=True)
    W = torch.cat([W, ones_filter, -ones_filter], dim=0)
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

def ZCA(data, real_cov, epsilon=1e-6):
    if real_cov:
        C = torch.cov(data)
    else:
        C = (data @ data.T) / data.size(1)
    D, V = torch.linalg.eigh(C)
    sorted_indices = torch.argsort(D, descending=True)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    D = torch.clamp(D, min=0)  # Replace negative values with zero
    Di = (D + epsilon)**(-0.5)
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
        filters = torch.zeros((num_colors, F_norm_no_edge.shape[-1]), dtype=W.dtype)
        group_num = (filt_size-2)**2
        for i in range(num_colors):
            filters[i,:] = F_norm_no_edge[i*group_num:(i+1)*group_num].mean(dim=0)
    return filters

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


###############################################################################################################################

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
conv0_outchannels=10
conv0_kernel_size=3
nimg = 10000
zca_epsilon = 5e-4 # 5e-3 5e-4 5e-6 5e-7
addgray = True
init_bias = True
save_dir = f'output/{aug_type}aug'
if init_bias:
    save_dir += '_initbias'
save_dir += f'_{conv0_kernel_size}kernerlsize_{conv0_outchannels}channels_{zca_epsilon}eps'
os.makedirs(save_dir, exist_ok=True)





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
train_dataset = datasets.ImageFolder(root="/data/datasets/ImageNet-100/train", transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
train_imgs,train_labels = next(iter(train_loader))





### Calculate ZCA layer
zca_layer = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', bias=True)
act = nn.Mish()
zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
zca_dataset = datasets.ImageFolder(root="/data/datasets/ImageNet-100/train", transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=nimg, shuffle=True)
zca_imgs,_ = next(iter(zca_loader))
# save imgs channel mean in json format
zca_input_mean = zca_imgs.mean(dim=(0,2,3)).tolist()
with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f:
    json.dump(zca_input_mean, f)
patches = extract_patches(zca_imgs, conv0_kernel_size, step=conv0_kernel_size)
# get weight and bias with ZCA
weight, bias = get_filters(patches, 
                            real_cov=False,
                            add_gray=addgray,
                            zca_epsilon = zca_epsilon, 
                            save_dir=save_dir)
zca_layer.weight = torch.nn.Parameter(weight)
if init_bias:
    zca_layer.bias = torch.nn.Parameter(bias)

# plot final filters
num_filters = weight.shape[0]
plt.figure(figsize=(5*num_filters,5))
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





### Pass data through zca layer (forward pass using cuda)
train_imgs = train_imgs.cuda()
zca_layer = zca_layer.cuda()
zca_layer.eval()
with torch.no_grad():
    zca_out = zca_layer(train_imgs)
    zca_out = act(zca_out)
    zca_out = zca_out.cpu()
# save zca_out
np.save(f'{save_dir}/zca_out_raw_values.npy', zca_out.numpy())





### Save input images
imgs = train_imgs.cpu().detach()
imgs = imgs * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
grid = torchvision.utils.make_grid(imgs, nrow=8)
plt.figure(figsize=(20,20))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title(f'{aug_type}', fontsize=20)
plt.savefig(f'{save_dir}/input_imgs.png', bbox_inches='tight')
plt.close()

if conv0_outchannels==3:
    ### Save zca_out images
    zca_out = zca_out * torch.tensor(std).view(1,3,1,1) + torch.tensor(mean).view(1,3,1,1)
    grid = torchvision.utils.make_grid(zca_out, nrow=8)
    plt.figure(figsize=(20,20))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'{aug_type}', fontsize=20)
    plt.savefig(f'{save_dir}/zca_out_imgs.png', bbox_inches='tight')
    plt.close()




### Save the mean of the absolute values of the zca_out
zca_out_abs = zca_out.abs()
zca_out_abs_mean = zca_out_abs.mean(1)
zca_out_abs_mean = zca_out_abs_mean.unsqueeze(1)
grid = torchvision.utils.make_grid(zca_out_abs_mean, nrow=8)
plt.figure(figsize=(20,20))
plt.imshow(grid[0])
plt.axis('off')
plt.title(f'{aug_type}', fontsize=20)
plt.savefig(f'{save_dir}/zca_out_mean_abs.png', bbox_inches='tight')
plt.close()






### Save total statictics (mean, std, max, min)
zca_out_mean = zca_out.mean().item()
zca_out_std = zca_out.std().item()
zca_out_max = zca_out.max().item()
zca_out_min = zca_out.min().item()
with open(f'{save_dir}/total_statistics_zca_out.json', 'w') as f:
    json.dump({'mean': zca_out_mean, 'std': zca_out_std, 'max': zca_out_max, 'min': zca_out_min}, f)





### Violin plots of zca
fig, ax = plt.subplots()
ax.violinplot(zca_out.view(-1).numpy(), showmeans=False, showmedians=True)
ax.set_title('ZCA out')
ax.set_xlabel(f'{aug_type}')
ax.set_ylabel('Values')
plt.savefig(f'{save_dir}/violin_zca_out.png')
plt.close()


# ### Get covariance matrix and plot it
# zca_out_aux = zca_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
# cov = torch.cov(zca_out_aux.T)
# plt.imshow(cov)
# plt.colorbar()
# plt.savefig(f'{save_dir}/covariance_matrix_zca_out.png')
# plt.close()


# # Plot histogram per channel
# for i in range(conv0_outchannels):
#     plt.hist(zca_out_aux[:,i], bins=100)
#     plt.savefig(f'{save_dir}/histogram_zca_out_channel_{i}.png')
#     plt.close()

# ### save statistics of zca out per channel (mean, std, max, min)
# zca_out_mean = zca_out.mean(axis=(0,2,3)).tolist()
# zca_out_std = zca_out.std(axis=(0,2,3)).tolist()
# zca_out_max = zca_out.max(axis=(0,2,3)).tolist()
# zca_out_min = zca_out.min(axis=(0,2,3)).tolist()
# with open(f'{save_dir}/mean_zca_out.json', 'w') as f:
#     json.dump(zca_out_mean, f)
# with open(f'{save_dir}/std_zca_out.json', 'w') as f:
#     json.dump(zca_out_std, f)
# with open(f'{save_dir}/max_zca_out.json', 'w') as f:
#     json.dump(zca_out_max, f)
# with open(f'{save_dir}/min_zca_out.json', 'w') as f:
#     json.dump(zca_out_min, f)

    
    

