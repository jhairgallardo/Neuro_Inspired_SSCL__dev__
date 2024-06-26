import os
import json

import torch
import torch.nn as nn

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
aug_type = 'barlowtwins'

conv0_outchannels=10
conv0_kernel_size=3
nimg = 10000
zca_epsilon = 5e-4
addgray = True
init_bias = True

save_dir = f'output/{aug_type}aug'
if init_bias:
    save_dir += '_initbias'
os.makedirs(save_dir, exist_ok=True)


### Load data
mean=[0.485, 0.456, 0.406]
std=[1.0, 1.0, 1.0]
if aug_type == "default":
    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
elif aug_type == "barlowtwins":
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
elif aug_type == "no_aug":
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
else:
    raise ValueError("Define correct augmentation type")
train_dataset = datasets.ImageFolder(root="/data/datasets/ImageNet-100/train", transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=16, pin_memory=True)


### Calculate ZCA layer
zca_layer = nn.Conv2d(3, conv0_outchannels, kernel_size=conv0_kernel_size, stride=1, padding='same', bias=True)
zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
zca_dataset = datasets.ImageFolder(root="/data/datasets/ImageNet-100/train", transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=nimg, shuffle=True)
imgs,labels = next(iter(zca_loader))
# save imgs channel mean in json format
mean = imgs.mean(dim=(0,2,3)).tolist()
with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f:
    json.dump(mean, f)
patches = extract_patches(imgs, conv0_kernel_size, step=conv0_kernel_size)
# get weight and bias with ZCA
weight, bias = get_filters(patches, 
                            real_cov=False,
                            add_gray=addgray,
                            zca_epsilon = zca_epsilon, 
                            save_dir=save_dir)
zca_layer.weight = torch.nn.Parameter(weight)
if init_bias:
    zca_layer.bias = torch.nn.Parameter(bias)


### Pass data through zca layer (forward pass using cuda)
train_imgs,train_labels = next(iter(train_loader))
train_imgs = train_imgs.cuda()
zca_layer = zca_layer.cuda()
zca_layer.eval()
with torch.no_grad():
    zca_out = zca_layer(train_imgs)
    zca_out = zca_out.cpu()

### Get covariance matrix and plot it
zca_out_aux = zca_out.permute(0,2,3,1).reshape(-1, conv0_outchannels)
cov = torch.cov(zca_out_aux.T)
plt.imshow(cov)
plt.colorbar()
plt.savefig(f'{save_dir}/covariance_matrix_zca_out.png')
plt.close()


# save statistics of zca out per channel
zca_out_mean = zca_out.mean(axis=(0,2,3)).tolist()
zca_out_std = zca_out.std(axis=(0,2,3)).tolist()
with open(f'{save_dir}/mean_zca_out.json', 'w') as f:
    json.dump(zca_out_mean, f)
with open(f'{save_dir}/std_zca_out.json', 'w') as f:
    json.dump(zca_out_std, f)

# Plot histogram per channel
for i in range(conv0_outchannels):
    plt.hist(zca_out_aux[:,i], bins=100)
    plt.savefig(f'{save_dir}/histogram_zca_out_channel_{i}.png')
    plt.close()


    
    

