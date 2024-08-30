import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from resnet_gn_mish import *

from torchvision import transforms
from torchvision import datasets

from scipy.ndimage import uniform_filter

from scipy.stats import gennorm
from scipy.special import gamma, gammainc

import torch.nn.functional as F
from scipy.stats import norm
import cv2
from multiprocessing import Pool, cpu_count

import json, os, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def fit_gennorm(feature_map):
    return gennorm.fit(feature_map)
    
def fit_gennorm_to_batch(feats):
    batch_size, num_filters, height, width = feats.shape
    
    # Flatten the feature maps for all filters in the batch
    flattened_feats = feats.view(batch_size * height * width, num_filters).cpu().numpy()
    # print(flattened_feats.shape)   # (..., 10)

    with Pool(processes=int(cpu_count()/2)) as pool:
        ggd_params = pool.map(fit_gennorm, [flattened_feats[:, j] for j in range(num_filters)])
    
    return ggd_params

def compute_saliency_map_ggd_batch(features, ggd_params, weighted=False):
    features = features.to(torch.float64)
    batch_size, num_filters, height, width = features.shape

    # Compute the joint probability of all filters for each image in the batch
    saliency_maps = torch.ones((batch_size, height, width), dtype=torch.float64)
    for i in range(num_filters):
        theta, loc, sigma = ggd_params[i]
        if weighted:
            theta_inv = 1.0 / theta
            gamma_incomplete = gammainc(theta_inv, (torch.abs(features[:, i, :, :]) ** theta) * (sigma ** -theta))
            gamma_func = gamma(theta_inv)
            improved_feats = gamma_incomplete / gamma_func
            p = gennorm.pdf(improved_feats.flatten(), theta, loc, sigma)
        else:
            p = gennorm.pdf(features[:, i, :, :].flatten(), theta, loc, sigma)
        p = p.reshape(batch_size, height, width)
        saliency_maps *= torch.tensor(p, dtype=torch.float64)
    saliency_maps = 1 / (saliency_maps + 1e-5)
    # for b in range(batch_size):
    #     saliency_maps[b] = torch.from_numpy(cv2.GaussianBlur(saliency_maps[b].numpy(), (15, 15), 0))
    saliency_maps /= saliency_maps.sum(dim=(1, 2), keepdim=True)
    saliency_maps = saliency_maps.to(torch.float32)
    return saliency_maps

def mean_pooling_batch(feats, kernel_size, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    pooled_feats = F.avg_pool2d(feats, kernel_size, stride=stride, padding=padding)
    return pooled_feats

def l2_pooling_batch(feats, kernel_size, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    squared_feats = feats ** 2
    pooled_feats = F.avg_pool2d(squared_feats, kernel_size, stride=stride, padding=padding)#, divisor_override=1) # divisor_override=1 to do sum pooling
    # when using divisor_override=1, it causes the saliency map to flip probabilities for some reason. numerical issue
    return torch.sqrt(pooled_feats)

def resize_saliency_map_batch(saliency_maps, original_shape=(224,224)):
    # saliency_maps: tensor of shape (batch_size, height, width)
    saliency_maps_tensor = saliency_maps.unsqueeze(1)  # Add channel dimension
    resized_saliency_maps = F.interpolate(saliency_maps_tensor, size=original_shape, mode='bilinear', align_corners=True)
    resized_saliency_maps = resized_saliency_maps.squeeze(1)  # Remove channel dimension

    # Normalize each saliency map in the batch
    batch_sums = resized_saliency_maps.view(resized_saliency_maps.shape[0], -1).sum(dim=1)  # Compute sum of each saliency map
    resized_saliency_maps /= batch_sums.view(-1, 1, 1)  # Normalize using broadcasting
    return resized_saliency_maps

### CAT2000 dataset
def load_image(image_path, imgtransform):
    return imgtransform(Image.open(image_path).convert('RGB'))
    
def load_gt_saliency(gt_saliencty_path, maptransform):
    return maptransform(Image.open(gt_saliencty_path).convert('L'))

def load_cat2000_data(stimuli_folder, category, imgtransform, maptransform):
    images = []
    saliency_maps = []
    stimuli_category_folder = os.path.join(stimuli_folder, category)
    saliency_maps_category_folder = os.path.join(stimuli_category_folder, 'Output')
    for filename in os.listdir(stimuli_category_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(stimuli_category_folder, filename)
            saliency_map_path = os.path.join(saliency_maps_category_folder, filename.replace('.jpg', '_SaliencyMap.jpg'))
            images.append(load_image(image_path,imgtransform))
            saliency_maps.append(load_gt_saliency(saliency_map_path, maptransform))  # Load as grayscale
    return torch.stack(images), torch.stack(saliency_maps)



# seed everything
seed = 0
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

pretrained_folder = "output/resnet18_barlowtwins_zca6filters_kernel3_eps0.001/"
zca_outchannels = 6
zca_kernel_size = 3
pool_kernel_size = 8
stride = 1
weighted = True
save_dir = os.path.join(pretrained_folder, f"saliency_maps_seed{seed}")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

### Load args
with open(pretrained_folder+"args.json", "r") as f:
    args = json.load(f)

### Load dataset
data_path = args["data_path"]
print('data_path:', data_path)
mean=[0.485, 0.456, 0.406]
std=[1.0, 1.0, 1.0]
zca_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
zca_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=zca_transform)
zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=512, shuffle=True)

### Get batch of images
batch_zca_image, _ = next(iter(zca_loader))

### Load model
model = eval(args["model_name"])(num_classes=args["num_classes"], conv0_flag=True, conv0_outchannels=zca_outchannels, conv0_kernel_size=zca_kernel_size)
pretranined_model = args["model_name"] + '_best.pth'
model_state_dict = torch.load(os.path.join(pretrained_folder,pretranined_model))
model.load_state_dict(model_state_dict)
model.to('cuda')
model.eval()

### Get GGD params for a batch_zca_image (no pooling, meanpool and L2 pool)
with torch.no_grad():
    batch_zca_feats = model.conv0(batch_zca_image.to('cuda')).squeeze()
batch_abs_zca_feats = batch_zca_feats.abs().detach().cpu()
batch_meanpool_zca_feats = mean_pooling_batch(batch_abs_zca_feats, kernel_size=pool_kernel_size, stride=stride)
batch_l2pool_zca_feats = l2_pooling_batch(batch_abs_zca_feats, kernel_size=pool_kernel_size, stride=stride)

ggd_params = fit_gennorm_to_batch(batch_abs_zca_feats)
meanpool_ggd_params = fit_gennorm_to_batch(batch_meanpool_zca_feats)
l2pool_ggd_params = fit_gennorm_to_batch(batch_l2pool_zca_feats)




### Load a training batch from CAT2000
image_folder='/data/datasets/CAT2000/trainSet/Stimuli'
img_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
map_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
category = 'Action'
batch_image, batch_GT_saliency_maps = load_cat2000_data(image_folder, category, img_transform, map_transform)
batch_GT_saliency_maps = batch_GT_saliency_maps.squeeze()
batch_GT_saliency_maps /= batch_GT_saliency_maps.sum(dim=(1, 2), keepdim=True) # make gt saliency maps a probability distribution

### Get features for the batch
with torch.no_grad():
    batch_feats = model.conv0(batch_image.to('cuda')).squeeze()
batch_abs_feats = batch_feats.abs().detach().cpu()
batch_meanpool_feats = mean_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)
batch_l2pool_feats = l2_pooling_batch(batch_abs_feats, kernel_size=pool_kernel_size, stride=stride)

### Compute saliency maps for the batch
batch_saliency_maps = compute_saliency_map_ggd_batch(batch_abs_feats, ggd_params, weighted=weighted)
batch_meanpool_saliency_maps = compute_saliency_map_ggd_batch(batch_meanpool_feats, meanpool_ggd_params, weighted=weighted)
batch_l2pool_saliency_maps = compute_saliency_map_ggd_batch(batch_l2pool_feats, l2pool_ggd_params, weighted=weighted)

### Resize saliency maps to original image size (only meanpool and L2pool need this)
original_shape = (224, 224)
batch_meanpool_saliency_maps = resize_saliency_map_batch(batch_meanpool_saliency_maps, original_shape)
batch_l2pool_saliency_maps = resize_saliency_map_batch(batch_l2pool_saliency_maps, original_shape)

### Calculate distribution matching metric between saliency map and GT saliency map
def kl_divergence(probs1, probs2, eps = 1e-5):
    kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum(dim=1)
    return kl

def symmetrized_kl_divergence(p, q):
    return 0.5*(kl_divergence(p, q) + kl_divergence(q, p))


### Compute metrics
batch_GT_saliency_maps_flatten = batch_GT_saliency_maps.view(batch_GT_saliency_maps.shape[0], -1).to(torch.float64)
batch_saliency_maps_flatten = batch_saliency_maps.view(batch_saliency_maps.shape[0], -1).to(torch.float64)
batch_meanpool_saliency_maps_flatten = batch_meanpool_saliency_maps.view(batch_meanpool_saliency_maps.shape[0], -1).to(torch.float64)
batch_l2pool_saliency_maps_flatten = batch_l2pool_saliency_maps.view(batch_l2pool_saliency_maps.shape[0], -1).to(torch.float64)

kl_div = symmetrized_kl_divergence(batch_GT_saliency_maps_flatten, batch_saliency_maps_flatten)
mse = F.mse_loss(batch_GT_saliency_maps_flatten, batch_saliency_maps_flatten, reduction='none').mean(dim=1)
kl_div_meanpool = symmetrized_kl_divergence(batch_GT_saliency_maps_flatten, batch_meanpool_saliency_maps_flatten)
mse_meanpool = F.mse_loss(batch_GT_saliency_maps_flatten, batch_meanpool_saliency_maps_flatten, reduction='none').mean(dim=1)
kl_div_l2pool = symmetrized_kl_divergence(batch_GT_saliency_maps_flatten, batch_l2pool_saliency_maps_flatten)
mse_l2pool = F.mse_loss(batch_GT_saliency_maps_flatten, batch_l2pool_saliency_maps_flatten, reduction='none').mean(dim=1)

print(f"KL div: {kl_div.mean().item()}, MSE: {mse.mean().item()}")
print(f"KL div (meanpool): {kl_div_meanpool.mean().item()}, MSE (meanpool): {mse_meanpool.mean().item()}")
print(f"KL div (L2pool): {kl_div_l2pool.mean().item()}, MSE (L2pool): {mse_l2pool.mean().item()}")

# Save mean metrics in json
metrics = {
    "KL div": kl_div.mean().item(),
    "KL div (meanpool)": kl_div_meanpool.mean().item(),
    "KL div (L2pool)": kl_div_l2pool.mean().item(),
    "MSE": mse.mean().item(),
    "MSE (meanpool)": mse_meanpool.mean().item(),
    "MSE (L2pool)": mse_l2pool.mean().item()
}
if weighted:
    metrics["Weighted"] = True
metrics_name = "metrics_weighted.json" if weighted else "metrics.json"
with open(os.path.join(save_dir, metrics_name), "w") as f:
    json.dump(metrics, f, indent=4)


### Plot image and saliency map for one image in the batch
for idx in range(50):
# idx = 5 # 5 (bad) 8 (good) 23 (bad) 24 (nice)
    image = batch_image[0+idx:1+idx]

    unnorm_image = image * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
    unnorm_image = unnorm_image.squeeze().numpy()
    unnorm_image = np.moveaxis(unnorm_image, 0, -1)

    zca_image = batch_feats[idx,:3,:,:].cpu().detach().numpy()
    zca_image = (zca_image - zca_image.min()) / (zca_image.max() - zca_image.min())
    zca_image = np.moveaxis(zca_image, 0, -1)

    saliencymap_image = batch_saliency_maps[idx].numpy()
    saliencymap_meanpool_image = batch_meanpool_saliency_maps[idx].numpy()
    saliencymap_l2pool_image = batch_l2pool_saliency_maps[idx].numpy()

    # Gt saliency map
    gt_saliency_map = batch_GT_saliency_maps[idx].squeeze().numpy()
    gt_saliency_map = (gt_saliency_map - gt_saliency_map.min()) / (gt_saliency_map.max() - gt_saliency_map.min())

    # plot
    plt.figure(figsize=(24, 24))

    # plot original image
    plt.subplot(3,3,1)
    plt.imshow(unnorm_image)
    plt.title('Original', fontsize=20)
    plt.axis('off')

    # plot zca version
    plt.subplot(3,3,2)
    plt.imshow(zca_image)
    plt.title('ZCA (3 first channels)', fontsize=20)
    plt.axis('off')

    # plot mean of batch_abs of idx image
    plt.subplot(3,3,3)
    plt.imshow(batch_abs_feats[idx].mean(0).cpu().detach().numpy())
    plt.title(f'Mean Abs feat', fontsize=20)
    plt.axis('off')

    # plot saliency map
    plt.subplot(3,3,4)
    plt.imshow(saliencymap_image, cmap='gray')
    if weighted:
        plt.title(f'Weighted GGD', fontsize=20)
    else:
        plt.title(f'GGD', fontsize=20)
    plt.axis('off')

    # plot saliency map (mean pool)
    plt.subplot(3,3,5)
    plt.imshow(saliencymap_meanpool_image, cmap='gray')
    if weighted:
        plt.title(f'Weighted GGD (mean pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (mean pool)', fontsize=20)
    plt.axis('off')

    # plot saliency map (L2 pool)
    plt.subplot(3,3,6)
    plt.imshow(saliencymap_l2pool_image, cmap='gray')
    if weighted:
        plt.title(f'Weighted GGD (L2 pool, kernel: {pool_kernel_size})', fontsize=20)
    else:
        plt.title(f'GGD (L2 pool)', fontsize=20)
    plt.axis('off')

    # plot GT saliency map
    plt.subplot(3,3,7)
    plt.imshow(gt_saliency_map, cmap='gray')
    plt.title(f'GT saliency map\nKL div: {kl_div[idx].item():.3f}, MSE: {mse[idx].item():.3e}', fontsize=20)
    plt.axis('off')

    # plot GT saliency map
    plt.subplot(3,3,8)
    plt.imshow(gt_saliency_map, cmap='gray')
    plt.title(f'GT saliency map\nKL div: {kl_div_meanpool[idx].item():.3f}, MSE: {mse_meanpool[idx].item():.3e}', fontsize=20)
    plt.axis('off')

    # plot GT saliency map
    plt.subplot(3,3,9)
    plt.imshow(gt_saliency_map, cmap='gray')
    plt.title(f'GT saliency map\nKL div: {kl_div_l2pool[idx].item():.3f}, MSE: {mse_l2pool[idx].item():.3e}', fontsize=20)
    plt.axis('off')

    # save plot
    if weighted:
        plt.savefig(os.path.join(save_dir,f"image_{idx}_saliency_weighted_pool_kernel_{pool_kernel_size}.png"), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(save_dir,f"image_{idx}_saliency_pool_kernel_{pool_kernel_size}.png"), bbox_inches='tight', dpi=300)
    plt.close()


    # Plot all feature maps of idx
    feats = batch_feats[idx]
    num_feats = feats.shape[0]
    figsize=(6*(num_feats/2), 10)
    plt.figure(figsize=figsize)
    for j in range(feats.shape[0]):
        plt.subplot(1,num_feats,j+1)
        plt.imshow(feats[j].cpu().detach().numpy())
        plt.axis('off')
    plt.savefig(os.path.join(save_dir,f"image_{idx}_zcafeats.png"), bbox_inches='tight', dpi=300)
    plt.close()



