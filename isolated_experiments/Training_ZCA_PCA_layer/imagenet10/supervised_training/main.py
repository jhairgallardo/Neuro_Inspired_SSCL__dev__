import os
import logging
import time
from datetime import timedelta

import torch
import torch.nn as nn
from copy import deepcopy

from torchvision import transforms
from torchvision import datasets

from resnet_gn_mish import *
from tqdm import tqdm

import numpy as np
import torch.backends.cudnn as cudnn
import random
import argparse
import json

from function_zca import calculate_ZCA_conv0_weights, scaled_filters

from PIL import ImageOps, ImageFilter
import matplotlib.pyplot as plt

class ElapsedFilter(logging.Filter):
    def __init__(self):
        self.start_time = time.time()
    
    def filter(self, record):
   
        elapsed_seconds = record.created - self.start_time
        #using timedelta here for convenient default formatting
        elapsed = timedelta(seconds = elapsed_seconds)
        record.elapsed=elapsed
        return True

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

def plot_filters_and_hist(filters, name, save_dir):
    # plot filters
    num_filters = filters.shape[0]
    plt.figure(figsize=(5*num_filters,5))
    for i in range(num_filters):
        filter_m = deepcopy(filters[i])
        filter_m = (filter_m - filter_m.min()) / (filter_m.max() - filter_m.min())
        filter_m = filter_m.cpu().numpy().transpose(1,2,0)
        plt.subplot(1,num_filters,i+1)
        plt.imshow(filter_m, vmax=1, vmin=0)
        plt.axis('off')
    plt.savefig(os.path.join(save_dir,f'{name}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    # plot hist of filters
    plt.figure(figsize=(5,5*num_filters))
    for i in range(num_filters):
        filter_m = filters[i]
        plt.subplot(num_filters,1,i+1)
        plt.hist(filter_m.flatten(), label=f'filter {i}')
        plt.legend()
    plt.savefig(os.path.join(save_dir,f'{name}_hist.jpg'), bbox_inches='tight')
    plt.close()

    return None

def plot_zca_layer_output_hist(zca_layer, imgs, name, save_dir):
    zca_layer.eval()
    zca_layer_output = zca_layer(imgs)
    zca_layer_output = zca_layer_output.flatten().cpu().numpy()
    plt.figure(figsize=(18,6))
    plt.subplot(2,1,1)
    plt.hist(zca_layer_output, bins=100)
    plt.ylabel('frequency')
    plt.title(name)
    plt.subplot(2,1,2)
    plt.hist(zca_layer_output, bins=100)
    plt.yscale('log')
    plt.ylabel('log scale frequency')
    plt.savefig(f'{save_dir}/{name}.png', bbox_inches='tight')
    plt.close()
    return None

def plot_zca_layer_act_output_hist(zca_layer, act, imgs, name, save_dir):
    zca_layer.eval()
    zca_layer_output = zca_layer(imgs)
    zca_layer_output = act(zca_layer_output)
    zca_layer_output = zca_layer_output.flatten().cpu().numpy()
    plt.figure(figsize=(18,6))
    plt.subplot(2,1,1)
    plt.hist(zca_layer_output, bins=100)
    plt.ylabel('frequency')
    plt.title(name)
    plt.subplot(2,1,2)
    plt.hist(zca_layer_output, bins=100)
    plt.yscale('log')
    plt.ylabel('log scale frequency')
    plt.savefig(f'{save_dir}/{name}.png', bbox_inches='tight')
    plt.close()
    return None

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_path', type=str, default="/data/datasets/ImageNet-10")
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--model_name', type=str, default="resnet18")
parser.add_argument('--zero_init_residual', action='store_true', default=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--aug_type', type=str, default="default", choices=["default", "barlowtwins"])

parser.add_argument('--zca', action='store_true')
parser.add_argument('--epsilon', type=float, default=1e-4)
parser.add_argument('--zca_num_imgs', type=int, default=10000)
parser.add_argument('--zca_num_channels', type=int, default=6)
parser.add_argument('--zca_act_out', type=str, default='mish', choices=['noact', 'mish', 'tanh', 'mishtanh', 'relutanh', 'softplustanh'])
parser.add_argument('--zca_scale_filter', action='store_true')
parser.add_argument('--zca_kernel_size', type=int, default=3)

parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="output")
args = parser.parse_args()

### Seed everything
seed = args.seed 
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

### Create save dir
save_dir = os.path.join(args.save_dir, f"{args.model_name}")

if args.aug_type == "barlowtwins":
    save_dir += "_barlowtwins"

if args.zca: 
    save_dir += f"_zca{args.zca_num_channels}filters_kernel{args.zca_kernel_size}_eps{args.epsilon}"
    if args.zca_scale_filter:
        save_dir += "_scaled"
    save_dir += f"_{args.zca_act_out}"
save_dir += f"_seed{seed}"
os.makedirs(save_dir, exist_ok=True)
args.save_dir = save_dir

### Save args as json file
with open(os.path.join(save_dir,"args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

### Init logging
logging.basicConfig(filename=os.path.join(save_dir, "Log.log"), 
                    level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d Elapsed: %(elapsed)s, %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
logging.getLogger().addFilter(ElapsedFilter())

### Print args
text_print = f"args: {args}"
print(text_print), logging.info(text_print)

### Load dataset
args.mean=[0.485, 0.456, 0.406]
args.std=[0.229, 0.224, 0.225]
if args.zca: args.std = [1.0, 1.0, 1.0]

if args.aug_type == "default":
    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std)])
elif args.aug_type == "barlowtwins":
    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                    saturation=0.2, hue=0.1)],
                            p=0.8
                            ),
                        transforms.RandomGrayscale(p=0.2),
                        GaussianBlur(p=0.1),
                        Solarization(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std)])
else:
    raise ValueError("Define correct augmentation type")

transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std)])
train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=transform_train)
val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=transform_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                            shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=args.workers, pin_memory=True)

### Load model
if args.zca_act_out == 'mish':
    act = nn.Mish()
elif args.zca_act_out == 'tanh':
    act = nn.Tanh()
elif args.zca_act_out == 'mishtanh':
    act = nn.Sequential(nn.Mish(), nn.Tanh())
elif args.zca_act_out == 'relutanh':
    act = nn.Sequential(nn.ReLU(), nn.Tanh())
elif args.zca_act_out == 'softplustanh':
    act = nn.Sequential(nn.Softplus(), nn.Tanh())
elif args.zca_act_out == 'noact':
    act = nn.Identity()
else:
    raise ValueError("Activation not found")

if 'resnet' in args.model_name:
    model = eval(args.model_name)(num_classes=args.num_classes, 
                                zero_init_residual=args.zero_init_residual, 
                                conv0_flag= args.zca, 
                                conv0_outchannels=args.zca_num_channels,
                                conv0_kernel_size=args.zca_kernel_size,
                                act0=act)
else:
    raise ValueError("Model not found")

### Calculate filters and load it to the model
if args.zca:
    zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=args.mean, std=args.std)])
    zca_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=zca_transform)
    zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=args.zca_num_imgs, shuffle=True)
    zca_input_imgs,_ = next(iter(zca_loader))
    mean_of_inputs = zca_input_imgs.mean(dim=(0,2,3)).tolist()
    with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f: json.dump(mean_of_inputs, f)
    weight = calculate_ZCA_conv0_weights(imgs=zca_input_imgs, kernel_size=args.zca_kernel_size, zca_epsilon=args.epsilon)

    if args.zca_scale_filter:
        aux_conv0 = torch.nn.Conv2d(3, weight.shape[0], kernel_size=args.zca_kernel_size, stride=1, padding='same', bias=False)
        aux_conv0.weight = torch.nn.Parameter(weight)
        aux_conv0.weight.requires_grad = False
        weight = scaled_filters(aux_conv0, imgs=zca_input_imgs)
        del aux_conv0

    if args.zca_num_channels>=6:
        weight = torch.cat([weight, -weight], dim=0)
    
    ### Load weights into model conv0 (zca layer)
    model.conv0.weight = torch.nn.Parameter(weight)
    model.conv0.weight.requires_grad = False

    ### Plot stats of zca layer
    plot_filters_and_hist(model.conv0.weight, 'ZCA_filters', save_dir)
    plot_zca_layer_output_hist(model.conv0, zca_input_imgs, 'ZCA_layer_output', save_dir)
    plot_zca_layer_act_output_hist(model.conv0,act, zca_input_imgs, 'ZCA_layer_act_output', save_dir)
        
    del zca_input_imgs, zca_dataset, zca_loader

### Add model to GPU
model = torch.nn.DataParallel(model)
model.to("cuda")

### Load optimizer and criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr/1000, total_iters=args.warmup_epochs)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

### Save init model
state_dict = model.module.state_dict()
torch.save(state_dict, os.path.join(save_dir, f"{args.model_name}_init.pth"))

### Train model
best_acc = 0
train_loss_all = []
train_accuracy_all = []
val_loss_all = []
val_accuracy_all = []
for epoch in range(args.epochs):
        
    ## Train step ##
    model.train()
    corrects = 0
    total = 0
    loss_sum = 0
    with tqdm(train_loader) as tepoch:
        for data, target in tepoch:
            data, target = data.to("cuda"), target.to("cuda")
            # forward backward pass
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # collect stats
            loss_sum += loss.item()
            corrects += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
            # update progress bar
            tepoch.set_postfix(loss=loss.item(), accuracy=100*corrects/total)
    train_loss = loss_sum/len(train_loader)
    train_accuracy = corrects/total*100

    ## Validation step ##
    model.eval()
    corrects_val = 0
    total_val = 0
    loss_sum_val = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to("cuda"), target.to("cuda")
            # forward pass
            output = model(data)
            loss = criterion(output, target)
            # collect stats
            loss_sum_val += loss.item()
            corrects_val += output.argmax(dim=1).eq(target).sum().item()
            total_val += target.size(0)
    val_loss = loss_sum_val/len(val_loader)
    val_accuracy = corrects_val/total_val*100

    text_print = (f'Epoch {epoch} -- lr: {scheduler.get_last_lr()[0]:.6f} -- '
                    f'train_loss: {train_loss:.6f}, '
                    f'train_accuracy: {train_accuracy:.2f}% -- '
                    f'val_loss: {val_loss:.6f}, '
                    f'val_accuracy: {val_accuracy:.2f}%')
    
    # check best acc and save best model
    if corrects_val/total_val > best_acc:
        best_acc = corrects_val/total_val
        best_epoch = epoch
        state_dict = model.module.state_dict()
        torch.save(state_dict, os.path.join(save_dir, f"{args.model_name}_best.pth"))
        text_print += " (Saved)"
    print(text_print), logging.info(text_print)

    # accumulate all stats
    train_loss_all.append(train_loss)
    train_accuracy_all.append(train_accuracy)
    val_loss_all.append(val_loss)
    val_accuracy_all.append(val_accuracy)

    np.save(os.path.join(save_dir, "train_loss_all.npy"), np.array(train_loss_all))
    np.save(os.path.join(save_dir, "train_accuracy_all.npy"), np.array(train_accuracy_all))
    np.save(os.path.join(save_dir, "val_loss_all.npy"), np.array(val_loss_all))
    np.save(os.path.join(save_dir, "val_accuracy_all.npy"), np.array(val_accuracy_all))

    scheduler.step()
    
# save final model
state_dict = model.module.state_dict()
torch.save(state_dict, os.path.join(save_dir, f"{args.model_name}_final.pth"))

# save stats
with open(os.path.join(save_dir, "final_stats.txt"), "w") as f:
    f.write(f"train_loss: {train_loss_all[-1]}, train_accuracy: {train_accuracy_all[-1]:.2f}%\n")
    f.write(f"val_loss: {val_loss_all[-1]}, val_accuracy: {val_accuracy_all[-1]:.2f}%, best_val_acc: {best_acc*100:.2f}% at epoch {best_epoch}\n")
np.save(os.path.join(save_dir, "train_loss_all.npy"), np.array(train_loss_all))
np.save(os.path.join(save_dir, "train_accuracy_all.npy"), np.array(train_accuracy_all))
np.save(os.path.join(save_dir, "val_loss_all.npy"), np.array(val_loss_all))
np.save(os.path.join(save_dir, "val_accuracy_all.npy"), np.array(val_accuracy_all))

text_print='END'
print(text_print), logging.info(text_print)





