import os
import logging
import time
from datetime import timedelta

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision import datasets

from resnet_gn_mish import *
from tqdm import tqdm

import numpy as np
import torch.backends.cudnn as cudnn
import random
import argparse
import json

from function_zca import calculate_ZCA_conv0_weights

from PIL import Image, ImageOps, ImageFilter

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
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dp', action='store_true', default=True)
parser.add_argument('--zca', action='store_true', default=True)
parser.add_argument('--epsilon', type=float, default=1e-5)
parser.add_argument('--aug_type', type=str, default="default", choices=["default", "barlowtwins"])
parser.add_argument('--save_dir', type=str, default="output")
parser.add_argument('--zca_kernel_size', type=int, default=3)
parser.add_argument('--neg_filters', action='store_true', default=False)
parser.add_argument('--zca_num_channels', type=int, default=3)
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

if args.neg_filters and args.zca_num_channels<6:
    raise ValueError("For negative filters, zca_num_channels should be at least 6")

### Create save dir
save_dir = os.path.join(args.save_dir, f"{args.model_name}")

if args.aug_type == "barlowtwins":
    save_dir += "_barlowtwins"

if args.zca: 
    save_dir += f"_zca{args.zca_num_channels}filters_kernel{args.zca_kernel_size}_eps{args.epsilon}"
os.makedirs(save_dir, exist_ok=True)

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
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
if args.zca: std = [1.0, 1.0, 1.0]

if args.aug_type == "default":
    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
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
                        transforms.Normalize(mean=mean, std=std)])
else:
    raise ValueError("Define correct augmentation type")

transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=transform_train)
val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=transform_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                            shuffle=True, num_workers=16, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=16, pin_memory=True)

### Load model
if args.zca:
    conv0_kernel_size = args.zca_kernel_size
    conv0_outchannels = args.zca_num_channels
else:
    conv0_kernel_size = None
    conv0_outchannels = 3

if 'resnet' in args.model_name:
    model = eval(args.model_name)(num_classes=args.num_classes, 
                                zero_init_residual=args.zero_init_residual, 
                                conv0_flag= args.zca, 
                                conv0_outchannels=conv0_outchannels,
                                conv0_kernel_size=conv0_kernel_size)
elif 'vgg' in args.model_name:
    model = eval(args.model_name)(num_classes=args.num_classes, 
                                conv0_flag= args.zca, 
                                conv0_outchannels=conv0_outchannels,
                                conv0_kernel_size=conv0_kernel_size)
else:
    raise ValueError("Model not found")

### Calculate filters and load it to the model
if args.zca:
    zca_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
    zca_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=zca_transform)
    weight = calculate_ZCA_conv0_weights(model = model, dataset = zca_dataset,
                                        nimg = 10000, zca_epsilon=args.epsilon,
                                        save_dir = save_dir,
                                        neg_filters = args.neg_filters)
    model.conv0.weight = torch.nn.Parameter(weight)
    model.conv0.weight.requires_grad = False

### Add model to GPU
if args.dp: model = torch.nn.DataParallel(model)
model.to("cuda")

### Load optimizer and criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr/1000, total_iters=args.warmup_epochs)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

### Save init model
if args.dp: state_dict = model.module.state_dict()
else: state_dict = model.state_dict()
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
        if args.dp: state_dict = model.module.state_dict()
        else: state_dict = model.state_dict()
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
if args.dp: state_dict = model.module.state_dict()
else: state_dict = model.state_dict()
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





