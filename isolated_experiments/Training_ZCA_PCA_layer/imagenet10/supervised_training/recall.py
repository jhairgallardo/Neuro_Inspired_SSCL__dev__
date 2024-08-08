import os
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


class ImageNetDataset(Dataset):

    def __init__(self, image_dir, bbox_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.label_file = label_file
        self.image_files = []
        self.annotations = []
        self.label_to_index = self._load_label_mapping(label_file)
        self._gather_files(image_dir, bbox_dir)
        self.transform = transform

    def _load_label_mapping(self, label_file):
        label_to_index = {}
        with open(label_file, 'r') as f:
            for idx, line in enumerate(f):
                label = line.strip()
                label_to_index[label] = idx
        return label_to_index

    def _gather_files(self, image_dir, bbox_dir):
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.JPEG'):
                    image_path = os.path.join(root, file)
                    image_id = os.path.splitext(file)[0]
                    image_class = image_id.split('_')[0]
                    xml_path = os.path.join(bbox_dir, image_class)
                    xml_file = os.path.join(xml_path, f'{image_id}.xml')
                    if os.path.exists(xml_file):
                        self.image_files.append(image_path)
                        self.annotations.append(xml_file)
                    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        xml_file = self.annotations[idx]
        
        image_class = os.path.basename(image_path).split('_')[0]
        label_index = self.label_to_index.get(image_class, -1)  # -1 if not found

        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        if self.transform:
            image = self.transform(image)
        boxes = self.parse_bounding_box(xml_file, original_width, original_height, image.shape[2], image.shape[1])

        return image, torch.tensor(boxes), label_index

    def parse_bounding_box(self, xml_file, original_width, original_height, new_width, new_height):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Rescale the coordinates
            xmin = xmin * new_width // original_width
            ymin = ymin * new_height // original_height
            xmax = xmax * new_width // original_width
            ymax = ymax * new_height // original_height

            boxes.append([xmin, ymin, xmax, ymax])
        return boxes

def custom_collate_fn(batch):
    images, boxes, labels = zip(*batch)

    images = torch.stack(images, dim=0)
    max_num_boxes = max(len(box) for box in boxes)
    
    padded_boxes = torch.zeros((len(boxes), max_num_boxes, 4))
    for i, box in enumerate(boxes):
        padded_boxes[i, :len(box), :] = box
    
    labels = torch.tensor(labels, dtype=torch.int64)

    return images, padded_boxes, labels

# # Define the transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # Initialize the dataset and dataloader
# image_dir = '/data/datasets/ImageNet-100/train/'
# bbox_dir = '/home/helia/active_vision/ImageNet-100/bbox_train/'
# label_file = '/data/datasets/ImageNet-100/IN100.txt'

# dataset = ImageNetDataset(image_dir, bbox_dir, label_file, transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

# def draw_bounding_boxes_on_batch(images, boxes_batch, image_ids):
#     batch_size, channels, height, width = images.shape
#     images = images.permute(0, 2, 3, 1).numpy()  # Change to (batch_size, height, width, channels)

#     plt.figure(figsize=(15, 10))
    
#     for i in range(batch_size):
#         image = images[i]
#         boxes = boxes_batch[i]

#         # Convert image from tensor to numpy
#         image = (image * 255).astype(np.uint8)
#         plt.subplot(1, batch_size, i+1)
#         plt.imshow(image)
#         for (xmin, ymin, xmax, ymax) in boxes:
#             plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=2, alpha=0.5))
#         plt.title(f'Label: {image_ids[i]}')
#         plt.axis('off')
    
#     # plt.show()
#     plt.savefig('groundtruth.jpg',dpi=300,bbox_inches='tight')
#     plt.close()

# # Get a batch of images from the dataloader
# data_iter = iter(dataloader)
# try:
#     images, boxes_batch, image_ids = next(data_iter)
#     print("img: ", images.shape)
#     print("bbox: ", boxes_batch)
#     print("label: ", image_ids.shape)
#     # Visualize the bounding boxes
#     # draw_bounding_boxes_on_batch(images, boxes_batch, image_ids)
# except StopIteration:
#     print("DataLoader is empty, check your dataset and batch size.")


#################################################################
# Customized augmentations (crop)

import numpy as np
import torch
from scipy.stats import gennorm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count


import os,sys,math
import logging
import time
from datetime import timedelta

import torch.nn as nn

from torchvision import transforms
from torchvision import datasets

from resnet_gn_mish import *
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random
import argparse
import json

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import cv2
from PIL import Image

# from torch.utils.tensorboard import SummaryWriter
from function_zca import calculate_ZCA_conv0_weights, scaled_filters



def mean_pooling(feats, kernel_size=16, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    pooled_feats = F.avg_pool2d(feats, kernel_size, stride=stride, padding=padding)
    return pooled_feats
    
def l2_pooling(feats, kernel_size=16, stride=1, padding=0):
    # feats: tensor of shape (batch_size, num_filters, height, width)
    squared_feats = feats ** 2
    pooled_feats = F.avg_pool2d(squared_feats, kernel_size, stride=stride, padding=padding)  #, divisor_override=1)
    return torch.sqrt(pooled_feats)

def fit_gennorm(feature_map):
    return gennorm.fit(feature_map)

def get_gennorm_params(feats):
    batch_size, num_filters, height, width = feats.shape
    
    # Flatten the feature maps for all filters in the batch
    flattened_feats = feats.view(batch_size * height * width, num_filters).cpu().numpy()
    # print(flattened_feats.shape)   # (..., n_filters)

    with Pool(processes=cpu_count()) as pool:
        ggd_params = pool.map(fit_gennorm, [flattened_feats[:, j] for j in range(num_filters)])

    return ggd_params

def gennorm_pdf(x, theta, loc, sigma):
    # https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1
    # https://github.com/scipy/scipy/blob/87c46641a8b3b5b47b81de44c07b840468f7ebe7/scipy/stats/_continuous_distns.py#L11153
    theta = torch.tensor(theta)
    loc = torch.tensor(loc)
    sigma = torch.tensor(sigma)
    return torch.exp( torch.log(0.5*theta) - torch.log(sigma) - torch.lgamma(1.0/theta) - (torch.abs(x-loc)/sigma)**theta )

def compute_saliency_map_ggd_batch(features, ggd_params, weighted=False):
    features = features.to(torch.float64)
    batch_size, num_filters, height, width = features.shape

    # Compute the joint probability of all filters for each image in the batch
    saliency_maps = torch.ones((batch_size, height, width), dtype=torch.float64).to(features.device)
    for i in range(num_filters):
        theta, loc, sigma = ggd_params[i]
        if weighted:
            theta_inv = 1.0 / theta
            theta_inv = torch.tensor(theta_inv).to(features.device)
            gamma_func = torch.exp(torch.lgamma(theta_inv))
            gamma_incomplete = torch.special.gammainc(theta_inv, (torch.abs(features[:, i, :, :]) ** theta) * (sigma ** -theta))
            improved_feats = gamma_incomplete / gamma_func
            p = gennorm_pdf(improved_feats.flatten(), theta, loc, sigma)
        else:
            p = gennorm_pdf(features[:, i, :, :].flatten(), theta, loc, sigma)
        p = p.reshape(batch_size, height, width)
        saliency_maps *= p

    unique_values = torch.unique(saliency_maps.flatten())
    second_min = torch.topk(unique_values, 2, largest=False)[0][1].item()
    print('2nd min: ', second_min)
    saliency_maps = 1 / (saliency_maps + second_min)
    # saliency_maps = 1 / (saliency_maps + 1e-18)
    # saliency_maps /= saliency_maps.sum(dim=(1, 2), keepdim=True)
    saliency_maps = saliency_maps.to(torch.float32)
    return saliency_maps

def resize_saliency_map(saliency_maps, original_shape=(224,224)):
    # saliency_maps: tensor of shape (batch_size, height, width)
    saliency_maps_tensor = saliency_maps.unsqueeze(1)  # Add channel dimension
    resized_saliency_maps = F.interpolate(saliency_maps_tensor, size=original_shape, mode='bilinear', align_corners=True)
    resized_saliency_maps = resized_saliency_maps.squeeze(1)  # Remove channel dimension

    # # Normalize each saliency map in the batch
    # batch_sums = resized_saliency_maps.view(resized_saliency_maps.shape[0], -1).sum(dim=1)  # Compute sum of each saliency map
    # resized_saliency_maps /= batch_sums.view(-1, 1, 1)  # Normalize
    
    return resized_saliency_maps


class ElapsedFilter(logging.Filter):
    def __init__(self):
        self.start_time = time.time()
    
    def filter(self, record):
   
        elapsed_seconds = record.created - self.start_time
        #using timedelta here for convenient default formatting
        elapsed = timedelta(seconds = elapsed_seconds)
        record.elapsed=elapsed
        return True


def generate_guided_crop(features, images, gt_boxes, ggd_params, mean, std, weighted=False, output_size=224,
                        scale = [0.081, 0.081], ratio = [1.0/1.0, 1.0/1.0], num_crops=1, pooling = None,
                        save_dir=None):
    feats = features.detach()
    imgs = images.detach()
    gt_bboxes = gt_boxes.detach().cpu()
    batch_size, num_filters, height, width = feats.shape

    feats = feats.abs()
    # feats = feats.abs().mean(dim=1)

    if pooling == "l2": 
        feats = l2_pooling(feats)
    elif pooling == "avg":
        feats = mean_pooling(feats)

    # Compute saliency maps (turn on/off weighted)
    saliency_maps = compute_saliency_map_ggd_batch(feats, ggd_params, weighted=weighted)
    # saliency_maps = feats

    if (pooling == "l2") or (pooling == "avg"):
        saliency_maps = resize_saliency_map(saliency_maps)

    # Crop
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio))

    # Normalize saliency maps
    # print(f"smap idx 0 b4 normalizing:", saliency_maps[0])
    saliency_maps /= saliency_maps.sum(dim=(1, 2), keepdim=True)
    # print(f"smap idx 0 after normalizing (/sum):", saliency_maps[0])

    # all_crops = torch.zeros(batch_size*num_crops, 3, output_size, output_size, dtype=torch.int32)
    all_crops = torch.zeros(batch_size, num_crops, 4, dtype=torch.int32)

    for b in range(batch_size):
        saliency_map = saliency_maps[b]
        
        for k in range(num_crops):
            # Get w_crop and h_crop (size of crop)
            for i in range(10):
                target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
                aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
                h_crop = int(round(math.sqrt(target_area / aspect_ratio)))
                w_crop = int(round(math.sqrt(target_area * aspect_ratio)))
                
                if 0 < h_crop <= height and 0 < w_crop <= width:
                    break
                elif i == 9: # if it fails 10 times, then just take the whole image
                    h_crop = height
                    w_crop = width

            if h_crop == height and w_crop == width: # if the whole image is the crop, save time by not sampling position
                all_crops[b,k] = torch.tensor([0, 0, h_crop, w_crop])
                # crop = TF.resized_crop(imgs[b], 0, 0, h_crop, w_crop, output_size)
                # all_crops[b] = crop
                continue

            # Get idx_x and idx_y (top left corner of crop). Use saliency map as probability distribution
            for i in range(10):
                # idx = np.random.choice(np.arange(len(saliency_map.flatten())), p=saliency_map.flatten())
                probabilities = saliency_map.flatten()
                print(probabilities)
                idx = probabilities.multinomial(1).item()
                idx_cy, idx_cx = np.unravel_index(idx, saliency_map.shape) # center of crop
                
                # sanity check line: get position with highest saliency
                # idx_cy, idx_cx = np.unravel_index(np.argmax(saliency_map.flatten()), saliency_map.shape) 
                
                # if part of the crop falls outside the image, then move the center of the crop.
                # It makes sure that the sampled center is within the crop (not necesarily the center, but inside the crop)
                if idx_cy + h_crop // 2 >= height:
                    diff_cy = idx_cy + h_crop // 2 - height + (h_crop%2)
                    idx_cy -= diff_cy
                if idx_cy - h_crop // 2 <= 0:
                    diff_cy = h_crop // 2 - idx_cy
                    idx_cy += diff_cy
                if idx_cx + w_crop // 2 >= width:
                    diff_cx = idx_cx + w_crop // 2 - width + (w_crop%2)
                    idx_cx -= diff_cx
                if idx_cx - w_crop // 2 <= 0:
                    diff_cx = w_crop // 2 - idx_cx
                    idx_cx += diff_cx
                
                idx_y = idx_cy - (h_crop // 2)
                idx_x = idx_cx - (w_crop // 2)


                # make sure the complete crop is within the image (safety check)
                if 0 <= idx_x and idx_x + w_crop <= width and 0 <= idx_y and idx_y + h_crop <= height:
                    break
                elif i == 9: # if it fails 10 times, then just take the center of the image
                    idx_cx = width // 2
                    idx_cy = height // 2
                    idx_x = idx_cx - w_crop // 2
                    idx_y = idx_cy - h_crop // 2
        
            all_crops[b,k] = torch.tensor([idx_x, idx_y, idx_x+h_crop, idx_y+w_crop])
            # Perform the crop and resize operation on the imgs
            # crop = TF.resized_crop(imgs[b], idx_x, idx_y, h_crop, w_crop, output_size)
            # all_crops[b] = crop
    # Visualize
    fig, axs = plt.subplots(5, 8, figsize=(24, 15))
    for i, b in enumerate(range(0, batch_size, 25)):
        if i >= 20:  # Limit to 20 images
            break
        row = (i // 4)
        col = (i % 4) * 2
        # Unnormalize the image
        unnorm_image = imgs[b].cpu() * (torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
        unnorm_image = unnorm_image.squeeze().numpy()
        unnorm_image = np.moveaxis(unnorm_image, 0, -1)
        # Ground truth subplot
        axs[row, col].imshow(unnorm_image)
        for box in gt_bboxes[b]:
            axs[row, col].add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='red', linewidth=2, alpha=0.7))
        axs[row, col].set_title('Ground truth')
        axs[row, col].axis('off')
        # Guided BBoxes subplot
        axs[row, col+1].imshow(unnorm_image.mean(2), cmap='gray')
        axs[row, col+1].imshow(saliency_maps[b].cpu(), cmap='jet', alpha=0.5, interpolation='nearest')
        for gbox in all_crops[b]:
            axs[row, col+1].add_patch(plt.Rectangle((gbox[0], gbox[1]), gbox[2]-gbox[0], gbox[3]-gbox[1], fill=False, edgecolor='red', linewidth=2, alpha=0.7))
        axs[row, col+1].set_title('Guided BBoxes')
        axs[row, col+1].axis('off')
    plt.tight_layout()
    x = random.randint(0,100)
    plt.savefig(os.path.join(save_dir, f"conv0_act0_2ndmin_pksize3_bboxes_{x}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return all_crops.to(features.device)


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two sets of boxes.
    box1: [batch_size, num_boxes, 4]
    box2: [batch_size, num_boxes, 4]
    """
    # print("b1: ", box1.shape)
    # print("b2: ", box2.shape)
    
    x1 = torch.max(box1[:, :, None, 0], box2[:, None, :, 0])
    y1 = torch.max(box1[:, :, None, 1], box2[:, None, :, 1])
    x2 = torch.min(box1[:, :, None, 2], box2[:, None, :, 2])
    y2 = torch.min(box1[:, :, None, 3], box2[:, None, :, 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1])
    box2_area = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1])
    
    iou = inter_area / (box1_area[:, :, None] + box2_area[:, None, :] - inter_area)
    return iou


def compute_recall(pred_boxes, gt_boxes, threshold=0.05):
    """
    Compute the recall for a batch of predicted boxes against (potentially multiple) ground truth boxes.
    pred_boxes: [batch_size, 1, 4]
    gt_boxes: [batch_size, num_gt_boxes, 4]
    """
    # print("pred: ", pred_boxes.shape)
    # print("gt: ", gt_boxes.shape)

    batch_size = pred_boxes.size(0)
    recalls = []
    avg_recalls = []
    
    for i in range(batch_size):
        # # Repeat the predicted box to match the number of ground truth boxes
        # if pred_boxes.shape != gt_boxes.shape:
        #     pred_box = pred_boxes[i].repeat(gt_boxes[i].size(0), 1)
        # else:
        #     pred_box = pred_boxes[i]
        ious = compute_iou(pred_boxes[i].unsqueeze(0), gt_boxes[i].unsqueeze(0))

        # Recall for 1 pred_box with highest IOU per image
        max_iou = ious.max().item()
        recall = 1.0 if max_iou >= threshold else 0.0
        recalls.append(recall)

        # Avg Recall for all pred_boxes per image using one highest IOU per pred_box
        max_iou, _ = ious.max(dim=-1)
        recall = (max_iou >= threshold).float().mean().item()
        avg_recalls.append(recall)
    
    return torch.tensor(recalls).mean().item(), torch.tensor(avg_recalls).mean().item()

def main(args):
    print(f"Use ZCA filters: {args.zca}")
    print(f"Use guided crop: {args.guided_crop}")
    print(f"Use weighted ggd: {args.weighted_ggd}")

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
    save_dir = os.path.join(args.save_dir, f"{args.model_name}_recall")
    if args.zca: save_dir += f"_zca6_kernel{args.zca_kernel_size}_eps{args.epsilon}"
    # if args.zca: save_dir += f"_zca3f_kernel{args.zca_kernel_size}_eps{args.epsilon}"

    if args.guided_crop: 
        save_dir += "_guidedcrop"
    else: 
        save_dir += "_randomcrop"

    if args.scale == 0.005:
        save_dir += "16"
    elif args.scale == 0.02:
        save_dir += "32"
    elif args.scale == 0.081:
        save_dir += "64"
    elif args.scale == 0.324:
        save_dir += "128"

    if args.pooling == "l2":
        save_dir += "_l2pool"
    elif args.pooling == "avg":
        save_dir += "_avgpool"
    else:
        save_dir += "_nopool"

    if args.weighted_ggd: 
        save_dir += "_weightedggd"

    if args.zca_act_out == 'hardtanh':
        save_dir += "_hardtanh"
    elif args.zca_act_out == 'tanh':
        save_dir += "_tanh"
    elif args.zca_act_out == 'no_act':
        save_dir += "_noact"
    elif args.zca_act_out == 'mish':
        save_dir += "_mish"
    elif args.zca_act_out == 'mishtanh':
        save_dir += "_mishtanh"

    if args.zca_scale_filter:
        save_dir += "_scaledzca"
    
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
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        # transforms.RandomResizedCrop(size=224, scale=(args.scale, args.scale), ratio=(args.ratio, args.ratio)),
                        # transforms.RandomResizedCrop(224),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
    
    transform_val = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    # transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
    # train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=transform_train)
    bbox_dir = '/home/helia/active_vision/ImageNet-10/bbox_train/'
    train_dataset = ImageNetDataset(os.path.join(args.data_path, "train"), bbox_dir, os.path.join(args.data_path, "IN10.txt"), transform=transform_train)
    # val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=transform_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                shuffle=False, num_workers=16, pin_memory=True, collate_fn=custom_collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
    #                                             shuffle=False, num_workers=16, pin_memory=True)

    ### Load model
    if args.zca_act_out == 'mish':
        act0 = nn.Mish()
    elif args.zca_act_out == 'hardtanh':
        act0 = nn.Hardtanh(min_val=args.min_hardtanh, max_val=args.max_hardtanh)
    elif args.zca_act_out == 'tanh':
        act0 = nn.Tanh()
    elif args.zca_act_out == 'no_act':
        act0 = nn.Identity()
    elif args.zca_act_out == 'mishtanh':
        act0 = nn.Sequential(nn.Mish(), nn.Tanh())
    else:
        raise ValueError('ZCA Activation function not recognized')
    
    if args.zca:
        conv0_kernel_size = args.zca_kernel_size
        conv0_outchannels = 6
    elif args.pca: conv0_outchannels = 54
    else: conv0_outchannels = 3
    model = eval(args.model_name)(num_classes=args.num_classes, zero_init_residual=args.zero_init_residual, conv0_flag= args.zca or args.pca, 
                                  conv0_outchannels=conv0_outchannels, conv0_kernel_size=conv0_kernel_size, act0=act0)

    ### Calculate filters and load it to the model
    if args.zca:
        zca_transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
        zca_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=zca_transform)
        zca_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=10000, shuffle=True)
        zca_input_imgs,_ = next(iter(zca_loader))
        mean_of_inputs = zca_input_imgs.mean(dim=(0,2,3)).tolist()
        with open(f'{save_dir}/mean_imgs_input_for_zca.json', 'w') as f: json.dump(mean_of_inputs, f)
        weight = calculate_ZCA_conv0_weights(model=model, imgs=zca_input_imgs, zca_epsilon=args.epsilon)
        model.conv0.weight = torch.nn.Parameter(weight)
        model.conv0.weight.requires_grad = False

        if args.zca_scale_filter:
            if args.zca_scale_filter_mode == 'all':
                weight = scaled_filters(model.conv0, imgs=zca_input_imgs)
            elif args.zca_scale_filter_mode == 'per_channel':
                weight = scaled_filters(model.conv0, imgs=zca_input_imgs, per_channel=True)
            model.conv0.weight = torch.nn.Parameter(weight)
            model.conv0.weight.requires_grad = False

    ### Get GGD params for ZCA filter responses on zca dataset
    ggd_params = None
    if args.zca and args.guided_crop:
        
        # Path to save/load the GGD parameters
        if args.weighted_ggd: 
            ggd_params_path = os.path.join(save_dir.replace("_weightedggd", ""), 'ggd_params.npy')
        else:
            ggd_params_path = os.path.join(save_dir, 'ggd_params.npy')
        
        # Check if the file exists
        if os.path.exists(ggd_params_path):
            print('Loading GGD params from file...')
            ggd_params = np.load(ggd_params_path).tolist()
            print('Loaded GGD params:', ggd_params)
        else:
            print('Fitting ggd params...')
            start_time = time.time()
            # get images
            data_loader = torch.utils.data.DataLoader(zca_dataset, batch_size=10000, shuffle=True, num_workers=16)
            imgs,_ = next(iter(data_loader))
            model.eval()
            batch_feats = model.conv0(imgs)
            batch_feats = model.act0(batch_feats)
            batch_feats = batch_feats.squeeze()
            if args.pooling == "l2":
                # Apply L2 pooling
                batch_l2_feats = l2_pooling(batch_feats.abs().detach())
                ggd_params = get_gennorm_params(batch_l2_feats.cpu())
            elif args.pooling == "avg":
                # Apply avg pooling
                batch_avg_feats = mean_pooling(batch_feats.abs().detach())
                ggd_params = get_gennorm_params(batch_avg_feats.cpu())
            else:
                ggd_params = get_gennorm_params(batch_feats.abs().detach().cpu())
            end_time = time.time()
            print(f"Time taken to get GGD params: {end_time - start_time} seconds")
            print('Got params: ', ggd_params)
            # Save the GGD params
            np.save(ggd_params_path, ggd_params)
            print(f'Saved GGD params to {ggd_params_path}')
    
    del zca_input_imgs, zca_dataset, zca_loader
    
    ### Add model to GPU
    if args.dp: model = torch.nn.DataParallel(model)
    model.to("cuda")
    
    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr/1000, total_iters=args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
        
    ### Train model
    for epoch in range(args.epochs):

        ## Train step ##
        # model.train()
        model.eval()
        total_onemax_recall = 0.0
        total_avgmax_recall = 0.0
        num_batches = len(train_loader)
        print("Number of batches: ", num_batches)
        with torch.no_grad():
            for data, gt_boxes, target in train_loader:
        
                data, gt_boxes, target = data.to("cuda"), gt_boxes.to("cuda"), target.to("cuda")
                # gt_boxes = torch.tensor([box.to("cuda") for box in gt_boxes])

                # Perform guided crop
                
                if args.zca and args.guided_crop:
                    if args.dp:
                        feats = model.module.conv0(data)
                        feats = model.module.act0(feats)
                    else:
                        feats = model.conv0(data)
                        feats = model.act0(feats)
                    
                    pred_boxes = generate_guided_crop(feats, data, gt_boxes, ggd_params, mean, std, weighted=args.weighted_ggd, scale = [args.scale, args.scale], ratio = [args.ratio, args.ratio], num_crops=args.num_crops, pooling=args.pooling, save_dir=save_dir)
                    # print('predicted boxes shape: ', pred_boxes.shape)
                    # print('gt box shape: ', gt_boxes.shape)
                    
                    onemax_recall, avgmax_recall = compute_recall(pred_boxes, gt_boxes, threshold=args.threshold)
                    print('Recall for one best pred_box per image (averaged across batch): ', onemax_recall)
                    print("Average Recall for all pred_boxes using one highest IOU for each pred_box (averaged across batch): ", avgmax_recall)
                    total_onemax_recall += onemax_recall
                    total_avgmax_recall += avgmax_recall
        final_onemax_recall = total_onemax_recall / num_batches
        final_avgmax_recall = total_avgmax_recall / num_batches

        text_print = (f'Epoch {epoch} -- lr: {scheduler.get_last_lr()[0]:.6f} -- '
                      f'one max recall: {final_onemax_recall:.5f} -- '
                        f'avg max recall: {final_avgmax_recall:.5f}, ')
                
        print(text_print), logging.info(text_print)

    # save stats
    with open(os.path.join(save_dir, "final_recalls_2ndmin.txt"), "a") as f:
        f.write(f"Threshold: {args.threshold:.2f}\n")
        f.write(f"onemax_recall: {final_onemax_recall:.5f}\n")
        f.write(f"avgmax_recall: {final_avgmax_recall:.5f}\n")
        
    text_print='END'
    print(text_print), logging.info(text_print)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_path', type=str, default="/data/datasets/ImageNet-10")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--model_name', type=str, default="resnet18")
    parser.add_argument('--zero_init_residual', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dp', action='store_true', default=True)
    parser.add_argument('--zca', action='store_true', default=False)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--epsilon', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default="output")
    parser.add_argument('--guided_crop', action='store_true', default=False)
    parser.add_argument('--weighted_ggd', action='store_true', default=False)
    parser.add_argument('--normalized_zca', action='store_true', default=False)
    parser.add_argument('--zca_kernel_size', type=int, default=3)
    parser.add_argument('--pooling', type=str, default="no", choices=["no", "l2", "avg"])
    parser.add_argument('--scale', type=float, default=0.005)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--num_crops', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--zca_act_out', type=str, default='mishtanh', choices=['no_act', 'mish', 'hardtanh', 'tanh','mishtanh'])
    parser.add_argument('--zca_scale_filter', action='store_true', default=True)
    parser.add_argument('--zca_scale_filter_mode', type=str, default='per_channel', choices=['all', 'per_channel'])
    args = parser.parse_args()


    main(args)
