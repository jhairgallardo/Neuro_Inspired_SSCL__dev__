import argparse
import os, time
import random
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as Transforms
import torchvision.datasets as Datasets
from torch.utils.data import Dataset, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torchvision.models import *

from PIL import Image, ImageFilter
from tqdm import tqdm
import json

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DIET Training')
parser.add_argument('--data', metavar='DIR', default='/data/datasets/CIFAR100/')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--ckpt_file', type=str, default='checkpoint.pth')
parser.add_argument('--save_dir', type=str, default='./experiments/')
parser.add_argument('--workers', type=int, metavar='N', default=8)
parser.add_argument('--epochs', type=int, metavar='N', default=1000) # 1000
parser.add_argument('--warm_epochs', default=10, type=int) # 10
parser.add_argument('--batch_size', type=int, metavar='N', default=2048) #256
parser.add_argument('--learning_rate', type=float, metavar='LR', default=0.001)
parser.add_argument('--weight_decay', type=float, metavar='W', default=0.05)
parser.add_argument('--label_smoothing', type=float, metavar='S', default=0.8)              
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--dp', action='store_true', help='use DP training')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--no_shuffle', action='store_true')
parser.add_argument('--cosine_softmax', action='store_true')
parser.add_argument('--noaug', action='store_true')
parser.add_argument('--soft_targets', action='store_true')
parser.add_argument('--run_name', default=None)

### TODO: implement KNN tracker evaluation

def main():
    args = parser.parse_args()
    # # args.soft_targets = True #######################################################################
    # args.dp = True #################################################################################
    args.shuffle = not args.no_shuffle

    if args.run_name is None:
        args.run_name = 'run1'

    if args.soft_targets:
        print('Soft targets activated! Label smoothing is disabled.')
        args.label_smoothing = 0

    print(vars(args))
    args.save_dir = os.path.join(args.save_dir, 
                                 args.run_name, 
                                 time.strftime("%y%m%d_%H%M%S") + f'_batch_{args.batch_size}' + f'{"_noaug" if args.noaug else ""}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
    # Save args as json file
    with open(os.path.join(args.save_dir, args.run_name + '_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    args.writer = SummaryWriter(os.path.join(args.save_dir,'tensorboard_tracking'))

    main_worker(args)

def main_worker(args):
    global iterator
    iterator=0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Define Transforms
    Data_Transform = Transforms.Compose([
            Transforms.RandomResizedCrop(32),
            Transforms.RandomHorizontalFlip(p=0.5),
            Transforms.RandomApply(
                [Transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4, hue=0.2)],
                p=0.3
            ),
            Transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.2),
            Transforms.ToTensor(),
            Transforms.RandomErasing(p=0.25),
            Transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])
    
    Data_Transform_noaug = Transforms.Compose([
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])

    # Load Dataset
    transform_funct = Data_Transform_noaug if args.noaug else Data_Transform
    train_dataset = Datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_funct)

    if args.no_shuffle: # order train data by class number
        print("Non-IID data")
        train_labels = train_dataset.targets
        sorted_indices = np.argsort(train_labels)
        train_dataset = Subset(train_dataset,sorted_indices)

    train_dataset = Datasetwithindex(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.workers, pin_memory=True)
    
    # Load Model
    model = DIET(args, num_samples=len(train_dataset))
    if args.dp:
        model = nn.DataParallel(model)
    model = model.cuda(args.gpu)
                
    # Load optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate*args.batch_size/256, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    warmup_scheduler = LinearLR(optimizer = optimizer, 
                                start_factor = 1./3., # start at 1/3 * lr
                                total_iters = len(train_loader) * args.warm_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer = optimizer,
                                         T_max = len(train_loader)*(args.epochs - args.warm_epochs),
                                         eta_min=args.learning_rate*0.001) # end at 1e-3*lr
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[len(train_loader)*args.warm_epochs])

    # Train model
    print('\nstarting training...')
    for epoch in tqdm(range(args.epochs)):
        loss = train(args, epoch, train_loader, model, optimizer, criterion, scheduler)
        print(f'Epoch {epoch} loss: {loss}')

        if (epoch+1) % 100 == 0:
            save_dict = {
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.module.encoder.state_dict() if args.dp else model.encoder.state_dict(),
                    }
            torch.save(save_dict, os.path.join(args.save_dir, args.arch + f'_encoder_{epoch}ep.pth'))

    # save encoder at the end
    save_dict = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.module.encoder.state_dict() if args.dp else model.encoder.state_dict(),
                }
    torch.save(save_dict, os.path.join(args.save_dir, args.arch + '_encoder.pth'))

    args.writer.close()

def train(args, epoch, loader, model, optimizer, criterion, scheduler):
    global iterator
    model.train()

    loss_sum=0
    for x, labels, indexes in loader:
        x = x.cuda(args.gpu, non_blocking=True)
        indexes = indexes.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        output = model(x)
        
        if args.soft_targets: # replace indexes with soft targets
            batch_idx = torch.arange(len(indexes)).cuda(args.gpu)
            soft_targets = F.softmax(output.detach(), dim=1)
            # plt.plot(soft_targets[0].cpu().numpy(), '.')
            # plt.savefig('img_before.jpg')
            # plt.close()
            max_values, _ = soft_targets.max(dim=1) # get max value and index
            soft_targets[batch_idx,indexes] = 10*max_values # replace correct class prob with 1.5 max value
            
            soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True) # normalize so each row sums to 1 (no softmax)
            indexes = soft_targets # assing new targets

            # plt.plot(indexes[0].cpu().numpy(), '.')
            # plt.savefig('img.jpg')
            # plt.close()

        # if args.soft_targets: # my label smoothing (it works!)
        #     batch_idx = torch.arange(len(indexes)).cuda(args.gpu)
        #     soft_targets = F.softmax(output.detach(), dim=1)
        #     plt.plot(soft_targets[0].cpu().numpy(), '.')
        #     plt.savefig('img_before.jpg')
        #     plt.close()
        #     soft_targets[:,:] = 0.8/50000
        #     soft_targets[batch_idx,indexes] = (1-0.8) + soft_targets[batch_idx,indexes]
        #     indexes = soft_targets # assing new targets
        #     plt.plot(soft_targets[0].cpu().numpy(), '.')
        #     plt.savefig('img.jpg')
        #     plt.close()

        loss = criterion(output, indexes)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()

        args.writer.add_scalar('train_loss', loss.item(), iterator)
        args.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iterator)

        iterator+=1

    loss_mean = loss_sum/len(loader)

    return loss_mean


class DIET(nn.Module):
    def __init__(self, args, num_samples):
        super().__init__()
        self.args = args
        self.encoder = eval(args.arch)(zero_init_residual=True)
        # Accomodate for CIFAR100
        self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.encoder.maxpool = torch.nn.Identity()
        # get feature dimension and remove final fully connected layer
        features_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        # Linear head
        if args.cosine_softmax:
            print("Using cosine softmax")
            self.linear_head = CosineLinear(features_dim, num_samples, bias=False)
        else:
            self.linear_head = nn.Linear(features_dim, num_samples, bias=False)

    def forward(self, x, indexes=None):
        # encoder
        x = self.encoder(x)
        x = self.linear_head(x)
        return x

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sigma=True): # sigma is 1/temp
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features)) # C x d i.e., 1000 x 1280
        if sigma: self.sigma = Parameter(torch.Tensor(1))
        else: self.register_parameter('sigma', None)
        if bias: self.bias = Parameter(torch.Tensor(out_features, 1))
        else: self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 0.01)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):       
        if self.bias is not None:
            input = torch.cat((input, (torch.ones(len(input),1).cuda())), dim=1)
            concat_weight = torch.cat((self.weight, self.bias), dim=1)
            out = F.linear(F.normalize(input,p=2,dim=1,eps=1e-8), 
                           F.normalize(concat_weight,p=2,dim=1,eps=1e-8))
        else:
            out = F.linear(F.normalize(input,p=2,dim=1,eps=1e-8), 
                           F.normalize(self.weight,p=2,dim=1,eps=1e-8))
            ## N:B: eps 1e-8 is better than default 1e-12
        if self.sigma is not None:
            out = self.sigma * out
        return out
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Datasetwithindex(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y, index

if __name__ == '__main__':
    main()