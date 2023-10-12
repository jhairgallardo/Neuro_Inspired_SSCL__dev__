import argparse
import os, time
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as Transforms
import torchvision.datasets as Datasets
from torch.utils.data import Dataset

from torchvision.models import *

from PIL import Image, ImageFilter
from tqdm import tqdm
import json

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='DIET Training')
parser.add_argument('--data', metavar='DIR', default='/data/datasets/CIFAR100/')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--pretrained_folder', type=str, default='./experiments/')
parser.add_argument('--pretrained_network', type=str, default='resnet18_encoder_999ep.pth')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--workers', type=int, metavar='N', default=8)
parser.add_argument('--epochs', type=int, metavar='N', default=100)
parser.add_argument('--batch_size', type=int, metavar='N', default=256)
parser.add_argument('--learning_rate', type=float, metavar='LR', default=0.001) # 0.05
parser.add_argument('--milestones', nargs='+', type=float, default=[0.5, 0.75])
parser.add_argument('--weight_decay', type=float, metavar='W', default=0.01) # 0.005             
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--run_name', default=None)
parser.add_argument('--num_classes',default=100,type=int)

def main():
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = args.pretrained_network.split('.')[0]

    print(vars(args))
    args.save_dir = os.path.join(args.pretrained_folder, 'linear_eval', args.run_name, f'{args.learning_rate}lr_{args.batch_size}bs_{args.weight_decay}wd{args.epochs}ep')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    # Save args as json file
    with open(os.path.join(args.save_dir, args.run_name + '_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    args.writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard_tracking'))

    main_worker(args)

def main_worker(args):
    global iterator
    iterator=0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Define Transforms
    train_transform = Transforms.Compose([
            Transforms.RandomResizedCrop(32),
            Transforms.RandomHorizontalFlip(p=0.5),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])
    val_transform = Transforms.Compose([
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])

    # Load Datasets
    train_dataset = Datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    val_dataset = Datasets.CIFAR100(root=args.data, train=False, download=True, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    ## Load Model
    model = eval(args.arch)(num_classes=args.num_classes)
    # Accomodate for CIFAR100
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    model.maxpool = torch.nn.Identity()
    state_dict = torch.load(os.path.join(args.pretrained_folder, args.pretrained_network))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict['state_dict'], strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    model = model.cuda(args.gpu)

    # Freeze model except linear head
    for _,p in model.named_parameters():
        p.requires_grad = False
    for _,p in model.fc.named_parameters():
        p.requires_grad = True

    # Load optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    milestones = [int(args.epochs * x) for x in args.milestones]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Train model
    print('\nstarting training...')
    for epoch in tqdm(range(args.epochs)):
        train_loss = train(args, epoch, train_loader, model, optimizer, criterion)
        scheduler.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for x, labels in val_loader:
                x = x.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

                output = model(x)
                loss = criterion(output, labels)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100. * correct / total
        
        args.writer.add_scalar('train_loss', train_loss, epoch)
        args.writer.add_scalar('val_loss', val_loss, epoch)
        args.writer.add_scalar('val_acc', val_acc, epoch)

    # save model at the end
    save_dict = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                }
    torch.save(save_dict, os.path.join(args.save_dir, args.arch + '_lineval.pth'))
    args.writer.close()

def train(args, epoch, loader, model, optimizer, criterion):
    global iterator
    model.train()

    loss_sum=0
    for x, labels in loader:
        x = x.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        args.writer.add_scalar('train_loss_log', loss.item(), iterator)
        args.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iterator)

        iterator+=1

    loss_mean = loss_sum/len(loader)

    return loss_mean

        
if __name__ == '__main__':
    main()