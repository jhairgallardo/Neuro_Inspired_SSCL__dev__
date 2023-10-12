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
from torch.utils.data import Dataset, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torchvision.models import *

from PIL import ImageFilter
from tqdm import tqdm
import json

from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser(description='DIET Training')
parser.add_argument('--data', metavar='DIR', default='/data/datasets/CIFAR100/')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--ckpt_file', type=str, default='checkpoint.pth')
parser.add_argument('--save_dir', type=str, default='./output/')
parser.add_argument('--workers', type=int, metavar='N', default=8)
parser.add_argument('--epochs', type=int, metavar='N', default=1000)
parser.add_argument('--warm_epochs', default=10, type=int) 
parser.add_argument('--batch_size', type=int, metavar='N', default=2048)
parser.add_argument('--learning_rate', type=float, metavar='LR', default=0.001)
parser.add_argument('--weight_decay', type=float, metavar='W', default=0.05)
parser.add_argument('--label_smoothing', type=float, metavar='S', default=0.8)              
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--dp', action='store_true', help='use DP training')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--no_shuffle', action='store_true')
parser.add_argument('--noaug', action='store_true')
parser.add_argument('--run_name', default=None)

def main():
    args = parser.parse_args()
    args.shuffle = not args.no_shuffle

    if args.run_name is None:
        args.run_name = 'run1'

    print(vars(args))
    args.save_dir = os.path.join(args.save_dir, 
                                 args.run_name,
                                 f'{args.label_smoothing}ls_' +
                                 f'{"_noaug" if args.noaug else ""}' +
                                 time.strftime("%y%m%d_%H%M%S"))
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
            Transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                std=[0.2673, 0.2564, 0.2762])
        ])
    
    Data_Transform_noaug = Transforms.Compose([
            Transforms.CenterCrop(32),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                std=[0.2673, 0.2564, 0.2762])
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

    # Get validation batch for KNN tracking
    train_dataset_for_knn = Datasets.CIFAR100(root=args.data, train=True, download=True, transform=Data_Transform_noaug)
    knn_data = get_knn_batch(args, train_dataset_for_knn, num_imges=8192)
    del train_dataset_for_knn

    # Train model
    print('\nstarting training...')
    for epoch in tqdm(range(args.epochs)):
        loss = train(args, epoch, train_loader, model, optimizer, criterion, scheduler, knn_data)
        args.writer.add_scalar('train_loss_epoch', loss, epoch)

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

def train(args, epoch, loader, model, optimizer, criterion, scheduler, knn_data=None):
    global iterator
    model.train()

    loss_sum=0
    for x, labels, indexes in loader:
        x = x.cuda(args.gpu, non_blocking=True)
        indexes = indexes.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, indexes)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()

        args.writer.add_scalar('train_loss_batch', loss.item(), iterator)
        args.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iterator)

        iterator+=1

        if knn_data is not None and iterator%10 == 0:
            knn_acc = knn(args, model, knn_data)
            args.writer.add_scalar('knn_acc', knn_acc, iterator)

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
        self.linear_head = nn.Linear(features_dim, num_samples, bias=False)

    def forward(self, x):
        # encoder
        x = self.encoder(x)
        x = self.linear_head(x)
        return x
    
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

def get_knn_batch(args, val_dataset, num_imges):
    if os.path.exists('knn_data_indices.npy'):
        knn_indices = np.load('knn_data_indices.npy')
    else: 
        knn_indices = np.random.randint(len(val_dataset), size=num_imges)
        np.save('knn_data_indices.npy', knn_indices)
    knn_dataset = Subset(val_dataset, knn_indices)
    knn_loader = torch.utils.data.DataLoader(knn_dataset, batch_size=num_imges, shuffle=True)
    knn_x, knn_y = next(iter(knn_loader))
    # split into train and test for knn
    knn_x_train, knn_x_test, knn_y_train, knn_y_test = train_test_split(knn_x, knn_y, 
                                                                        test_size=0.1, 
                                                                        random_state=args.seed,
                                                                        stratify=knn_y.numpy())
    knn_data = [knn_x_train.cuda(args.gpu, non_blocking=True), 
                knn_y_train.numpy(),
                knn_x_test.cuda(args.gpu, non_blocking=True),
                knn_y_test.numpy()]
    del knn_dataset, knn_loader
    return knn_data

def knn(args, model, knn_data, k=20):
    knn_x_train, knn_y_train, knn_x_test, knn_y_test = knn_data

    # get features
    model.eval()
    with torch.no_grad():
        if args.dp:
            features = model.module.encoder(knn_x_train)
            test_features = model.module.encoder(knn_x_test)
        else:
            features = model.encoder(knn_x_train)
            test_features = model.encoder(knn_x_test)
        features = features.detach().cpu().numpy()
        test_features = test_features.detach().cpu().numpy()

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(features,knn_y_train)
    knn_acc = neigh.score(test_features,knn_y_test)

    # return model to train mode
    model.train()

    return knn_acc

if __name__ == '__main__':
    main()