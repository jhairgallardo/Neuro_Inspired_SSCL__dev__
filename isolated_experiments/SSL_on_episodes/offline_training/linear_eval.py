import argparse
import os, time
import random
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets

from resnet_gn_mish import *
import numpy as np

parser = argparse.ArgumentParser(description='Linear evaluation on ImageNet-100')
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-100')
parser.add_argument('--model_name', type=str, default='resnet18')
parser.add_argument('--pretrained_model', type=str, default=None)
parser.add_argument('--zero_init_res', action='store_true', default=True)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dp', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./output/lineval/')
parser.add_argument('--seed', type=int, default=0)

def main():
    args = parser.parse_args()

    # Seed Everything
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Define folder to save results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # print and save args
    print(args)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main_worker(args, device)


def main_worker(args, device):


    print('\n==> Preparing data...')
    ### Load data
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    train_dataset = datasets.ImageFolder(traindir, transform=transform_train)
    val_dataset = datasets.ImageFolder(valdir, transform=transform_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                            shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers, pin_memory=True)
    

    print('\n==> Building and loading model')
    ### Load model
    model = eval(args.model_name)(num_classes=args.num_classes, zero_init_residual=args.zero_init_res)
    if args.pretrained_model is not None:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.pretrained_model), strict=False)
        assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
        model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes).cuda()
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        # Freeze model except linear head
        for _,p in model.named_parameters():
            p.requires_grad = False
        for _,p in model.fc.named_parameters():
            p.requires_grad = True
    else: # error, no pretrained model
        raise ValueError('No pretrained model provided')
    # send to gpu
    if args.dp:
        model = torch.nn.DataParallel(model)
    model = model.to(device)


    print('\n==> Setting optimizer and scheduler')
    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr*1e-6, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])


    print('\n==> Training model')
    ### Train model
    # train
    init_time = time.time()
    train_loss_all = []
    val_loss_all = []
    top1_acc_all = []
    top5_acc_all = []
    best_top1 = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        train_metrics = train_step(model, train_loader, optimizer, criterion, scheduler, epoch, device)
        val_metrics = validation_step(model, val_loader, criterion, device)

        # Get results
        train_loss = train_metrics['loss']
        val_loss = val_metrics['loss']
        top1 = val_metrics['top1']
        top5 = val_metrics['top5']

        # Save best top 1 model
        if top1 > best_top1:
            best_epoch = epoch
            best_top1 = top1
            if args.dp: state_dict = model.module.state_dict()
            else: state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(args.save_dir, f'best_resnet18_lineval.pth'))
            best_top5 = top5
        
        # Save current model (also will be the last model)
        if args.dp: state_dict = model.module.state_dict()
        else: state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(args.save_dir, f'checkpoint_resnet18_lineval.pth'))

        # Print results
        print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} --',
              f'Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))} --',
              f'Train Loss: {train_loss:.6f} -- Val Loss: {val_loss:.6f} -- Top1: {top1:.4f} -- Top5: {top5:.4f} ----',
              f'Best Top1: {best_top1:.4f} at epoch {best_epoch}, with Top5: {best_top5:.4f}')
       
        # Save results
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        top1_acc_all.append(top1)
        top5_acc_all.append(top5)
        np.save(os.path.join(args.save_dir, 'train_loss.npy'), np.array(train_loss_all))
        np.save(os.path.join(args.save_dir, 'val_loss.npy'), np.array(val_loss_all))
        np.save(os.path.join(args.save_dir, 'top1_acc.npy'), np.array(top1_acc_all))
        np.save(os.path.join(args.save_dir, 'top5_acc.npy'), np.array(top5_acc_all))

    print('\n==> Linear evaluation finished')

    return None

def train_step(model, train_loader, optimizer, criterion, scheduler, epoch, device):
    model.train()
    train_loss_meter = AverageMeter('Loss', ':.6f')
    for i, (x, labels) in enumerate(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        output = model(x)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_meter.update(loss.item(), x.size(0))
    return {'loss': train_loss_meter.avg}

def validation_step(model, val_loader, criterion, device):
    model.eval()
    val_loss_meter = AverageMeter('Loss', ':.6f')
    top1_meter = AverageMeter('Acc@1', ':6.2f')
    top5_meter = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for i, (x, labels) in enumerate(val_loader):
            x = x.to(device)
            labels = labels.to(device)
            output = model(x)
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            val_loss_meter.update(loss.item(), x.size(0))
            top1_meter.update(acc1[0].item(), x.size(0))
            top5_meter.update(acc5[0].item(), x.size(0))
    return {'loss': val_loss_meter.avg, 'top1': top1_meter.avg, 'top5': top5_meter.avg}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()

