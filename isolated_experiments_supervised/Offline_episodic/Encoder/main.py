import argparse
import os, time

import torch
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from models_encGNMish import *
from loss_functions import KoLeoLoss
from augmentations import Episode_Transformations

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Network parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
# Training parameters
parser.add_argument('--lr', type=float, default=0.01) # 0.003 for 128 total batchsize, 0.01 for 512 total batchsize
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--episode_batch_size', type=int, default=512) # 128 for 1 gpu, 512 for 4 gpus
parser.add_argument('--num_views', type=int, default=6)
parser.add_argument('--koleo_gamma', type=float, default=0.01)
parser.add_argument('--workers', type=int, default=48) # 8 for 1 gpu, 48 for 4 gpus
parser.add_argument('--save_dir', type=str, default="output/run_encoder_offline")
parser.add_argument('--print_frequency', type=int, default=5) # batch iterations.
parser.add_argument('--seed', type=int, default=0)
## DDP args
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    return None

def main():

    ### Parse arguments
    args = parser.parse_args()

    ### DDP init
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        args.ddp = True
        print(f"DDP used, local rank set to {args.local_rank}. {torch.distributed.get_world_size()} GPUs training.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.local_rank = 0
        args.ddp = False
        print("DDP not used, local rank set to 0. 1 GPU training.")

    # Create save dir folders and save args
    if args.local_rank == 0:
        print(args)
        if not os.path.exists(args.save_dir): # create save dir
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Calculate batch size per GPU
    if args.ddp:
        args.episode_batch_size_per_gpu = int(args.episode_batch_size / torch.distributed.get_world_size())
    else:
        args.episode_batch_size_per_gpu = args.episode_batch_size
    # Calculate number of workers per GPU
    if args.ddp:
        args.workers_per_gpu = int(args.workers / torch.distributed.get_world_size())
    else:
        args.workers_per_gpu = args.workers

    ### Seed everything
    final_seed = args.seed + args.local_rank
    seed_everything(seed=final_seed)

    ### Define tensoboard writer
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load data
    if args.local_rank == 0:
        print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=False)
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std),
                        ])
    train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
    val_dataset = datasets.ImageFolder(valdir, transform=val_transform)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True if train_sampler is None else False,
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                             sampler=val_sampler, num_workers=args.workers_per_gpu, pin_memory=True)

    ### Load models
    if args.local_rank == 0:
        print('\n==> Preparing network...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool=True)
    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.fc.weight.shape[1], num_classes=args.num_classes)
    view_encoder.fc = torch.nn.Identity() # remove last layer
                                                  
    ### Print models
    if args.local_rank == 0:
        print('\nView encoder')
        print(view_encoder)
        print('\nClassifier')
        print(classifier)
        print('\n')

    ### Dataparallel and move models to device
    view_encoder = view_encoder.to(device)
    classifier = classifier.to(device)
    if args.ddp:
        view_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(view_encoder)
        view_encoder = torch.nn.parallel.DistributedDataParallel(view_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)

    ### Load optimizer and criterion
    param_groups = [{'params': view_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': classifier.parameters(), 'lr': args.lr, 'weight_decay': args.wd}]
    optimizer = torch.optim.AdamW(param_groups, lr=0, weight_decay=0)
    criterion_sup = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_koleo = KoLeoLoss().to(device)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr*1e-6, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    #### Train loop ####
    if args.local_rank == 0:
        print('\n==> Training model')
    init_time = time.time()
    scaler = GradScaler()

    for epoch in range(args.epochs):
        start_time = time.time()

        if args.local_rank == 0:
            print(f'\n==> Epoch {epoch}/{args.epochs}')

        # DDP init
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        

        ##################
        ### Train STEP ###
        ##################
        loss_accum=0
        view_encoder.train()
        total_correct=0
        total_samples=0
        for i, (batch_episodes_imgs, batch_labels) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes_imgs.to(device, non_blocking=True) # (B, V, C, H, W)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True) # (B, V)

            # Forward pass
            loss_sup = 0
            loss_koleo = 0
            with (autocast()):
                for v in range(args.num_views):
                    batch_imgs = batch_episodes_imgs[:,v]
                    batch_labels = batch_episodes_labels[:,v]
                    batch_tensors = view_encoder(batch_imgs)
                    batch_logits = classifier(batch_tensors)
                    loss_sup += criterion_sup(batch_logits, batch_labels)
                    loss_koleo += criterion_koleo(batch_tensors.mean(dim=(2,3))) # pass the average pooled version (koleo works on vectors)
                    _, predicted = batch_logits.max(1)
                    correct = predicted.eq(batch_labels).sum().item()
                    total_correct += correct
                    total_samples += batch_labels.size(0)
                loss_sup /= args.num_views
                loss_koleo /= args.num_views
                loss = loss_sup + args.koleo_gamma * loss_koleo
                loss_accum += loss.item()

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update lr scheduler
            scheduler.step()

            # All reduce across GPUs for metric logging
            if args.ddp:
                torch.distributed.all_reduce(loss_sup, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(loss_koleo, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss_sup /= torch.distributed.get_world_size()
                loss_koleo /= torch.distributed.get_world_size()
                loss /= torch.distributed.get_world_size()

            if (args.local_rank == 0) and ((i % args.print_frequency) == 0):
                print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                      f'lr: {scheduler.get_last_lr()[0]:.6f} -- ' +
                      f'Loss Sup: {loss_sup.item():.6f} -- ' +
                      f'Loss Koleo: {loss_koleo.item():.6f} -- ' +
                      f'Loss Total: {loss.item():.6f}'
                    )
            if args.local_rank == 0:
                writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Supervised', loss_sup.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Koleo', loss_koleo.item(), epoch*len(train_loader)+i)            
                writer.add_scalar('Loss Total', loss.item(), epoch*len(train_loader)+i)

        # Train Epoch metrics
        if args.ddp:
            loss_accum = torch.tensor(loss_accum).to(device)
            total_correct = torch.tensor(total_correct).to(device)
            total_samples = torch.tensor(total_samples).to(device)
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
            loss_accum = loss_accum.item() / torch.distributed.get_world_size()
            total_correct = total_correct.item() / torch.distributed.get_world_size()
            total_samples = total_samples.item() / torch.distributed.get_world_size()
        loss_accum /= len(train_loader)
        accuracy = total_correct / total_samples
        if args.local_rank == 0:
            writer.add_scalar('Loss Total (per epoch)', loss_accum, epoch)
            writer.add_scalar('Accuracy (per epoch)', accuracy, epoch)
            print(f'Epoch [{epoch}] Total Train Loss per Epoch: {loss_accum:.6f}, Train Accuracy: {accuracy:.6f}')



        #######################
        ### Validation STEP ###
        #######################
        view_encoder.eval()
        loss_accum=0
        total_correct=0
        total_samples=0
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(val_loader):
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                with autocast():
                    batch_tensors = view_encoder(batch_imgs)
                    batch_logits = classifier(batch_tensors)
                    loss = criterion_sup(batch_logits, batch_labels)
                loss_accum += loss.item()
                _, predicted = batch_logits.max(1)
                correct = predicted.eq(batch_labels).sum().item()
                total_correct += correct
                total_samples += batch_labels.size(0)

        # All reduce across GPUs
        if args.ddp:
            loss_accum = torch.tensor(loss_accum).to(device)
            total_correct = torch.tensor(total_correct).to(device)
            total_samples = torch.tensor(total_samples).to(device)
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
            loss_accum = loss_accum.item() / torch.distributed.get_world_size()
            total_correct = total_correct.item() / torch.distributed.get_world_size()
            total_samples = total_samples.item() / torch.distributed.get_world_size()
        # Validation Epoch metrics
        loss_accum /= len(val_loader)
        accuracy = total_correct / total_samples
        if args.local_rank == 0:
            writer.add_scalar('Loss Total Validation (per epoch)', loss_accum, epoch)
            writer.add_scalar('Accuracy Validation (per epoch)', accuracy, epoch)
            print(f'Epoch [{epoch}] Total Validation Loss per Epoch: {loss_accum:.6f}, Validation Accuracy: {accuracy:.6f}')

        ## Save model ##
        if (args.local_rank == 0) and (((epoch+1) % 10) == 0) or epoch==0:
            if args.ddp: 
                view_encoder_state_dict = view_encoder.module.state_dict()
                classifier_state_dict = classifier.module.state_dict()
            else: 
                view_encoder_state_dict = view_encoder.state_dict()
                classifier_state_dict = classifier.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(classifier_state_dict, os.path.join(args.save_dir, f'classifier_epoch{epoch}.pth'))

        if args.local_rank == 0:
            print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    return None

if __name__ == '__main__':
    main()