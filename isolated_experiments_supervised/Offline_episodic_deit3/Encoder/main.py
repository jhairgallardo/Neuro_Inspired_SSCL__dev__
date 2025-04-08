import argparse
import os, time

import torch
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from models_deit3 import *
from loss_functions import KoLeoLoss
from augmentations import Episode_Transformations
from utils import MetricLogger, accuracy

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
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--episode_batch_size', type=int, default=512) # 128 for 1 gpu, 512 for 4 gpus
parser.add_argument('--num_views', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.002) # 0.003 for 128 total batchsize, 0.01 for 512 total batchsize
parser.add_argument('--wd', type=float, default=0.0125)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--koleo_gamma', type=float, default=0) #0.01
parser.add_argument('--drop_path', type=float, default=0.0125)
# Other parameters
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
    
    # ### Plot some images as examples
    # if args.local_rank == 0:
    #     print('\n==> Plotting some images as examples...')
    #     import matplotlib.pyplot as plt
    #     import torchvision.utils as vutils
    #     import numpy as np
    #     batch_episodes_imgs, batch_labels = next(iter(train_loader))
    #     for i in range(5):
    #         episode = batch_episodes_imgs[i]
    #         label = batch_labels[i]
    #         grid = vutils.make_grid(episode, nrow=6, padding=2, normalize=True)
    #         plt.figure(figsize=(12, 8))
    #         plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
    #         plt.title(f'Labels: {label}')
    #         plt.axis('off')
    #         plt.show()
    #         plt.savefig(os.path.join(args.save_dir, f'example_images_{i}.png'), bbox_inches='tight', dpi=300)
    #         plt.close()

    ### Load models
    if args.local_rank == 0:
        print('\n==> Preparing network...')
    view_encoder = eval(args.enc_model_name)(drop_path_rate=args.drop_path, output_before_pool=True)
    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.embed_dim, num_classes=args.num_classes)
    view_encoder.head = torch.nn.Identity() # remove the head of the encoder
                                                  
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
    criterion_sup = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_koleo = KoLeoLoss().to(device)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1e-6/args.lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
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
        train_loss_total = MetricLogger('Train Loss Total')
        train_top1 = MetricLogger('Train Top1 ACC')
        train_top5 = MetricLogger('Train Top5 ACC')
        view_encoder.train()
        for i, (batch_episodes_imgs, batch_labels) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes_imgs.to(device, non_blocking=True) # (B, V, C, H, W)non_blocking=True)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True) # (B, V)

            ## Forward pass
            loss_sup = 0
            loss_koleo = 0
            acc1 = 0
            acc5 = 0
            with (autocast()):
                for v in range(args.num_views):
                    batch_imgs = batch_episodes_imgs[:,v]
                    batch_labels = batch_episodes_labels[:,v]
                    batch_tensors = view_encoder(batch_imgs)
                    batch_logits = classifier(batch_tensors)
                    loss_sup_view = criterion_sup(batch_logits, batch_labels)
                    loss_koleo_view = criterion_koleo(batch_tensors[:,0]) # apply koleo to the cls token vector
                    acc1_view, acc5_view = accuracy(batch_logits, batch_labels, topk=(1, 5))
                    loss_sup += loss_sup_view
                    loss_koleo += loss_koleo_view
                    acc1 += acc1_view
                    acc5 += acc5_view
                loss_sup /= args.num_views
                loss_koleo /= args.num_views
                acc1 /= args.num_views
                acc5 /= args.num_views
            # Calculate Total loss
            loss_total = loss_sup + args.koleo_gamma * loss_koleo

            ## Backward pass with clip norm
            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(view_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ## Track losses and acc for per epoch plotting
            train_loss_total.update(loss_total.item(), batch_imgs.size(0))
            train_top1.update(acc1.item(), batch_imgs.size(0))
            train_top5.update(acc5.item(), batch_imgs.size(0))

            if (args.local_rank == 0) and ((i % args.print_frequency) == 0):
                print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                      f'lr: {scheduler.get_last_lr()[0]:.6f} -- ' +
                      f'Loss Sup: {loss_sup.item():.6f} -- ' +
                      f'Loss Koleo: {loss_koleo.item():.6f} -- ' +
                      f'Loss Total: {loss_total.item():.6f}'
                    )
            if args.local_rank == 0:
                writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Supervised', loss_sup.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Koleo', loss_koleo.item(), epoch*len(train_loader)+i)            
                writer.add_scalar('Loss Total', loss_total.item(), epoch*len(train_loader)+i)
        
        # Train Epoch metrics
        if args.ddp:
            train_loss_total.all_reduce()
            train_top1.all_reduce()
            train_top5.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Total (per epoch)', train_loss_total.avg, epoch)
            writer.add_scalar('Accuracy (per epoch)', train_top1.avg/100.0, epoch)
            writer.add_scalar('Accuracy Top5 (per epoch)', train_top5.avg/100.0, epoch)
            print(f'Epoch [{epoch}] Train Loss Total: {train_loss_total.avg:.6f} -- Train Top1: {train_top1.avg/100.0:.3f} -- Train Top5: {train_top5.avg/100.0:.3f}')

        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes to finish the training epoch

        #######################
        ### Validation STEP ###
        #######################
        val_loss_total = MetricLogger('Val Loss Total')
        val_top1 = MetricLogger('Val Top1 ACC')
        val_top5 = MetricLogger('Val Top5 ACC')
        view_encoder.eval()
        for i, (batch_imgs, batch_labels) in enumerate(val_loader):
            with torch.no_grad():
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                with autocast():
                    batch_tensors = view_encoder(batch_imgs)
                    batch_logits = classifier(batch_tensors)
                    loss_sup = criterion_sup(batch_logits, batch_labels)
                    loss_koleo = criterion_koleo(batch_tensors[:,0]) # apply koleo to the cls token vector
                loss_total = loss_sup + args.koleo_gamma * loss_koleo
                acc1, acc5 = accuracy(batch_logits, batch_labels, topk=(1, 5))
                # Track losses and acc for per epoch plotting
                val_loss_total.update(loss_total.item(), batch_imgs.size(0))
                val_top1.update(acc1.item(), batch_imgs.size(0))
                val_top5.update(acc5.item(), batch_imgs.size(0))

        if args.ddp:
            val_loss_total.all_reduce()
            val_top1.all_reduce()
            val_top5.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Total Validation (per epoch)', val_loss_total.avg, epoch)
            writer.add_scalar('Accuracy Validation (per epoch)', val_top1.avg/100.0, epoch)
            writer.add_scalar('Accuracy Validation Top5 (per epoch)', val_top5.avg/100.0, epoch)
            print(f'Epoch [{epoch}] Val Loss Total: {val_loss_total.avg:.6f} -- Val Top1: {val_top1.avg/100.0:.3f} -- Val Top5: {val_top5.avg/100.0:.3f}')

        if args.ddp:
            torch.distributed.barrier() # Wait for all processes to finish the validation epoch

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