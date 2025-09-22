import argparse
import os, time

import torch
import torchvision
from torchvision import transforms

from torch.amp import GradScaler
from torch.amp import autocast

from continuum.datasets import InMemoryDataset
from continuum import InstanceIncremental
from continuum.tasks import TaskType

from models_deit3_clscausal import *
from utils import get_imgpath_label, MetricLogger, reduce_tensor, accuracy, time_duration_print

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/home/jhair/datasets/Create_ImageNet100_ID_OOD/ImageNet-100_gmedian_quantile@alpha0.9_id_ood_dinov3_vits16plus')
parser.add_argument('--class_idx_file', type=str, default='./IM100_class_index/imagenet100_class_index.json')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Pre-trained folder
parser.add_argument('--pretrained_folder', type=str, default='./output/Pretrained_IM100B/Causal_deit_tiny_patch16_LS_100c_4viewsCLSallviews_bs80_epochs100_ENC_lr0.0005wd0.05_CONDGEN_lr0.0003wd0_loss@ent0.1firstview@penalty0.1CEweight0.1_sup1.0_condgen1.0_attndiv1.0_seed0')
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--enc_model_checkpoint', type=str, default='view_encoder_epoch99.pth')
parser.add_argument('--drop_path', type=float, default=0)
# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=512) 
parser.add_argument('--label_smoothing', type=float, default=0.0) 
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.0008)
# Other parameters
parser.add_argument('--workers', type=int, default=32) # 8 for 1 gpu, 48 for 4 gpus
parser.add_argument('--save_dir', type=str, default="output/Linear_probing/run_debug")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations.
parser.add_argument('--seed', type=int, default=0)
## DDP args
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

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

def training_step(train_loader, view_encoder, linear_cls, criterion, scaler, optimizer, scheduler, device, ddp):

    view_encoder.eval()
    linear_cls.train()

    results = {'acc1': [], 'acc5': [], 'loss': []}
    train_loss_total = MetricLogger('Train Loss Total')
    train_top1 = MetricLogger('Train Top1 ACC')
    train_top5 = MetricLogger('Train Top5 ACC')

    for i, (batch_imgs, batch_labels, _) in enumerate(train_loader):
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            batch_tensors = view_encoder(batch_imgs)
            batch_cls_tokens = batch_tensors[:, 0, :] # (B, D)
            batch_logits = linear_cls(batch_cls_tokens) # (B, num_classes)
            loss_sup = criterion(batch_logits, batch_labels)
            acc1, acc5 = accuracy(batch_logits, batch_labels, topk=(1, 5))
        # backward pass with scaler
        optimizer.zero_grad()
        scaler.scale(loss_sup).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # Track losses and acc for per epoch plotting
        train_loss_total.update(loss_sup.item(), batch_imgs.size(0))
        train_top1.update(acc1.item(), batch_imgs.size(0))
        train_top5.update(acc5.item(), batch_imgs.size(0))
    
    if ddp:
        train_loss_total.all_reduce()
        train_top1.all_reduce()
        train_top5.all_reduce()
    
    results['acc1'] = train_top1.avg
    results['acc5'] = train_top5.avg
    results['loss'] = train_loss_total.avg
    
    return results

def validation_step(val_loader, view_encoder, linear_cls, criterion, device, ddp):

    view_encoder.eval()
    linear_cls.eval()

    results = {'acc1': [], 'acc5': [], 'loss': []}
    val_loss_total = MetricLogger('Val Loss Total')
    val_top1 = MetricLogger('Val Top1 ACC')
    val_top5 = MetricLogger('Val Top5 ACC')

    for i, (batch_imgs, batch_labels, _) in enumerate(val_loader):
        with torch.no_grad():
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                batch_tensors = view_encoder(batch_imgs)
                batch_cls_tokens = batch_tensors[:, 0, :] # (B, D)
                batch_logits = linear_cls(batch_cls_tokens) # (B, num_classes)
            loss_sup = criterion(batch_logits, batch_labels)
            acc1, acc5 = accuracy(batch_logits, batch_labels, topk=(1, 5))
        # Track losses and acc for per epoch plotting
        val_loss_total.update(loss_sup.item(), batch_imgs.size(0))
        val_top1.update(acc1.item(), batch_imgs.size(0))
        val_top5.update(acc5.item(), batch_imgs.size(0))
    
    if ddp:
        val_loss_total.all_reduce()
        val_top1.all_reduce()
        val_top5.all_reduce()
    
    results['acc1'] = val_top1.avg
    results['acc5'] = val_top5.avg
    results['loss'] = val_loss_total.avg

    return results

def main():

    ### Parse arguments
    args = parser.parse_args()

    ### DDP init
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://', device_id=device)
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

    # Calculate batch size and workers per GPU
    if args.ddp:
        args.batch_size_per_gpu = int(args.batch_size / torch.distributed.get_world_size())
        args.workers_per_gpu = int(args.workers / torch.distributed.get_world_size())
    else:
        args.batch_size_per_gpu = args.batch_size
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
    # Get transforms
    # train_transform = transforms.Compose([
    #                     transforms.RandomResizedCrop(224),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize(mean=args.mean, std=args.std),
    #                     ])
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std),
                        ])
    # No aug #######################################################################
    train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=args.mean, std=args.std),
                    ])


    # Get img_paths, labels and tasks
    traindir = os.path.join(args.data_path, 'train_ID')
    valdir_ID = os.path.join(args.data_path, 'val_ID')
    valdir_OOD = os.path.join(args.data_path, 'val_OOD')
    train_imgpaths, train_labels_array = get_imgpath_label(traindir, args.class_idx_file)
    val_imgpaths_ID, val_labels_array_ID = get_imgpath_label(valdir_ID, args.class_idx_file)
    val_imgpaths_OOD, val_labels_array_OOD = get_imgpath_label(valdir_OOD, args.class_idx_file)

    # Get datasets and loaders
    train_dataset_continuum = InMemoryDataset(train_imgpaths, train_labels_array, data_type=TaskType.IMAGE_PATH)
    val_dataset_continuum_ID = InMemoryDataset(val_imgpaths_ID, val_labels_array_ID, data_type=TaskType.IMAGE_PATH)
    val_dataset_continuum_OOD = InMemoryDataset(val_imgpaths_OOD, val_labels_array_OOD, data_type=TaskType.IMAGE_PATH)
    train_dataset = InstanceIncremental(train_dataset_continuum, transformations=[train_transform], nb_tasks=1)[0]
    val_dataset_ID = InstanceIncremental(val_dataset_continuum_ID, transformations=[val_transform], nb_tasks=1)[0]
    val_dataset_OOD = InstanceIncremental(val_dataset_continuum_OOD, transformations=[val_transform], nb_tasks=1)[0]
    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish

    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler_ID = torch.utils.data.distributed.DistributedSampler(val_dataset_ID, shuffle=False)
        val_sampler_OOD = torch.utils.data.distributed.DistributedSampler(val_dataset_OOD, shuffle=False)
    else:
        train_sampler = None
        val_sampler_ID = None
        val_sampler_OOD = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_per_gpu, shuffle=True if train_sampler is None else False,
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True)
    val_loader_ID = torch.utils.data.DataLoader(val_dataset_ID, batch_size=args.batch_size_per_gpu, shuffle=False,
                                             sampler=val_sampler_ID, num_workers=args.workers_per_gpu, pin_memory=True)
    val_loader_OOD = torch.utils.data.DataLoader(val_dataset_OOD, batch_size=args.batch_size_per_gpu, shuffle=False,
                                                 sampler=val_sampler_OOD, num_workers=args.workers_per_gpu, pin_memory=True)
    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish

    ### Load models
    if args.local_rank == 0:
        print('\n==> Prepare models...')
    view_encoder = eval(args.enc_model_name)(drop_path_rate=args.drop_path, output_before_pool=True)
    view_encoder.head = torch.nn.Identity() # remove the head of the encoder
    linear_cls = torch.nn.Linear(view_encoder.embed_dim, args.num_classes)

    ### Load pre-trained view encoder
    if args.enc_model_checkpoint is not None:
        if args.local_rank == 0:
            print(f'Loading view encoder from {args.pretrained_folder}/{args.enc_model_checkpoint}')
        view_encoder.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.enc_model_checkpoint), map_location=device), strict=True)
                                       
    ### Print models
    if args.local_rank == 0:
        print('\nView encoder')
        print(view_encoder)
        print('\nLinear classifier')
        print(linear_cls)

    ### Dataparallel and move models to device
    view_encoder = view_encoder.to(device)
    linear_cls = linear_cls.to(device)
    if args.ddp:
        view_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(view_encoder)
        view_encoder = torch.nn.parallel.DistributedDataParallel(view_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        linear_cls = torch.nn.parallel.DistributedDataParallel(linear_cls, device_ids=[args.local_rank], output_device=args.local_rank)

    ### Freeze view encoder
    for param in view_encoder.parameters():
        param.requires_grad = False
    view_encoder.eval()

    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(linear_cls.parameters(), lr=args.lr, weight_decay= args.wd)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1e-6/args.lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    #######################################
    ### Validation STEP before training ###
    #######################################
    results_ID = validation_step(val_loader_ID, view_encoder, linear_cls, criterion, device, args.ddp)  
    results_OOD = validation_step(val_loader_OOD, view_encoder, linear_cls, criterion, device, args.ddp)
    if args.local_rank == 0:
        print(f'Epoch [{-1}] Val_ID --> Loss: {results_ID["loss"]:.6f} -- Top1 Acc: {results_ID["acc1"]:.2f} -- Top5 Acc: {results_ID["acc5"]:.2f}')
        print(f'Epoch [{-1}] Val_OOD --> Loss: {results_OOD["loss"]:.6f} -- Top1 Acc: {results_OOD["acc1"]:.2f} -- Top5 Acc: {results_OOD["acc5"]:.2f}')
    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish the validation step

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
        if args.ddp:
            train_sampler.set_epoch(epoch)
            val_sampler_ID.set_epoch(epoch)
            val_sampler_OOD.set_epoch(epoch)
        
        ##################
        ### Train STEP ###
        ##################
        results = training_step(train_loader, view_encoder, linear_cls, criterion, scaler, optimizer, scheduler, device, args.ddp)
        if args.local_rank == 0:
            print(f'Epoch [{epoch}] Loss: {results["loss"]:.6f} -- Top1 Acc: {results["acc1"]:.2f} -- Top5 Acc: {results["acc5"]:.2f}')
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch*len(train_loader))
            writer.add_scalar('Loss', results["loss"], epoch*len(train_loader))
            writer.add_scalar('Top1 Acc', results["acc1"], epoch*len(train_loader))
            writer.add_scalar('Top5 Acc', results["acc5"], epoch*len(train_loader))

        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes to finish the training epoch

        #######################
        ### Validation STEP ###
        #######################

        results_ID = validation_step(val_loader_ID, view_encoder, linear_cls, criterion, device, args.ddp)  
        results_OOD = validation_step(val_loader_OOD, view_encoder, linear_cls, criterion, device, args.ddp)
        if args.local_rank == 0:
            print(f'Epoch [{epoch}] Val_ID --> Loss: {results_ID["loss"]:.6f} -- Top1 Acc: {results_ID["acc1"]:.2f} -- Top5 Acc: {results_ID["acc5"]:.2f}')
            print(f'Epoch [{epoch}] Val_OOD --> Loss: {results_OOD["loss"]:.6f} -- Top1 Acc: {results_OOD["acc1"]:.2f} -- Top5 Acc: {results_OOD["acc5"]:.2f}')
            writer.add_scalar('Val_ID Loss', results_ID["loss"], epoch*len(val_loader_ID))
            writer.add_scalar('Val_ID Top1 Acc', results_ID["acc1"], epoch*len(val_loader_ID))
            writer.add_scalar('Val_ID Top5 Acc', results_ID["acc5"], epoch*len(val_loader_ID))
            writer.add_scalar('Val_OOD Loss', results_OOD["loss"], epoch*len(val_loader_OOD))
            writer.add_scalar('Val_OOD Top1 Acc', results_OOD["acc1"], epoch*len(val_loader_OOD))
            writer.add_scalar('Val_OOD Top5 Acc', results_OOD["acc5"], epoch*len(val_loader_OOD))
        if args.ddp:
            torch.distributed.barrier() # Wait for all processes to finish the validation epoch

        ### Save model ###
        if (args.local_rank == 0) and (((epoch+1) % 10) == 0) or epoch==0:
            if args.ddp:
                view_encoder_state_dict = view_encoder.module.state_dict()
                linear_cls_state_dict = linear_cls.module.state_dict()
            else:
                view_encoder_state_dict = view_encoder.state_dict()
                linear_cls_state_dict = linear_cls.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(linear_cls_state_dict, os.path.join(args.save_dir, f'linear_cls_epoch{epoch}.pth'))

        if args.local_rank == 0:
            epoch_time = time.time() - start_time
            elapsed_time = time.time() - init_time
            print(f"Epoch [{epoch}] Epoch Time: {time_duration_print(epoch_time)} -- Elapsed Time: {time_duration_print(elapsed_time)}")

    # Close tensorboard writer
    if args.local_rank == 0:
        writer.close()

    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish the validation step
        torch.distributed.destroy_process_group()  # Destroy the process group

    return None

if __name__ == '__main__':
    main()