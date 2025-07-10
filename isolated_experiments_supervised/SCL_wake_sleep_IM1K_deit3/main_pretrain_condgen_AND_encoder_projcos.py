import argparse
import os, time

import torch
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torchvision

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental

# from models_deit3_projcos import *
from models_deit3_projcos_augcausal import *
from augmentations import Episode_Transformations, collate_function
from utils import MetricLogger, accuracy, time_duration_print

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet2012')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--num_pretraining_classes', type=int, default=10)#100)
parser.add_argument('--data_order_file_name', type=str, default='./IM1K_data_class_orders/imagenet_class_order_siesta.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
parser.add_argument('--enc_lr', type=float, default=0.001)
parser.add_argument('--enc_wd', type=float, default=0.05)
parser.add_argument('--drop_path', type=float, default=0.0125) # 0.0125 for tiny, 0.05 for small, 0.2 for base
# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--img_num_tokens', type=int, default=196)
parser.add_argument('--cond_num_layers', type=int, default=8)
parser.add_argument('--cond_nhead', type=int, default=8)
parser.add_argument('--cond_dim_ff', type=int, default=1024)
parser.add_argument('--cond_dropout', type=float, default=0)
parser.add_argument('--aug_feature_dim', type=int, default=64)
parser.add_argument('--aug_num_tokens_max', type=int, default=16)
parser.add_argument('--aug_n_layers', type=int, default=2)
parser.add_argument('--aug_n_heads', type=int, default=4)
parser.add_argument('--aug_dim_ff', type=int, default=256)
parser.add_argument('--upsampling_num_Blocks', type=list, default=[1,1,1,1])
parser.add_argument('--upsampling_num_out_channels', type=int, default=3)
parser.add_argument('--condgen_lr', type=float, default=0.001)
parser.add_argument('--condgen_wd', type=float, default=0)
# Conditional generator loss weight
parser.add_argument('--gen_alpha', type=float, default=1.0)
# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--episode_batch_size', type=int, default=80) #512
parser.add_argument('--num_views', type=int, default=4)
# Other parameters
parser.add_argument('--workers', type=int, default=48) # 8 for 1 gpu, 48 for 4 gpus
parser.add_argument('--save_dir', type=str, default="output/Pretrained_condgen_AND_enc/run_debug")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations.
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

def random_mask_tokens(tensor, mask_ratio=0.5):
    """
    Randomly mask tokens in a tensor along the sequence dimension.
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, T, D).
        mask_ratio (float): Ratio of tokens to mask.
    Returns:
        torch.Tensor: Tensor with masked tokens.
    """
    N, T, D = tensor.shape
    num_masked_tokens = int(T * mask_ratio)
    mask_indices = torch.randperm(T)[:num_masked_tokens]
    
    masked_tensor = tensor.clone()
    masked_tensor[:, mask_indices, :] = 0  # Set masked tokens to zero
    return masked_tensor

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
    '''Create tasks
    Note: To make continuum work with action_codes, I edited the following file:
    /home/jhair/anaconda3/envs/py39gpu/lib/python3.9/site-packages/continuum/tasks/image_array_task_set.py
    Specifically, line 112. The idea is to be able to pass a tuple variable on x, where the img is on x[0]
    Maybe I should create a fork version of continuum with that change (and install with setup.py)
    def _prepare_data(self, x, y, t):
        if self.trsf is not None:
            x = self.get_task_trsf(t)(x)
        if type(x) is tuple: ### Change to be able to output action vectors
            if not isinstance(x[0], torch.Tensor):
                x[0] = self._to_tensor(x[0])
        elif not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return x, y, t
    '''
    if args.local_rank == 0:
        print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std),
                        ])
    train_dataset_continuum = ImageFolderDataset(traindir)
    val_dataset_continuum = ImageFolderDataset(valdir)

    # Get data classfolder order from file (the file has order using folder names)
    with open(args.data_order_file_name, 'r') as f:
        data_classfolder_order = f.read().splitlines()
    # Create a classfolder to class index mapping
    classfolder_to_idx={}
    img_paths, classidxes, _ = train_dataset_continuum.get_data()
    for img_path, classidx in zip(img_paths, classidxes):
        class_name = str(img_path).split('/')[-2]
        if class_name not in classfolder_to_idx:
            classfolder_to_idx[class_name] = classidx
    idx_to_classfolder = {v: k for k, v in classfolder_to_idx.items()}
    # Transform the class folder order to class index order
    data_class_order = []
    for class_name in data_classfolder_order:
        if class_name in classfolder_to_idx:
            data_class_order.append(classfolder_to_idx[class_name])
        else:
            raise ValueError(f"Class {class_name} not found in the dataset.")

    train_tasks = ClassIncremental(train_dataset_continuum, increment=1, initial_increment=args.num_pretraining_classes, transformations=[train_transform], class_order=data_class_order)
    val_tasks = ClassIncremental(val_dataset_continuum, increment=1, initial_increment=args.num_pretraining_classes, transformations=[val_transform], class_order=data_class_order)
    train_dataset = train_tasks[0] # Create the train dataset taking only the first task (the first 100 classes)
    val_dataset = val_tasks[0] # Create the val dataset taking only the first task (the first 100 classes)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True if train_sampler is None else False,
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True,
                                               collate_fn=collate_function)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                             sampler=val_sampler, num_workers=args.workers_per_gpu, pin_memory=True)
    
    ### Plot some images as examples
    # if args.local_rank == 0:
    #     print('\n==> Plotting some images as examples...')
    #     import matplotlib.pyplot as plt
    #     import torchvision.utils as vutils
    #     import numpy as np
    #     batch_episodes, batch_labels, _ = next(iter(train_loader))
    #     for i in range(5):
    #         episode = batch_episodes[0][i]
    #         label = batch_labels[i]
    #         grid = vutils.make_grid(episode, nrow=args.num_views, padding=2, normalize=True)
    #         plt.figure(figsize=(12, 8))
    #         plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
    #         plt.title(f'Labels: {label}')
    #         plt.axis('off')
    #         plt.show()
    #         plt.savefig(os.path.join(args.save_dir, f'example_images_{i}.png'), bbox_inches='tight', dpi=300)
    #         plt.close()

    ### Load models
    if args.local_rank == 0:
        print('\n==> Prepare models...')
    view_encoder = eval(args.enc_model_name)(drop_path_rate=args.drop_path, output_before_pool=True)
    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.embed_dim, num_classes=args.num_classes)
    cond_generator = eval(args.condgen_model_name)(img_num_tokens=args.img_num_tokens,
                                                img_feature_dim = view_encoder.head.weight.shape[1],
                                                num_layers = args.cond_num_layers,
                                                nhead = args.cond_nhead,
                                                dim_ff = args.cond_dim_ff,
                                                drouput = args.cond_dropout,
                                                aug_num_tokens_max = args.aug_num_tokens_max,
                                                aug_feature_dim = args.aug_feature_dim,
                                                aug_n_layers = args.aug_n_layers,
                                                aug_n_heads = args.aug_n_heads,
                                                aug_dim_ff = args.aug_dim_ff,
                                                upsampling_num_Blocks = args.upsampling_num_Blocks,
                                                upsampling_num_out_channels = args.upsampling_num_out_channels)
    view_encoder.head = torch.nn.Identity() # remove the head of the encoder
                                                  
    ### Print models
    if args.local_rank == 0:
        print('\nView encoder')
        print(view_encoder)
        print('\nClassifier')
        print(classifier)
        print('\nConditional generator')
        print(cond_generator)
        print('\n')

    ### Dataparallel and move models to device
    view_encoder = view_encoder.to(device)
    classifier = classifier.to(device)
    cond_generator = cond_generator.to(device)
    if args.ddp:
        view_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(view_encoder)
        view_encoder = torch.nn.parallel.DistributedDataParallel(view_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
        cond_generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cond_generator)
        cond_generator = torch.nn.parallel.DistributedDataParallel(cond_generator, device_ids=[args.local_rank], output_device=args.local_rank)

    ### Load optimizer and criterion
    param_groups_encoder = [{'params': view_encoder.parameters(), 'lr': args.enc_lr, 'weight_decay': args.enc_wd},
                    {'params': classifier.parameters(), 'lr': args.enc_lr, 'weight_decay': args.enc_wd}]
    optimizer_encoder = torch.optim.AdamW(param_groups_encoder, lr=0, weight_decay=0)
    linear_warmup_scheduler_encoder = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_encoder, start_factor=1e-6/args.enc_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_encoder = torch.optim.lr_scheduler.SequentialLR(optimizer_encoder, [linear_warmup_scheduler_encoder, cosine_scheduler_encoder], milestones=[args.warmup_epochs*len(train_loader)])

    optimizer_condgen = torch.optim.AdamW(cond_generator.parameters(), lr=args.condgen_lr, weight_decay=args.condgen_wd)
    linear_warmup_scheduler_condgen = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_condgen, start_factor=1e-6/args.condgen_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_condgen = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_condgen, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_condgen = torch.optim.lr_scheduler.SequentialLR(optimizer_condgen, [linear_warmup_scheduler_condgen, cosine_scheduler_condgen], milestones=[args.warmup_epochs*len(train_loader)])

    criterion_sup = torch.nn.CrossEntropyLoss()
    criterion_condgen = torch.nn.MSELoss()

    ### Save one batch for plot purposes
    seed_everything(final_seed)  # Reset seed to ensure reproducibility for the batch
    if args.local_rank == 0:
        episodes_plot, _, _ = next(iter(train_loader))

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
            val_sampler.set_epoch(epoch)
        
        ##################
        ### Train STEP ###
        ##################
        losscondgen_total_log = MetricLogger('LossCondgen Total')
        loss_gen1_log = MetricLogger('Loss Gen1')
        loss_gen2_log = MetricLogger('Loss Gen2')
        loss_gen3_log = MetricLogger('Loss Gen3')

        train_loss_total = MetricLogger('Train Loss Total')
        train_top1 = MetricLogger('Train Top1 ACC')
        train_top5 = MetricLogger('Train Top5 ACC')

        view_encoder.train()
        cond_generator.train()
        for i, (batch_episodes, batch_labels, _) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes[0].to(device, non_blocking=True) # (B, V, C, H, W)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True) # (B, V)
            batch_episodes_actions = batch_episodes[1] # (B, V, A)

            ## Forward pass
            loss_gen1 = 0
            loss_gen2 = 0
            loss_gen3 = 0
            loss_sup = 0
            acc1 = 0
            acc5 = 0
            B, V, C, H, W = batch_episodes_imgs.shape
            with (autocast()):
                # Flatten the batch and views
                flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                flat_feats_and_cls = view_encoder(flat_imgs) # (B*V, 1+T, D)
                # Reshape and get first view features
                all_feats = flat_feats_and_cls.view(B, V, flat_feats_and_cls.size(1), -1) # (B, V, 1+T, D)
                first_view_feats = all_feats[:, 0, 1:, :].detach() # (B, T, D) # Discard the CLS token. Shape is (B, T, D)
                # Reshape to get the CLS token and features
                flat_tensors = all_feats[:, :, 1:, :].reshape(B * V, -1, all_feats.size(-1))  # → (B·V, T, D)
                flat_cls = all_feats[:, :, 0, :].reshape(B * V, all_feats.size(-1))    # → (B·V, D)
                # Reshape and expand the first view features
                flat_first_feats = first_view_feats.unsqueeze(1)  # (B, 1,  T, D)
                flat_first_feats = flat_first_feats.expand(-1, V, -1, -1) # (B, V,  T, D)
                flat_first_feats = flat_first_feats.reshape(B * V, *first_view_feats.shape[1:])   # (B*V, T, D)
                # flat_first_feats = random_mask_tokens(flat_first_feats, mask_ratio=0.1)  # (B*V, T, D)
                # Get actions
                flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V
                # Run the conditional generator
                flat_gen_imgs, flat_gen_feats = cond_generator(flat_first_feats, flat_actions) # (B*V, C, H, W), (B*V, T, D)
                flat_gen_dec_feats = view_encoder(flat_gen_imgs)[:, 1:, :]  # (B*V, T, D)
                # Run the generator directly (skip conditioning)
                flat_gen_imgs_dir = cond_generator(flat_tensors, None, skip_conditioning=True)  # (B*V, C, H, W)
                flat_gen_dir_feats = view_encoder(flat_gen_imgs_dir)[:, 1:, :]                  # (B*V, T, D)
                # Get generator losses
                loss_gen1 = criterion_condgen(flat_gen_feats, flat_tensors.detach())
                loss_gen2 = criterion_condgen(flat_gen_dec_feats, flat_tensors.detach())
                loss_gen3 = criterion_condgen(flat_gen_dir_feats, flat_tensors.detach())

                # Supervised loss & accuracy on v≠0
                mask = torch.ones(B, V, dtype=torch.bool, device=device)
                mask[:, 0] = False                                                                     # zero out first view
                flat_mask = mask.reshape(-1)                                                           # (B*V,)
                sup_logits = classifier(flat_cls[flat_mask])                                           # (B*(V-1), num_classes)
                sup_labels = batch_episodes_labels.reshape(-1)[flat_mask]                              # (B*(V-1),)
                loss_sup  = criterion_sup(sup_logits, sup_labels)
                acc1, acc5 = accuracy(sup_logits, sup_labels, topk=(1, 5))
 
            # Calculate Total loss for the batch
            losssup_total = loss_sup
            losscondgen_total = loss_gen1 + loss_gen2 + loss_gen3
            loss_total = losssup_total + losscondgen_total*args.gen_alpha

            ## Backward pass with clip norm
            optimizer_encoder.zero_grad()
            optimizer_condgen.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer_encoder)
            torch.nn.utils.clip_grad_norm_(view_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optimizer_encoder)
            scaler.unscale_(optimizer_condgen)
            torch.nn.utils.clip_grad_norm_(cond_generator.parameters(), 1.0)
            scaler.step(optimizer_condgen)
            scaler.update()
            scheduler_encoder.step()
            scheduler_condgen.step()

            ## Track losses for per batch plotting (Encoder)
            train_loss_total.update(losssup_total.item(), batch_episodes_imgs.size(0))
            train_top1.update(acc1.item(), batch_episodes_imgs.size(0))
            train_top5.update(acc5.item(), batch_episodes_imgs.size(0))

            ## Track losses for per epoch plotting (CondGen)
            losscondgen_total_log.update(losscondgen_total.item(), batch_episodes_imgs.size(0))
            loss_gen1_log.update(loss_gen1.item(), batch_episodes_imgs.size(0))
            loss_gen2_log.update(loss_gen2.item(), batch_episodes_imgs.size(0))
            loss_gen3_log.update(loss_gen3.item(), batch_episodes_imgs.size(0))

            if (args.local_rank == 0) and ((i % args.print_frequency) == 0):
                print(
                    f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                    f'lr Encoder: {scheduler_encoder.get_last_lr()[0]:.6f} -- ' +
                    f'Loss Sup: {losssup_total.item():.6f} -- ' +
                    f'lr CondGen: {scheduler_condgen.get_last_lr()[0]:.6f} -- ' +
                    f'Loss Gen1: {loss_gen1.item():.6f} -- ' +
                    f'Loss Gen2: {loss_gen2.item():.6f} -- ' +
                    f'Loss Gen3: {loss_gen3.item():.6f} -- ' +
                    f'LossCondgen Total: {losscondgen_total.item():.6f}'
                    )
            if args.local_rank == 0:
                writer.add_scalar('lr Encoder', scheduler_encoder.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Supervised', losssup_total.item(), epoch*len(train_loader)+i)
                writer.add_scalar('lr CondGen', scheduler_condgen.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Gen1', loss_gen1.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Gen2', loss_gen2.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Gen3', loss_gen3.item(), epoch*len(train_loader)+i)
                writer.add_scalar('LossCondgen Total', losscondgen_total.item(), epoch*len(train_loader)+i)
        
        # Train Epoch metrics
        if args.ddp:
            train_loss_total.all_reduce()
            losscondgen_total_log.all_reduce()
            loss_gen1_log.all_reduce()
            loss_gen2_log.all_reduce()
            loss_gen3_log.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Supervised (per epoch)', train_loss_total.avg, epoch)
            writer.add_scalar('Accuracy (per epoch)', train_top1.avg/100.0, epoch)
            writer.add_scalar('Accuracy Top5 (per epoch)', train_top5.avg/100.0, epoch)
            writer.add_scalar('LossCondgen Total (per epoch)', losscondgen_total_log.avg, epoch)
            writer.add_scalar('Loss Gen1 (per epoch)', loss_gen1_log.avg, epoch)
            writer.add_scalar('Loss Gen2 (per epoch)', loss_gen2_log.avg, epoch)
            writer.add_scalar('Loss Gen3 (per epoch)', loss_gen3_log.avg, epoch)
            print(f'Epoch [{epoch}] Loss Supervised: {train_loss_total.avg:.6f} -- LossCondgen Total: {losscondgen_total_log.avg:.6f} -- Loss Gen1: {loss_gen1_log.avg:.6f} -- Loss Gen2: {loss_gen2_log.avg:.6f} -- Loss Gen3: {loss_gen3_log.avg:.6f}')

        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes to finish the training epoch

        #######################
        ### Validation STEP ###
        #######################

        # This is only for the supervised task
        val_loss_total = MetricLogger('Val Loss Total')
        val_top1 = MetricLogger('Val Top1 ACC')
        val_top5 = MetricLogger('Val Top5 ACC')
        view_encoder.eval()
        for i, (batch_imgs, batch_labels, _) in enumerate(val_loader):
            with torch.no_grad():
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                with autocast():
                    batch_tensors = view_encoder(batch_imgs)
                    batch_logits = classifier(batch_tensors[:,0]) # pass cls token to classifier
                    loss_sup = criterion_sup(batch_logits, batch_labels)
                losssup_total_val = loss_sup
                acc1, acc5 = accuracy(batch_logits, batch_labels, topk=(1, 5))
                # Track losses and acc for per epoch plotting
                val_loss_total.update(losssup_total_val.item(), batch_imgs.size(0))
                val_top1.update(acc1.item(), batch_imgs.size(0))
                val_top5.update(acc5.item(), batch_imgs.size(0))
        
        if args.ddp:
            val_loss_total.all_reduce()
            val_top1.all_reduce()
            val_top5.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Supervised Validation (per epoch)', val_loss_total.avg, epoch)
            writer.add_scalar('Accuracy Validation (per epoch)', val_top1.avg/100.0, epoch)
            writer.add_scalar('Accuracy Validation Top5 (per epoch)', val_top5.avg/100.0, epoch)
            print(f'Epoch [{epoch}] Val Loss Total: {val_loss_total.avg:.6f} -- Val Top1: {val_top1.avg/100.0:.3f} -- Val Top5: {val_top5.avg/100.0:.3f}')

        if args.ddp:
            torch.distributed.barrier() # Wait for all processes to finish the validation epoch

        ### Save model ###
        if (args.local_rank == 0) and (((epoch+1) % 10) == 0) or epoch==0:
            if args.ddp:
                view_encoder_state_dict = view_encoder.module.state_dict()
                classifier_state_dict = classifier.module.state_dict()
                cond_generator_state_dict = cond_generator.module.state_dict()
            else:
                view_encoder_state_dict = view_encoder.state_dict()
                classifier_state_dict = classifier.state_dict()
                cond_generator_state_dict = cond_generator.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(classifier_state_dict, os.path.join(args.save_dir, f'classifier_epoch{epoch}.pth'))
            torch.save(cond_generator_state_dict, os.path.join(args.save_dir, f'cond_generator_epoch{epoch}.pth'))

        ### Plot reconstructions examples ###
        if args.local_rank == 0:
            if (epoch+1) % 5 == 0 or epoch==0:
                view_encoder.eval()
                cond_generator.eval()
                n = 16
                episodes_plot_imgs = episodes_plot[0][:n].to(device, non_blocking=True)
                episodes_plot_actions = episodes_plot[1][:n]
                episodes_plot_gen_imgs = torch.empty(0)
                with torch.no_grad():
                    first_view_tensors = view_encoder(episodes_plot_imgs[:,0])[:, 1:, :] # Discard the CLS token. Shape is (B, T, D)
                    for v in range(args.num_views):
                        actions = [episodes_plot_actions[j][v] for j in range(episodes_plot_imgs.shape[0])]
                        gen_images, _ = cond_generator(first_view_tensors, actions)
                        episodes_plot_gen_imgs = torch.cat([episodes_plot_gen_imgs, gen_images.unsqueeze(1).detach().cpu()], dim=1)
                episodes_plot_imgs = episodes_plot_imgs.detach().cpu()
                # plot each episode
                for i in range(n):
                    episode_i_imgs = episodes_plot_imgs[i]
                    episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_imgs]
                    episode_i_imgs = torch.stack(episode_i_imgs, dim=0)

                    episode_i_gen_imgs = episodes_plot_gen_imgs[i]
                    episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_gen_imgs]
                    episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0)
                    episode_i_gen_imgs = torch.clamp(episode_i_gen_imgs, 0, 1) # Clip values to [0, 1]

                    grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs], dim=0), nrow=args.num_views)
                    grid = grid.permute(1, 2, 0).cpu().numpy()
                    grid = (grid * 255).astype(np.uint8)
                    grid = Image.fromarray(grid)
                    image_name = f'epoch{epoch}_episode{i}.png'
                    save_plot_dir = os.path.join(args.save_dir, 'gen_plots')
                    # create folder if it doesn't exist
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    grid.save(os.path.join(save_plot_dir, image_name))

        if args.local_rank == 0:
            epoch_time = time.time() - start_time
            elapsed_time = time.time() - init_time
            print(f"Epoch [{epoch}] Epoch Time: {time_duration_print(epoch_time)} -- Elapsed Time: {time_duration_print(elapsed_time)}")

    return None

if __name__ == '__main__':
    main()