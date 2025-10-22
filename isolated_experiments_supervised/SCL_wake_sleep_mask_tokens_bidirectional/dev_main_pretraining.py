import argparse
import os, time

import torch
from torchvision import transforms

from torch.amp import GradScaler
from torch.amp import autocast

import torchvision
from torchvision.datasets import ImageFolder

from dev_models_deit3 import *
from augs_episodes_firstbigcontext import Episode_Transformations, collate_function_notaskid, DeterministicEpisodes, ImageFolderDetEpisodes
from utils import MetricLogger, reduce_tensor, accuracy, time_duration_print, build_stratified_indices, make_plot_batch

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-100B')
parser.add_argument('--val_episode_seed', type=int, default=12345)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--mean', type=list, default=[0.5, 0.5, 0.5])
parser.add_argument('--std', type=list, default=[0.5, 0.5, 0.5])
# View encoder parameters
parser.add_argument('--view_enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--venc_lr', type=float, default=0.0005)
parser.add_argument('--venc_wd', type=float, default=0.05)
parser.add_argument('--venc_drop_path', type=float, default=0.0125) # 0.0125 for tiny, 0.05 for small, 0.2 for base
# Classifier parameters
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Network')
# Action encoder parameters
parser.add_argument('--action_enc_model_name', type=str, default='Action_Encoder_Network')
parser.add_argument('--act_enc_dim', type=int, default=64)
parser.add_argument('--act_enc_n_layers', type=int, default=2)
parser.add_argument('--act_enc_n_heads', type=int, default=4)
parser.add_argument('--act_enc_dim_ff', type=int, default=256)
# View predictor parameters
parser.add_argument('--view_predictor_model_name', type=str, default='View_Predictor_Network')
parser.add_argument('--vpred_lr', type=float, default=0.0003)
parser.add_argument('--vpred_wd', type=float, default=0)
parser.add_argument('--vpred_dim', type=int, default=256)
parser.add_argument('--vpred_n_layers', type=int, default=8)
parser.add_argument('--vpred_n_heads', type=int, default=8)
parser.add_argument('--vpred_dim_ff', type=int, default=1024)
parser.add_argument('--vpred_dropout', type=float, default=0)
# Generator parameters
parser.add_argument('--generator_model_name', type=str, default='Generator_Network')
parser.add_argument('--gen_num_Blocks', type=list, default=[1,1,1,1])
# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--episode_batch_size', type=int, default=10)#64) #80
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--label_smoothing', type=float, default=0.0) # Label smoothing for the supervised loss
# Other parameters
parser.add_argument('--workers', type=int, default=32) # 8 for 1 gpu, 32 for 4 gpus
parser.add_argument('--save_dir', type=str, default="output/Pretrained_IM100B/run_debug")
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

    ### Load Training data
    if args.local_rank == 0:
        print('\n==> Preparing Training data...')
    traindir = os.path.join(args.data_path, 'train')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
    train_dataset = ImageFolder(traindir, transform=train_transform)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True if train_sampler is None else False,
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True,
                                               collate_fn=collate_function_notaskid)
    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish

    ### Load Validation data
    if args.local_rank == 0:
        print('\n==> Preparing Validation data...')
    valdir = os.path.join(args.data_path, 'val')
    val_base_transform = Episode_Transformations(num_views=args.num_views, mean=args.mean, std=args.std)
    val_transform = DeterministicEpisodes(val_base_transform, base_seed=args.val_episode_seed)
    val_dataset = ImageFolderDetEpisodes(valdir, transform=val_transform)
    if args.ddp:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                             sampler=val_sampler, num_workers=args.workers_per_gpu, pin_memory=True,
                                             collate_fn=collate_function_notaskid)
    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish
    

    ### Load models
    if args.local_rank == 0:
        print('\n==> Prepare models...')
    view_encoder = eval(args.view_enc_model_name)(drop_path_rate=args.venc_drop_path, output_before_pool=True)
    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.embed_dim, num_classes=args.num_classes)
    action_encoder = eval(args.action_enc_model_name)(d_model=args.act_enc_dim, n_layers=args.act_enc_n_layers, n_heads=args.act_enc_n_heads, dim_ff=args.act_enc_dim_ff)
    view_predictor = eval(args.view_predictor_model_name)(d_model=args.vpred_dim, 
                                                          n_img_tokens=view_encoder.patch_embed.num_patches,
                                                          imgfttok_dim=view_encoder.embed_dim,
                                                          acttok_dim=args.act_enc_dim,
                                                          num_layers=args.vpred_n_layers,
                                                          nhead=args.vpred_n_heads,
                                                          dim_ff=args.vpred_dim_ff,
                                                          dropout=args.vpred_dropout)


    generator = eval(args.generator_model_name)(in_planes=view_encoder.embed_dim, num_Blocks=args.gen_num_Blocks, nc=3)
    view_encoder.head = torch.nn.Identity() # remove the head of the encoder    
                                                  
    ### Print models
    if args.local_rank == 0:
        print('\nView encoder')
        print(view_encoder)
        print('\nClassifier')
        print(classifier)
        print('\nAction encoder')
        print(action_encoder)
        print('\nView predictor')
        print(view_predictor)
        print('\nGenerator')
        print(generator)
        print('\n')

    ### Dataparallel and move models to device
    view_encoder = view_encoder.to(device)
    classifier = classifier.to(device)
    action_encoder = action_encoder.to(device)
    view_predictor = view_predictor.to(device)
    generator = generator.to(device)
    if args.ddp:
        view_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(view_encoder)
        view_encoder = torch.nn.parallel.DistributedDataParallel(view_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
        action_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(action_encoder)
        action_encoder = torch.nn.parallel.DistributedDataParallel(action_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        view_predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(view_predictor)
        view_predictor = torch.nn.parallel.DistributedDataParallel(view_predictor, device_ids=[args.local_rank], output_device=args.local_rank)
        generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank], output_device=args.local_rank)

    ### Load optimizers and learning rate schedulers
    # View encoder and classifier share the same optimizer and scheduler
    param_groups_venc_cls = [{'params': view_encoder.parameters(), 'lr': args.venc_lr, 'weight_decay': args.venc_wd},
                            {'params': classifier.parameters(), 'lr': args.venc_lr, 'weight_decay': args.venc_wd}]
    optimizer_venc_cls = torch.optim.AdamW(param_groups_venc_cls, lr=0, weight_decay=0)
    linear_warmup_scheduler_venc_cls = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_venc_cls, start_factor=1e-6/args.venc_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_venc_cls = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_venc_cls, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_venc_cls = torch.optim.lr_scheduler.SequentialLR(optimizer_venc_cls, [linear_warmup_scheduler_venc_cls, cosine_scheduler_venc_cls], milestones=[args.warmup_epochs*len(train_loader)])
    # Action encoder, view predictor, and generator share the same optimizer and scheduler
    param_groups_vpred_actenc_gen = [{'params': action_encoder.parameters(), 'lr': args.vpred_lr, 'weight_decay': args.vpred_wd},
                                    {'params': view_predictor.parameters(), 'lr': args.vpred_lr, 'weight_decay': args.vpred_wd},
                                    {'params': generator.parameters(), 'lr': args.vpred_lr, 'weight_decay': args.vpred_wd}
                                    ]
    optimizer_vpred_actenc_gen = torch.optim.AdamW(param_groups_vpred_actenc_gen, lr=0, weight_decay=0)
    linear_warmup_scheduler_vpred_actenc_gen = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_vpred_actenc_gen, start_factor=1e-6/args.vpred_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_vpred_actenc_gen = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vpred_actenc_gen, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_vpred_actenc_gen = torch.optim.lr_scheduler.SequentialLR(optimizer_vpred_actenc_gen, [linear_warmup_scheduler_vpred_actenc_gen, cosine_scheduler_vpred_actenc_gen], milestones=[args.warmup_epochs*len(train_loader)])

    ### Load criterions
    criterion_CE = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_CE_val = torch.nn.CrossEntropyLoss() # without label smoothing for validation
    criterion_MSE = torch.nn.MSELoss()

    ### Save one batch for plot purposes
    seed_everything(final_seed)  # Reset seed to ensure reproducibility for the batch
    if args.local_rank == 0:
        PLOT_N = 8   # or make this a CLI arg
        plot_indices = build_stratified_indices(val_dataset, PLOT_N)
        episodes_plot, episodes_labels = make_plot_batch(val_dataset, plot_indices, collate_function_notaskid)
        # Quickly plot the first episode to see if it is correct (not generated images, just plot directly the images)
        episode_0_imgs = episodes_plot[0][0]
        episode_0_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_0_imgs]
        episode_0_imgs = torch.stack(episode_0_imgs, dim=0)
        grid = torchvision.utils.make_grid(episode_0_imgs, nrow=args.num_views)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
        grid.save(os.path.join(args.save_dir, 'episode_0_imgs.png'))


    #### Train and Validation loop ####
    if args.local_rank == 0:
        print('\n==> Training and Validating model')
    init_time = time.time()
    scaler = GradScaler()

    for epoch in range(args.epochs):
        start_time = time.time()

        if args.local_rank == 0:
            print(f'\n==> Epoch {epoch}/{args.epochs}')

        # DDP init
        if args.ddp:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(0)


        ##################
        ### Train STEP ###
        ##################

        train_loss_CE = MetricLogger('Train Loss CE')
        train_loss_MSE_1 = MetricLogger('Train Loss MSE 1')
        train_loss_MSE_2 = MetricLogger('Train Loss MSE 2')
        # train_loss_MSE_3 = MetricLogger('Train Loss MSE 3')
        train_loss_MSE_total = MetricLogger('Train Loss MSE Total')
        train_loss_total = MetricLogger('Train Loss Total')
        train_top1 = MetricLogger('Train Top1 ACC')
        train_top5 = MetricLogger('Train Top5 ACC')

        view_encoder.train()
        classifier.train()
        action_encoder.train()
        view_predictor.train()
        generator.train()

        for i, (batch_episodes, batch_labels) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes[0].to(device, non_blocking=True) # (B, V, C, H, W)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True) # (B, V)
            batch_episodes_actions = batch_episodes[1] # (B, V, A)

            ## Forward pass
            B, V, C, H, W = batch_episodes_imgs.shape
            with (autocast(device_type='cuda', dtype=torch.float16)):
                # View Encoder forward pass
                flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                flat_clstok_and_imgfttoks = view_encoder(flat_imgs) # (B*V, 1+Timg, D)
                noflat_clstok_and_imgfttoks = flat_clstok_and_imgfttoks.view(B, V, flat_clstok_and_imgfttoks.size(1), -1) # (B, V, 1+Timg, Dimg)
                noflat_clstok = noflat_clstok_and_imgfttoks[:, :, 0:1, :] # (B, V, 1, D)
                noflat_imgfttoks = noflat_clstok_and_imgfttoks[:, :, 1:, :] # (B, V, Timg, Dimg)

                # Action Encoder forward pass
                flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)] # list length B*V
                flat_acttok = action_encoder(flat_actions) # (B*V, 1, D)
                noflat_acttok = flat_acttok.view(B, V, flat_acttok.size(1), -1) # (B, V, 1, D)

                # View Predictor forward pass (first view output is a bunch of zeros here. We are not predicting it. It is always available inside the transformer)
                noflat_PRED_imgfttoks, mask_indices = view_predictor(noflat_imgfttoks, noflat_acttok) # (B, V, Timg, Dimg), (Timg)

                # Generator + View Encoder forward pass
                noflat_PRED_imgs = generator(noflat_PRED_imgfttoks)
                flat_PRED_imgs = noflat_PRED_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                flat_PRED2_clstok_and_imgfttoks = view_encoder(flat_PRED_imgs) # (B*V, 1+Timg, D)
                noflat_PRED2_clstok_and_imgfttoks = flat_PRED2_clstok_and_imgfttoks.view(B, V, flat_PRED2_clstok_and_imgfttoks.size(1), -1) # (B, V, 1+Timg, Dimg)
                noflat_PRED2_imgfttoks = noflat_PRED2_clstok_and_imgfttoks[:, :, 1:, :] # (B, V, Timg, Dimg)

                # # Direct Generator + View Encoder forward pass
                # noflat_directPRED_imgs = generator(noflat_imgfttoks)
                # flat_directPRED_imgs = noflat_directPRED_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                # flat_directPRED2_clstok_and_imgfttoks = view_encoder(flat_directPRED_imgs) # (B*V, 1+Timg, D)
                # noflat_directPRED2_clstok_and_imgfttoks = flat_directPRED2_clstok_and_imgfttoks.view(B, V, flat_directPRED2_clstok_and_imgfttoks.size(1), -1) # (B, V, 1+Timg, Dimg)
                # noflat_directPRED2_imgfttoks = noflat_directPRED2_clstok_and_imgfttoks[:, :, 1:, :] # (B, V, Timg, Dimg)

                # MSE losses dev3-> Include view 1
                noflat_imgfttoks_detach = noflat_imgfttoks.detach()
                # Apply loss_mse_1 only to masked tokens
                loss_mse_1 = criterion_MSE(noflat_PRED_imgfttoks[:, :, mask_indices, :], noflat_imgfttoks_detach[:, :, mask_indices, :])
                # Apply loss_mse_2 to all tokens because it comes from a generated image
                loss_mse_2 = criterion_MSE(noflat_PRED2_imgfttoks, noflat_imgfttoks_detach)
                # loss_mse_3 = criterion_MSE(noflat_directPRED2_imgfttoks, noflat_imgfttoks_detach)

                # CE loss (Take view encoder cls tokens output)
                flat_clstok= noflat_clstok.reshape(B*V, noflat_clstok.size(2), -1).squeeze(1) # (B*V, D)
                flat_logits = classifier(flat_clstok)
                flat_labels = batch_episodes_labels.reshape(-1) # (B*V,)
                loss_ce = criterion_CE(flat_logits, flat_labels)
                acc1, acc5 = accuracy(flat_logits, flat_labels, topk=(1, 5))

            # Calculate Total loss for the batch
            loss_mse_total = loss_mse_1 + loss_mse_2 #+ loss_mse_3
            loss_total = loss_mse_total + loss_ce

            ## Backward pass with clip norm
            optimizer_venc_cls.zero_grad()
            optimizer_vpred_actenc_gen.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer_venc_cls)
            torch.nn.utils.clip_grad_norm_(view_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optimizer_venc_cls)
            scaler.unscale_(optimizer_vpred_actenc_gen)
            torch.nn.utils.clip_grad_norm_(action_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(view_predictor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(optimizer_vpred_actenc_gen)
            scaler.update()
            scheduler_venc_cls.step()
            scheduler_vpred_actenc_gen.step()

            ## Track training metrics per batch
            train_loss_total.update(loss_total.item(), B)
            train_loss_MSE_total.update(loss_mse_total.item(), B)
            train_loss_CE.update(loss_ce.item(), B)
            train_loss_MSE_1.update(loss_mse_1.item(), B)
            train_loss_MSE_2.update(loss_mse_2.item(), B)
            # train_loss_MSE_3.update(loss_mse_3.item(), B)
            train_top1.update(acc1.item(), B)
            train_top5.update(acc5.item(), B)

            if (args.local_rank == 0) and ((i % args.print_frequency) == 0):
                print(
                    f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                    f'lr Enc_cls: {scheduler_venc_cls.get_last_lr()[0]:.6f} -- ' +
                    f'lr Vpred_actenc_gen: {scheduler_vpred_actenc_gen.get_last_lr()[0]:.6f} -- ' +
                    f'Loss Total: {loss_total.item():.6f} -- ' +
                    f'Loss MSE Total: {loss_mse_total.item():.6f} -- ' +
                    f'Loss CE: {loss_ce.item():.6f} -- ' +
                    f'Loss MSE 1: {loss_mse_1.item():.6f} -- ' +
                    f'Loss MSE 2: {loss_mse_2.item():.6f} -- ' +
                    # f'Loss MSE 3: {loss_mse_3.item():.6f} -- ' +
                    f'Top1 ACC: {acc1.item():.3f} -- ' +
                    f'Top5 ACC: {acc5.item():.3f}'
                    )
            if args.local_rank == 0:
                writer.add_scalar('lr Enc_cls', scheduler_venc_cls.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('lr Vpred_actenc_gen', scheduler_vpred_actenc_gen.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Total', loss_total.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss MSE Total', loss_mse_total.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss CE', loss_ce.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss MSE 1', loss_mse_1.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss MSE 2', loss_mse_2.item(), epoch*len(train_loader)+i)
                # writer.add_scalar('Loss MSE 3', loss_mse_3.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Top1 ACC', acc1.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Top5 ACC', acc5.item(), epoch*len(train_loader)+i)
        
        # Track training metrics per epoch
        if args.ddp:
            train_loss_total.all_reduce()
            train_loss_MSE_total.all_reduce()
            train_loss_CE.all_reduce()
            train_loss_MSE_1.all_reduce()
            train_loss_MSE_2.all_reduce()
            # train_loss_MSE_3.all_reduce()
            train_top1.all_reduce()
            train_top5.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Total (per epoch)', train_loss_total.avg, epoch)
            writer.add_scalar('Loss MSE Total (per epoch)', train_loss_MSE_total.avg, epoch)
            writer.add_scalar('Loss CE (per epoch)', train_loss_CE.avg, epoch)
            writer.add_scalar('Loss MSE 1 (per epoch)', train_loss_MSE_1.avg, epoch)
            writer.add_scalar('Loss MSE 2 (per epoch)', train_loss_MSE_2.avg, epoch)
            # writer.add_scalar('Loss MSE 3 (per epoch)', train_loss_MSE_3.avg, epoch)
            writer.add_scalar('Top1 ACC (per epoch)', train_top1.avg, epoch)
            writer.add_scalar('Top5 ACC (per epoch)', train_top5.avg, epoch)
            print(
                f'Epoch [{epoch}] Train --> Loss Total: {train_loss_total.avg:.6f} -- '
                f'Loss MSE Total: {train_loss_MSE_total.avg:.6f} -- '
                f'Loss CE: {train_loss_CE.avg:.6f} -- '
                f'Loss MSE 1: {train_loss_MSE_1.avg:.6f} -- '
                f'Loss MSE 2: {train_loss_MSE_2.avg:.6f} -- '
                # f'Loss MSE 3: {train_loss_MSE_3.avg:.6f} -- '
                f'Top1 ACC: {train_top1.avg:.3f} -- '
                f'Top5 ACC: {train_top5.avg:.3f}')

        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes to finish the training step



        #######################
        ### Validation STEP ###
        #######################

        val_loss_CE = MetricLogger('Val Loss CE')
        val_loss_MSE_1 = MetricLogger('Val Loss MSE 1')
        val_loss_MSE_2 = MetricLogger('Val Loss MSE 2')
        # val_loss_MSE_3 = MetricLogger('Val Loss MSE 3')
        val_loss_MSE_total = MetricLogger('Val Loss MSE Total')
        val_loss_total = MetricLogger('Val Loss Total')
        val_top1 = MetricLogger('Val Top1 ACC')
        val_top5 = MetricLogger('Val Top5 ACC')

        view_encoder.eval()
        classifier.eval()
        action_encoder.eval()
        view_predictor.eval()
        generator.eval()

        with torch.no_grad():
            for j, (batch_episodes, batch_labels) in enumerate(val_loader):
                batch_episodes_imgs = batch_episodes[0].to(device, non_blocking=True)  # (B, V, C, H, W)
                batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True)  # (B, V)
                batch_episodes_actions = batch_episodes[1]  # list of lists (B,V,ops)

                B, V, C, H, W = batch_episodes_imgs.shape
                with autocast(device_type='cuda', dtype=torch.float16):
                    # View Encoder
                    flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                    flat_clstok_and_imgfttoks = view_encoder(flat_imgs)  # (B*V, 1+Timg, D)
                    noflat_clstok_and_imgfttoks = flat_clstok_and_imgfttoks.view(B, V, flat_clstok_and_imgfttoks.size(1), -1) # (B, V, 1+Timg, D)
                    noflat_clstok = noflat_clstok_and_imgfttoks[:, :, 0:1, :]          # (B,V,1,D)
                    noflat_imgfttoks = noflat_clstok_and_imgfttoks[:, :, 1:, :]        # (B,V,Timg,D)

                    # Action Encoder
                    flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # len B*V
                    flat_acttok = action_encoder(flat_actions)    # (B*V,1,D)
                    noflat_acttok = flat_acttok.view(B, V, 1, -1) # (B,V,1,D)

                    # View Predictor: predict IMG tokens + CLS tokens for each view
                    noflat_PRED_imgfttoks, mask_indices = view_predictor(noflat_imgfttoks, noflat_acttok)

                    # Generator + Encoder (re-encode predicted images)
                    noflat_PRED_imgs = generator(noflat_PRED_imgfttoks)
                    flat_PRED_imgs = noflat_PRED_imgs.reshape(B * V, C, H, W)
                    flat_PRED2 = view_encoder(flat_PRED_imgs)
                    noflat_PRED2 = flat_PRED2.view(B, V, flat_PRED2.size(1), -1)
                    noflat_PRED2_imgfttoks = noflat_PRED2[:, :, 1:, :]

                    # # Direct Generator + Encoder (teacher forcing baseline)
                    # noflat_directPRED_imgs = generator(noflat_imgfttoks)
                    # flat_directPRED_imgs = noflat_directPRED_imgs.reshape(B * V, C, H, W)
                    # flat_directPRED2_clstok_and_imgfttoks = view_encoder(flat_directPRED_imgs)
                    # noflat_directPRED2_clstok_and_imgfttoks = flat_directPRED2_clstok_and_imgfttoks.view(B, V, flat_directPRED2_clstok_and_imgfttoks.size(1), -1)
                    # noflat_directPRED2_imgfttoks = noflat_directPRED2_clstok_and_imgfttoks[:, :, 1:, :]

                    # MSEs (detach GT tokens) dev3-> Include view 1
                    noflat_imgfttoks_detach = noflat_imgfttoks #.detach() No need to detach because it is for validation with torch.no_grad()
                    # Apply loss_mse_1 only to masked tokens
                    loss_mse_1 = criterion_MSE(noflat_PRED_imgfttoks[:, :, mask_indices, :], noflat_imgfttoks_detach[:, :, mask_indices, :])
                    # Apply loss_mse_2 to all tokens because it comes from a generated image
                    loss_mse_2 = criterion_MSE(noflat_PRED2_imgfttoks, noflat_imgfttoks_detach)
                    # loss_mse_3 = criterion_MSE(noflat_directPRED2_imgfttoks, noflat_imgfttoks_detach)

                    # CE loss (Take view encoder cls tokens output)
                    flat_clstok = noflat_clstok.reshape(B*V, noflat_clstok.size(2), -1).squeeze(1) # (B*V, D)
                    flat_logits = classifier(flat_clstok)
                    flat_labels = batch_episodes_labels.reshape(-1)  # (B*V,)
                    loss_ce = criterion_CE_val(flat_logits, flat_labels)
                    acc1, acc5 = accuracy(flat_logits, flat_labels, topk=(1, 5))

                loss_mse_total = loss_mse_1 + loss_mse_2 #+ loss_mse_3
                loss_total = loss_mse_total + loss_ce

                # Match your train weighting: n = number of episodes (B)
                val_loss_total.update(loss_total.item(), B)
                val_loss_MSE_total.update(loss_mse_total.item(), B)
                val_loss_CE.update(loss_ce.item(), B)
                val_loss_MSE_1.update(loss_mse_1.item(), B)
                val_loss_MSE_2.update(loss_mse_2.item(), B)
                # val_loss_MSE_3.update(loss_mse_3.item(), B)
                val_top1.update(acc1.item(), B)
                val_top5.update(acc5.item(), B)

        # DDP aggregate
        if args.ddp:
            val_loss_total.all_reduce()
            val_loss_MSE_total.all_reduce()
            val_loss_CE.all_reduce()
            val_loss_MSE_1.all_reduce()
            val_loss_MSE_2.all_reduce()
            # val_loss_MSE_3.all_reduce()
            val_top1.all_reduce()
            val_top5.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Val Loss Total (per epoch)', val_loss_total.avg, epoch)
            writer.add_scalar('Val Loss MSE Total (per epoch)', val_loss_MSE_total.avg, epoch)
            writer.add_scalar('Val Loss CE (per epoch)', val_loss_CE.avg, epoch)
            writer.add_scalar('Val Loss MSE 1 (per epoch)', val_loss_MSE_1.avg, epoch)
            writer.add_scalar('Val Loss MSE 2 (per epoch)', val_loss_MSE_2.avg, epoch)
            # writer.add_scalar('Val Loss MSE 3 (per epoch)', val_loss_MSE_3.avg, epoch)
            writer.add_scalar('Val Top1 ACC (per epoch)', val_top1.avg, epoch)
            writer.add_scalar('Val Top5 ACC (per epoch)', val_top5.avg, epoch)
            print(
                f'Epoch [{epoch}] Val --> Loss Total: {val_loss_total.avg:.6f} -- '
                f'Loss MSE Total: {val_loss_MSE_total.avg:.6f} -- '
                f'Loss CE: {val_loss_CE.avg:.6f} -- '
                f'Loss MSE 1: {val_loss_MSE_1.avg:.6f} -- '
                f'Loss MSE 2: {val_loss_MSE_2.avg:.6f} -- '
                # f'Loss MSE 3: {val_loss_MSE_3.avg:.6f} -- '
                f'Top1 ACC: {val_top1.avg:.3f} -- '
                f'Top5 ACC: {val_top5.avg:.3f}'
            )


        ### Save model ###
        if (args.local_rank == 0) and (((epoch+1) % 10) == 0) or epoch==0:
            if args.ddp:
                view_encoder_state_dict = view_encoder.module.state_dict()
                classifier_state_dict = classifier.module.state_dict()
                action_encoder_state_dict = action_encoder.module.state_dict()
                view_predictor_state_dict = view_predictor.module.state_dict()
                generator_state_dict = generator.module.state_dict()
            else:
                view_encoder_state_dict = view_encoder.state_dict()
                classifier_state_dict = classifier.state_dict()
                action_encoder_state_dict = action_encoder.state_dict()
                view_predictor_state_dict = view_predictor.state_dict()
                generator_state_dict = generator.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(classifier_state_dict, os.path.join(args.save_dir, f'classifier_epoch{epoch}.pth'))
            torch.save(action_encoder_state_dict, os.path.join(args.save_dir, f'action_encoder_epoch{epoch}.pth'))
            torch.save(view_predictor_state_dict, os.path.join(args.save_dir, f'view_predictor_epoch{epoch}.pth'))
            torch.save(generator_state_dict, os.path.join(args.save_dir, f'generator_epoch{epoch}.pth'))

        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes to finish the validation step


        ### Plot reconstructions examples ###
        if args.local_rank == 0:
            if (epoch+1) % 1 == 0 or epoch==0:
                view_encoder.eval()
                action_encoder.eval()
                view_predictor.eval()
                generator.eval()
                classifier.eval()
                N = PLOT_N
                episodes_plot_imgs = episodes_plot[0][:N].to(device, non_blocking=True) # (N, V, C, H, W)
                episodes_plot_actions = episodes_plot[1][:N] # (N, V, A)
                _, V, C, H, W = episodes_plot_imgs.shape
                with torch.no_grad():
                    # View Encoder forward pass
                    flat_feats = view_encoder(episodes_plot_imgs.reshape(N * V, C, H, W))
                    Timg, Dimg = flat_feats.shape[1], flat_feats.shape[2]
                    noflat_feats = flat_feats.reshape(N, V, flat_feats.size(1), -1) # (N, V, 1+Timg, Dimg)
                    noflat_imgfttoks = noflat_feats[:, :, 1:, :] # (N, V, Timg, Dimg)

                    # Action Encoder forward pass
                    flat_actions = [episodes_plot_actions[b][v] for b in range(N) for v in range(V)] # list length N*V
                    flat_acttok = action_encoder(flat_actions) # (N*V, 1, D)
                    noflat_acttok = flat_acttok.view(N, V, flat_acttok.size(1), -1) # (N, V, 1, D)

                    # View Predictor forward pass
                    noflat_PRED_imgfttoks, mask_indices = view_predictor(noflat_imgfttoks, noflat_acttok) # (N, V, Timg, Dimg), (Timg)

                    # Generator forward pass
                    noflat_PRED_imgs = generator(noflat_PRED_imgfttoks) # (N, V, Timg, Dimg)
                    
                episodes_plot_gen_imgs = noflat_PRED_imgs.detach().cpu() # (N, V, Timg, Dimg)
                episodes_plot_imgs = episodes_plot_imgs.detach().cpu() # (N, V, C, H, W)
                # plot each episode
                for i in range(N):
                    episode_i_imgs = episodes_plot_imgs[i]
                    episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_imgs]
                    episode_i_imgs = torch.stack(episode_i_imgs, dim=0) # (V, C, H, W)

                    episode_i_gen_imgs = episodes_plot_gen_imgs[i]
                    episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_gen_imgs]
                    episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0) # (V, C, H, W)
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

    # Close tensorboard writer
    if args.local_rank == 0:
        writer.close()

    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish the validation step
        torch.distributed.destroy_process_group()  # Destroy the process group

    return None

if __name__ == '__main__':
    main()