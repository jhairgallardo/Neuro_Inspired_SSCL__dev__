import argparse
import os, time

import torch
import torchvision
from torchvision.datasets import ImageFolder

from dev_models_deit3 import *
from augs_episodes_firstbigcontext import Episode_Transformations, collate_function_notaskid, DeterministicEpisodes, ImageFolderDetEpisodes
from utils import MetricLogger, reduce_tensor, accuracy, time_duration_print, build_stratified_indices, make_plot_batch

import numpy as np
import json
import random
from PIL import Image

from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/caltech256/256_ObjectCategories_splits')
parser.add_argument('--val_episode_seed', type=int, default=12345)
parser.add_argument('--num_classes', type=int, default=256)
parser.add_argument('--mean', type=list, default=[0.5, 0.5, 0.5])
parser.add_argument('--std', type=list, default=[0.5, 0.5, 0.5])
# View encoder parameters
parser.add_argument('--view_enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--venc_lr', type=float, default=0.0008)
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
parser.add_argument('--vpred_lr', type=float, default=0.0008)
parser.add_argument('--vpred_wd', type=float, default=0)
parser.add_argument('--vpred_dim', type=int, default=192)
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
parser.add_argument('--episode_batch_size_per_gpu', type=int, default=20)
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--label_smoothing', type=float, default=0.0)
# Other parameters
parser.add_argument('--workers_per_gpu', type=int, default=8)
parser.add_argument('--save_dir', type=str, default="output/Pretrained_caltech256_dev/run_debug")
parser.add_argument('--print_frequency', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)


def main():

    ### Parse arguments
    args = parser.parse_args()

    ### Create save dir folder
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    ### Define loggers
    tb_logger = TensorBoardLogger(root_dir=os.path.join(args.save_dir, "logs"), name="tb_logs")
    csv_logger = CSVLogger(root_dir=os.path.join(args.save_dir, "logs"), name="csv_logs",  flush_logs_every_n_steps=1)

    ### Define Fabric and launch it
    fabric = Fabric(accelerator="gpu", strategy="ddp", devices="auto", precision="bf16-mixed", loggers=[tb_logger, csv_logger])
    fabric.launch()

    ### Seed everything
    fabric.seed_everything(args.seed)

    ### Print args
    fabric.print(args)

    ### Save args
    if fabric.is_global_zero:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    ### Seed everything
    fabric.seed_everything(args.seed)

    ### Load Training data
    fabric.print('\n==> Preparing Training data...')
    traindir = os.path.join(args.data_path, 'train')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
    train_dataset = ImageFolder(traindir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True,
                                               num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True,
                                               collate_fn=collate_function_notaskid)
    train_loader = fabric.setup_dataloaders(train_loader)

    ### Load Validation data
    fabric.print('\n==> Preparing Validation data...')
    valdir = os.path.join(args.data_path, 'val')
    val_base_transform = Episode_Transformations(num_views=args.num_views, mean=args.mean, std=args.std)
    val_transform = DeterministicEpisodes(val_base_transform, base_seed=args.val_episode_seed)
    val_dataset = ImageFolderDetEpisodes(valdir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                             num_workers=args.workers_per_gpu, pin_memory=True,
                                             collate_fn=collate_function_notaskid)
    val_loader = fabric.setup_dataloaders(val_loader)

    ### Define models
    fabric.print('\n==> Prepare models...')
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
    fabric.print('\nView encoder')
    fabric.print(view_encoder)
    fabric.print('\nClassifier')
    fabric.print(classifier)
    fabric.print('\nAction encoder')
    fabric.print(action_encoder)
    fabric.print('\nView predictor')
    fabric.print(view_predictor)
    fabric.print('\nGenerator')
    fabric.print(generator)
    fabric.print('\n')

    ### Setup models
    view_encoder = fabric.setup_module(view_encoder)
    classifier = fabric.setup_module(classifier)
    action_encoder = fabric.setup_module(action_encoder)
    view_predictor = fabric.setup_module(view_predictor)
    generator = fabric.setup_module(generator)

    ### Define optimizers
    # View encoder and classifier share the same optimizer and scheduler
    param_groups_venc_cls = [{'params': view_encoder.parameters(), 'lr': args.venc_lr, 'weight_decay': args.venc_wd},
                            {'params': classifier.parameters(), 'lr': args.venc_lr, 'weight_decay': args.venc_wd}]
    optimizer_venc_cls = torch.optim.AdamW(param_groups_venc_cls, lr=0, weight_decay=0)
    # Action encoder, view predictor, and generator share the same optimizer and scheduler
    param_groups_vpred_actenc_gen = [{'params': action_encoder.parameters(), 'lr': args.vpred_lr, 'weight_decay': args.vpred_wd},
                                    {'params': view_predictor.parameters(), 'lr': args.vpred_lr, 'weight_decay': args.vpred_wd},
                                    {'params': generator.parameters(), 'lr': args.vpred_lr, 'weight_decay': args.vpred_wd}]
    optimizer_vpred_actenc_gen = torch.optim.AdamW(param_groups_vpred_actenc_gen, lr=0, weight_decay=0)
    
    ### Setup optimizers
    optimizer_venc_cls = fabric.setup_optimizers(optimizer_venc_cls)
    optimizer_vpred_actenc_gen = fabric.setup_optimizers(optimizer_vpred_actenc_gen)

    ### Define schedulers
    linear_warmup_scheduler_venc_cls = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_venc_cls, start_factor=1e-6/args.venc_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_venc_cls = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_venc_cls, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_venc_cls = torch.optim.lr_scheduler.SequentialLR(optimizer_venc_cls, [linear_warmup_scheduler_venc_cls, cosine_scheduler_venc_cls], milestones=[args.warmup_epochs*len(train_loader)])
    linear_warmup_scheduler_vpred_actenc_gen = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_vpred_actenc_gen, start_factor=1e-6/args.vpred_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_vpred_actenc_gen = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vpred_actenc_gen, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_vpred_actenc_gen = torch.optim.lr_scheduler.SequentialLR(optimizer_vpred_actenc_gen, [linear_warmup_scheduler_vpred_actenc_gen, cosine_scheduler_vpred_actenc_gen], milestones=[args.warmup_epochs*len(train_loader)])

    ### Define criterions
    criterion_CE = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_CE_val = torch.nn.CrossEntropyLoss() # without label smoothing for validation
    criterion_MSE = torch.nn.MSELoss()

    ### Save one batch for plot purposes
    fabric.seed_everything(args.seed)  # Reset seed to ensure reproducibility for the plot batch
    if fabric.is_global_zero:
        PLOT_N = 8
        plot_indices = build_stratified_indices(val_dataset, PLOT_N)
        episodes_plot, _ = make_plot_batch(val_dataset, plot_indices, collate_function_notaskid)
        # Quickly plot the first episode to see if it is correct (not generated images, just plot directly the images)
        episode_0_imgs = episodes_plot[0][0]
        episode_0_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_0_imgs]
        episode_0_imgs = torch.stack(episode_0_imgs, dim=0)
        grid = torchvision.utils.make_grid(episode_0_imgs, nrow=args.num_views)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
        grid.save(os.path.join(args.save_dir, 'episode_0_imgs.png'))
    fabric.barrier()


    #### Train and Validation loop ####
    fabric.print('\n==> Training and Validating model')
    init_time = time.time()

    for epoch in range(args.epochs):
        start_time = time.time()
        fabric.print(f'\n==> Epoch {epoch}/{args.epochs}')

        ##################
        ### Train STEP ###
        ##################

        train_loss_CE = MetricLogger('Train Loss CE')
        train_loss_MSE_1 = MetricLogger('Train Loss MSE 1')
        train_loss_MSE_2 = MetricLogger('Train Loss MSE 2')
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
            batch_episodes_imgs = batch_episodes[0] # (B, V, C, H, W)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)) # (B, V)
            batch_episodes_actions = batch_episodes[1] # (B, V, A)
            B, V, C, H, W = batch_episodes_imgs.shape

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

            with fabric.autocast(): # Run losses calculations in mixed precision (models already run in mixed precision)
                # Reconstruction losses in latent space
                noflat_imgfttoks_detach = noflat_imgfttoks.detach()
                loss_mse_1 = criterion_MSE(noflat_PRED_imgfttoks[:, :, mask_indices, :], noflat_imgfttoks_detach[:, :, mask_indices, :])
                loss_mse_2 = criterion_MSE(noflat_PRED2_imgfttoks, noflat_imgfttoks_detach)

                # CE loss
                flat_clstok= noflat_clstok.reshape(B*V, noflat_clstok.size(2), -1).squeeze(1) # (B*V, D)
                flat_logits = classifier(flat_clstok)
                flat_labels = batch_episodes_labels.reshape(-1) # (B*V,)
                loss_ce = criterion_CE(flat_logits, flat_labels)

                # Calculate Total loss for the batch
                loss_mse_total = loss_mse_1 + loss_mse_2
                loss_total = loss_mse_total + loss_ce

                # Classification accuracy
                acc1, acc5 = accuracy(flat_logits, flat_labels, topk=(1, 5))

            ## Backward pass with clip norm
            optimizer_venc_cls.zero_grad()
            optimizer_vpred_actenc_gen.zero_grad()
            fabric.backward(loss_total)
            fabric.clip_gradients(view_encoder, optimizer_venc_cls, max_norm=1.0)
            fabric.clip_gradients(classifier, optimizer_venc_cls, max_norm=1.0)
            fabric.clip_gradients(action_encoder, optimizer_vpred_actenc_gen, max_norm=1.0)
            fabric.clip_gradients(view_predictor, optimizer_vpred_actenc_gen, max_norm=1.0)
            fabric.clip_gradients(generator, optimizer_vpred_actenc_gen, max_norm=1.0)
            optimizer_venc_cls.step()
            optimizer_vpred_actenc_gen.step()

            # Update schedulers
            scheduler_venc_cls.step()
            scheduler_vpred_actenc_gen.step()

            ## Track metrics
            train_loss_total.update(fabric.all_reduce(loss_total.detach(), reduce_op="mean").item(), B)
            train_loss_MSE_total.update(fabric.all_reduce(loss_mse_total.detach(), reduce_op="mean").item(), B)
            train_loss_CE.update(fabric.all_reduce(loss_ce.detach(), reduce_op="mean").item(), B)
            train_loss_MSE_1.update(fabric.all_reduce(loss_mse_1.detach(), reduce_op="mean").item(), B)
            train_loss_MSE_2.update(fabric.all_reduce(loss_mse_2.detach(), reduce_op="mean").item(), B)
            train_top1.update(fabric.all_reduce(acc1.detach(), reduce_op="mean").item(), B)
            train_top5.update(fabric.all_reduce(acc5.detach(), reduce_op="mean").item(), B)

            ## Log and print training metrics per batch
            if fabric.is_global_zero and ((i % args.print_frequency) == 0):
                fabric.log(name=f'lr Enc_cls', value=scheduler_venc_cls.get_last_lr()[0], step=epoch*len(train_loader)+i)
                fabric.log(name=f'lr Vpred_actenc_gen', value=scheduler_vpred_actenc_gen.get_last_lr()[0], step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss Total', value=train_loss_total.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss MSE Total', value=train_loss_MSE_total.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss CE', value=train_loss_CE.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss MSE 1', value=train_loss_MSE_1.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss MSE 2', value=train_loss_MSE_2.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Top1 ACC', value=train_top1.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Top5 ACC', value=train_top5.val, step=epoch*len(train_loader)+i)
                fabric.print(
                    f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                    f'lr Enc_cls: {scheduler_venc_cls.get_last_lr()[0]:.6f} -- ' +
                    f'lr Vpred_actenc_gen: {scheduler_vpred_actenc_gen.get_last_lr()[0]:.6f} -- ' +
                    f'Loss Total: {train_loss_total.val:.6f} -- '
                    f'Loss MSE Total: {train_loss_MSE_total.val:.6f} -- '
                    f'Loss CE: {train_loss_CE.val:.6f} -- '
                    f'Loss MSE 1: {train_loss_MSE_1.val:.6f} -- '
                    f'Loss MSE 2: {train_loss_MSE_2.val:.6f} -- '
                    f'Top1 ACC: {train_top1.val:.3f} -- '
                    f'Top5 ACC: {train_top5.val:.3f}'
                    )

        ## Log and print training metrics per epoch
        fabric.log(name=f'Loss Total (per epoch)', value=train_loss_total.avg, step=epoch)
        fabric.log(name=f'Loss MSE Total (per epoch)', value=train_loss_MSE_total.avg, step=epoch)
        fabric.log(name=f'Loss CE (per epoch)', value=train_loss_CE.avg, step=epoch)
        fabric.log(name=f'Loss MSE 1 (per epoch)', value=train_loss_MSE_1.avg, step=epoch)
        fabric.log(name=f'Loss MSE 2 (per epoch)', value=train_loss_MSE_2.avg, step=epoch)
        fabric.log(name=f'Top1 ACC (per epoch)', value=train_top1.avg, step=epoch)
        fabric.log(name=f'Top5 ACC (per epoch)', value=train_top5.avg, step=epoch)
        fabric.print(
            f'Epoch [{epoch}] Train --> Loss Total: {train_loss_total.avg:.6f} -- '
            f'Loss MSE Total: {train_loss_MSE_total.avg:.6f} -- '
            f'Loss CE: {train_loss_CE.avg:.6f} -- '
            f'Loss MSE 1: {train_loss_MSE_1.avg:.6f} -- '
            f'Loss MSE 2: {train_loss_MSE_2.avg:.6f} -- '
            f'Top1 ACC: {train_top1.avg:.3f} -- '
            f'Top5 ACC: {train_top5.avg:.3f}')

        ## Wait for all processes to finish the training step
        fabric.barrier()  # Wait for all processes to finish the training step



        #######################
        ### Validation STEP ###
        #######################

        val_loss_CE = MetricLogger('Val Loss CE')
        val_loss_MSE_1 = MetricLogger('Val Loss MSE 1')
        val_loss_MSE_2 = MetricLogger('Val Loss MSE 2')
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
                batch_episodes_imgs = batch_episodes[0] # (B, V, C, H, W)
                batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)) # (B, V)
                batch_episodes_actions = batch_episodes[1]  # list of lists (B,V,ops)
                B, V, C, H, W = batch_episodes_imgs.shape

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

                with fabric.autocast(): # Run losses calculations in mixed precision (models already run in mixed precision)
                    # Reconstruction losses in latent space
                    noflat_imgfttoks_detach = noflat_imgfttoks #.detach() No need to detach because it is for validation with torch.no_grad()
                    loss_mse_1 = criterion_MSE(noflat_PRED_imgfttoks[:, :, mask_indices, :], noflat_imgfttoks_detach[:, :, mask_indices, :])
                    loss_mse_2 = criterion_MSE(noflat_PRED2_imgfttoks, noflat_imgfttoks_detach)

                    # CE loss
                    flat_clstok= noflat_clstok.reshape(B*V, noflat_clstok.size(2), -1).squeeze(1) # (B*V, D)
                    flat_logits = classifier(flat_clstok)
                    flat_labels = batch_episodes_labels.reshape(-1) # (B*V,)
                    loss_ce = criterion_CE_val(flat_logits, flat_labels)

                    # Calculate Total loss for the batch
                    loss_mse_total = loss_mse_1 + loss_mse_2
                    loss_total = loss_mse_total + loss_ce

                    # Classification accuracy
                    acc1, acc5 = accuracy(flat_logits, flat_labels, topk=(1, 5))

                ## Track metrics
                val_loss_total.update(fabric.all_reduce(loss_total.detach(), reduce_op="mean").item(), B)
                val_loss_MSE_total.update(fabric.all_reduce(loss_mse_total.detach(), reduce_op="mean").item(), B)
                val_loss_CE.update(fabric.all_reduce(loss_ce.detach(), reduce_op="mean").item(), B)
                val_loss_MSE_1.update(fabric.all_reduce(loss_mse_1.detach(), reduce_op="mean").item(), B)
                val_loss_MSE_2.update(fabric.all_reduce(loss_mse_2.detach(), reduce_op="mean").item(), B)
                val_top1.update(fabric.all_reduce(acc1.detach(), reduce_op="mean").item(), B)
                val_top5.update(fabric.all_reduce(acc5.detach(), reduce_op="mean").item(), B)

        ## Log and print validation metrics per epoch
        fabric.log(name=f'Val Loss Total (per epoch)', value=val_loss_total.avg, step=epoch)
        fabric.log(name=f'Val Loss MSE Total (per epoch)', value=val_loss_MSE_total.avg, step=epoch)
        fabric.log(name=f'Val Loss CE (per epoch)', value=val_loss_CE.avg, step=epoch)
        fabric.log(name=f'Val Loss MSE 1 (per epoch)', value=val_loss_MSE_1.avg, step=epoch)
        fabric.log(name=f'Val Loss MSE 2 (per epoch)', value=val_loss_MSE_2.avg, step=epoch)
        fabric.log(name=f'Val Top1 ACC (per epoch)', value=val_top1.avg, step=epoch)
        fabric.log(name=f'Val Top5 ACC (per epoch)', value=val_top5.avg, step=epoch)
        fabric.print(
                f'Epoch [{epoch}] Val --> Loss Total: {val_loss_total.avg:.6f} -- '
                f'Loss MSE Total: {val_loss_MSE_total.avg:.6f} -- '
                f'Loss CE: {val_loss_CE.avg:.6f} -- '
                f'Loss MSE 1: {val_loss_MSE_1.avg:.6f} -- '
                f'Loss MSE 2: {val_loss_MSE_2.avg:.6f} -- '
                f'Top1 ACC: {val_top1.avg:.3f} -- '
                f'Top5 ACC: {val_top5.avg:.3f}'
            )

        ## Wait for all processes to finish the validation step
        fabric.barrier()  # Wait for all processes to finish the validation step

        ### Save models ###
        if (((epoch+1) % 10) == 0) or epoch==0: 
            state={"view_encoder": view_encoder, "classifier": classifier, "action_encoder": action_encoder, "view_predictor": view_predictor, "generator": generator}
            fabric.save(os.path.join(args.save_dir, f'models_checkpoint_epoch{epoch}.pth'), state=state)

        ## Wait for all processes to finish the save models step
        fabric.barrier()  # Wait for all processes to finish the save models step


        ### Plot reconstructions examples ###
        if fabric.is_global_zero:
            if (epoch+1) % 1 == 0 or epoch==0:
                view_encoder.eval()
                action_encoder.eval()
                view_predictor.eval()
                generator.eval()
                classifier.eval()
                N = PLOT_N
                episodes_plot_imgs = episodes_plot[0][:N] # (N, V, C, H, W)
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

        epoch_time = time.time() - start_time
        elapsed_time = time.time() - init_time
        fabric.print(f"Epoch [{epoch}] Epoch Time: {time_duration_print(epoch_time)} -- Elapsed Time: {time_duration_print(elapsed_time)}")

    return None

if __name__ == '__main__':
    main()