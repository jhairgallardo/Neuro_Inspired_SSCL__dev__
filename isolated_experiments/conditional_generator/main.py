import argparse
import os
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import time
import wandb

from resnet_cond_film import resnet18, ResNet18Dec, FeatureTransModelNew

import einops
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from action_code import aug_with_action_code, ActionCodeDataset


parser = argparse.ArgumentParser(description='Conditional Generator Training')
parser.add_argument('--name', type=str, default=None, help='Experiment name')
parser.add_argument('--data', default='/home/jchen175/scratch/dataset/imagenet10/', help='path to dataset')
parser.add_argument('--ckpt_file', type=str, default='best_ckpt.pth', help='checkpoint file name; used for saving')
parser.add_argument('--save_dir', type=str, default='./featCondGenAllActionsNas_ckpts', help='directory to save checkpoints')

parser.add_argument('-j', '--workers', default=48, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--image_size', type=int, default=224) # resolution 32,64,128,224
parser.add_argument('--warmup_epochs', type=int, default=5)

## DDP args
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)


parser.add_argument('--ckpt_dir', type=str, default='/scratch/jchen175/projects/nips/Standard_Generator_Code/SSL_100epochs100stop_12views_0.02lr_128bs_seed0/encoder_epoch99.pth', help='Path to the encoder checkpoint')
parser.add_argument('--noScheduler', action='store_true', help="Use coord convs")
parser.add_argument('--action_channels', type=int, default=64, help="feature transfer model: mlp out channels")
parser.add_argument('--num_layer', type=int, default=4, help="Number of layers (blocks) in feature transfer model")
parser.add_argument('--num_blocks', type=int, default=3, help="Number of blocks in the decoder")



def main():
    start_time = time.time()
    args = parser.parse_args()

    ## DDP init & exp logging
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')


    exp_name = (f'lr_{args.lr}_bs_{args.batch_size}_epochs_{args.epochs}_ActChannels_{args.action_channels}_numLayer_'
                f'{args.num_layer}_numBlocks_{args.num_blocks}_num_blocks_{args.num_blocks}')

    if args.name is not None:
        exp_name = args.name + '_' + exp_name
    exp_name = 'ALLACT_' + exp_name

    if args.local_rank == 0:
        # experiment = wandb.init(name=exp_name, project='FeatCondGenAllActions_1007')
        experiment = wandb.init(name=exp_name, project='FeatCondGenNas')
        wandb.config.update(args)
        print(f"Experiment name: {exp_name}")

    else:
        experiment = None

    args.save_dir = os.path.join(args.save_dir, exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)



    ## Define View Encoder Arch
    encoder = resnet18()
    ## Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    ssl_enc_ckpt = args.ckpt_dir
    enc_checkpoint = torch.load(ssl_enc_ckpt)
    # The checkpoint does not have the fc layer. Assert that the missing keys only are ['fc.weight', 'fc.bias'] and unexpected keys are [].
    missing_keys, unexpected_keys = encoder.load_state_dict(enc_checkpoint, strict=False)
    # assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    assert missing_keys == [] and unexpected_keys == []
    print("\nSuccessfully loaded pre-trained view encoder")


    ## Decoder (ResNet)
    num_blocks = [args.num_blocks, args.num_blocks, args.num_blocks, args.num_blocks]
    decoder = ResNet18Dec(num_Blocks=num_blocks)

    ## Feature Transfer Model
    feature = FeatureTransModelNew(action_dim=16, action_channels=args.action_channels, layer=args.num_layer)


    if args.local_rank == 0:
        print(f"encoder total parameters: {sum(p.numel() for p in encoder.parameters())/1e6}M")
        print(f"featureTans total parameters: {sum(p.numel() for p in feature.parameters())/1e6}M")
        print(f"decoder total parameters: {sum(p.numel() for p in decoder.parameters())/1e6}M")
        experiment.config.update({'encoder_total_params': sum(p.numel() for p in encoder.parameters())/1e6,
                                  'feature_total_params': sum(p.numel() for p in feature.parameters())/1e6,
                                  'decoder_total_params': sum(p.numel() for p in decoder.parameters())/1e6})


    # DDP init
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    feature = feature.to(device)

    decoder = nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
    feature = nn.SyncBatchNorm.convert_sync_batchnorm(feature)
    feature = torch.nn.parallel.DistributedDataParallel(feature, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)

    criterion = nn.MSELoss()

    # /// Optimizer /// #
    ### AdamW ###
    optimizer = torch.optim.AdamW(list(feature.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-2)


    # /// LR Schedule /// #
    ### Cosine w/ Warmup ###
    if not args.noScheduler:
        warmup_epochs=args.warmup_epochs
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                 T_max=args.epochs - warmup_epochs, eta_min=0.0)
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs])
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=1)

    print('\nloading the ImageNet dataset...') # ImageNet-100
    ## Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')


    size = int((256 / 224) * args.image_size)
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(args.image_size),
        ])

    print("\nUsing pytorch image augmentations")

    ## Data Augmentation
    action_generator = aug_with_action_code(size=args.image_size)

    train_dataset = ActionCodeDataset(traindir, action_generator, preprocess)
    val_dataset = ActionCodeDataset(valdir, action_generator, preprocess)

    ##DDP Sampler
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    print('\nStarting training...')
    # scaler for mixed precision
    scaler = GradScaler()

    for epoch in range(1, args.epochs+1):
        ## train for one epoch
        # DDP init
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        if epoch == 1:
            validate_recon(val_loader, encoder, decoder, feature, criterion, args, experiment, device)

        log_flag = True if epoch % 5 == 0 else False # log images every 5 epochs

        train_one_epoch(train_loader, encoder, decoder, feature, criterion, optimizer, epoch, args, experiment, device, scaler, log_flag)
        lr_scheduler.step()

        if epoch % 10 == 0:
            validate_recon(val_loader, encoder, decoder, feature, criterion, args, experiment, device)
        dist.barrier()



    ckpt_path = args.save_dir
    torch.save({
        'decoder_state_dict': decoder.state_dict(),
        'feature_state_dict': feature.state_dict(),
        'optimizer': optimizer.state_dict(),
    },
        f='./' + ckpt_path + '/{}'.format(args.ckpt_file)
    )

    print(f"Total time taken: {(time.time() - start_time)/60} mins")


### Training Function ###
def train_one_epoch(train_loader, encoder, decoder, feature, criterion, optimizer, epoch, args, exp, device, scaler,log_visual_flag=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    decoder.train()
    feature.train()
    start = time.time()

    for i, (img, trans_img, action_code, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        with autocast():
            z = encoder(img.to(device))
            z_t = encoder(trans_img.to(device))

            z_t_prime = feature(z, action_code.to(device))

            # feature transfer loss: compare transformed feature with original feature
            loss_1 = criterion(z_t_prime, z_t)

            # reconstruction
            r = decoder(z)
            r_t = decoder(z_t_prime)

            z_prime = encoder(r)
            # reconstruction loss for original image: compare reconstructed image feature with original feature (un augmented)
            loss_2 = criterion(z_prime, z)

            z_t_pp = encoder(r_t)
            # reconstruction loss for transformed image: compare reconstructed image feature with original feature (augmented)
            loss_3 = criterion(z_t_pp, z_t)

            # total loss
            loss = loss_1 + loss_2 + loss_3


        losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        if args.local_rank == 0:
            logged_dict = {'train_total_loss': loss.item(),
                           'train_feat_trans_loss': loss_1.item(),
                           'train_recon_orig_feat_loss': loss_2.item(),
                           'train_recon_trans_feat_loss': loss_3.item(),
                           'epoch': epoch,
                           'lr': optimizer.param_groups[0]['lr'],
                           }



            if i == 0 and log_visual_flag:
                image_first_batch = img.cpu()[:24]
                recon_first_batch = r.cpu()[:24]
                image_crop_first_batch = trans_img.cpu()[:24]
                recon_crop_first_batch = r_t.cpu()[:24]
                visualization = torch.stack([image_first_batch, recon_first_batch, image_crop_first_batch, recon_crop_first_batch], dim=0)
                visualization = einops.rearrange(visualization, 'a (n_h n_w) c h w -> c (n_h a h) (n_w w)', n_h=3, n_w=8)
                logged_dict['train reconstructions'] = wandb.Image(visualization)


                with torch.no_grad():
                    train_pixel_loss = criterion(r, img.to(device))
                    train_trans_pixel_loss = criterion(r_t, trans_img.to(device))
                logged_dict['train_pixel_loss'] = train_pixel_loss.item()
                logged_dict['train_trans_pixel_loss'] = train_trans_pixel_loss.item()

            exp.log(logged_dict)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    return


def validate_recon(val_loader, encoder, decoder, feature, criterion, args, exp, device):
    batch_time = AverageMeter()
    losses = torch.tensor(0.0).to(device)
    feature_losses = torch.tensor(0.0).to(device)
    samples = torch.tensor(0.0).to(device)

    encoder.eval()
    decoder.eval()
    feature.eval()
    if args.local_rank == 0:
        log_dict = {}

    with torch.no_grad():
        start = time.time()
        for i, (img, trans_img, action_code, label) in enumerate(val_loader):
            samples += img.size(0)

            z = encoder(img.to(device))
            z_t = encoder(trans_img.to(device))

            z_t_prime = feature(z, action_code.to(device))

            # feature transfer loss
            # loss_1 = criterion(z_t_prime, z_t)

            # reconstruction
            r = decoder(z)
            r_t = decoder(z_t_prime)

            # z_prime = encoder(r)
            # reconstruction loss for original image
#             loss_2 = criterion(z_prime, z)

            z_t_pp = encoder(r_t)
            # reconstruction loss for transformed image
            loss_3 = criterion(z_t_pp, z_t)

            loss = criterion(r_t, trans_img.to(device))  ## MSE loss
            losses += loss.item()

            feature_losses += loss_3.item()

            batch_time.update(time.time() - start)
            start = time.time()
            if i == 0 and args.local_rank == 0:
                image_first_batch = img.cpu()[:24]
                recon_first_batch = r.cpu()[:24]
                image_crop_first_batch = trans_img.cpu()[:24]
                recon_trans_first_batch = r_t.cpu()[:24]
                visualization = torch.stack([image_first_batch, recon_first_batch, image_crop_first_batch, recon_trans_first_batch], dim=0)
                visualization = einops.rearrange(visualization, 'a (n_h n_w) c h w -> c (n_h a h) (n_w w)', n_h=3,
                                                 n_w=8)
                log_dict['val reconstructions'] = [wandb.Image(visualization, caption="Reconstructions")]

    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)
    dist.reduce(feature_losses, 0, op=dist.ReduceOp.SUM)
    dist.reduce(samples, 0, op=dist.ReduceOp.SUM)


    if args.local_rank == 0:
        log_dict['val_pixel_loss'] = losses.item() / samples.item()
        log_dict['val_feature_loss'] = feature_losses.item() / samples.item()
        exp.log(log_dict)
        print('Validation Loss: ', losses.item() / samples.item())

    return




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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



if __name__ == '__main__':
    main()
