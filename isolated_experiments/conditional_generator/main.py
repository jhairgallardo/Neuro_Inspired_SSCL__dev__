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

from model import resnet18, ResNet18Dec, SeparatedFeatureTransModel, FeatureTransModelNew, ActionConditionedTransformer # , ActionConditionedTransformerFlash
import einops

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from action_code import aug_with_action_code, ActionCodeDataset



### This script trains DNN on ImageNet-100 (randomly sampled 100 classes from ImageNet-1K proposed by CMC paper)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--name', type=str, default=None)  # name of the experiment
parser.add_argument('--data', default='/home/jchen175/scratch/dataset/imagenet10/', help='path to dataset')
parser.add_argument('--ckpt_file', type=str, default='best_ckpt.pth')
parser.add_argument('--save_dir', type=str, default='./featCondGenAllActionsNas1028_ckpts')
parser.add_argument('-j', '--workers', default=48, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch_size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel') #256
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--image_size', type=int, default=224) # resolution 32,64,128,224
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--ckpt_dir', type=str, default='/scratch/jchen175/projects/nips/Standard_Generator_Code/SSL_100epochs100stop_12views_0.02lr_128bs_seed0/encoder_epoch99.pth')

## DDP args
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

# debug args; useless for now
parser.add_argument('--debug', action='store_true', help='Debugging flag')
parser.add_argument('--no_crop', action='store_true')
parser.add_argument('--no_film', action='store_true')
parser.add_argument('--pixel_loss', action='store_true')
parser.add_argument('--film_scale', type=float, default=1e0)

# coord; useless for now
parser.add_argument('--film_type', type=str, help="mlp/linear")
parser.add_argument('--coord', action='store_true', help="Use coord convs")
parser.add_argument('--noScheduler', action='store_true', help="Use coord convs")
parser.add_argument('--first_film_only', action='store_true', help="apply film only to the first block")
# parser.add_argument('--full_dim_film', action='store_true', help="Use full dim film")
# used to load the pretrained feature_transfer model and decoder model; useless for now
parser.add_argument('--load', action='store_true')
parser.add_argument('--feature_model_path', default="/home/jchen175/scratch/projects/nips/Standard_Generator_Code/supervised_dnn/resnet_deeper_3_lr_0.001_bs_1024_epochs_200/best_ckpt.pth")
parser.add_argument('--decoder_model_path', default="/home/jchen175/scratch/projects/nips/Standard_Generator_Code/unconditional_gen_ckpts/unconditional_1_lr_0.001_bs_512_epochs_200/best_ckpt.pth")

# decoder args
parser.add_argument('--num_blocks', type=int, default=3, help="Number of blocks in the decoder")

# feature transfer model args (resnet-based)
parser.add_argument('--action_channels', type=int, default=32, help="feature transfer model channels")
parser.add_argument('--num_layer', type=int, default=4, help="Number of layer in feature transfer model")
parser.add_argument('--no_action_code', action='store_true', help='reconstruction of the unaug is based on the original feature; if True, replace it with reconstructed feature w/ dummy action code')
parser.add_argument('--trans_loss', action='store_true', help='unconditional reconstruction of the augmented image')
parser.add_argument('--prob_orig_view', type=float, default=1.0, help='probability of using the original view for reconstruction')

# model arch
parser.add_argument('--arch', type=str, default='res', choices=['res', 'sep', 'vit'], help=""
                                                                                           "res: resnet-like, "
                                                                                           "sep: separated feature transformer,"
                                                                                           "vit: transformer encoder")
# feature transfer model args (transformer-based)
parser.add_argument('--nhead', type=int, default=8, help="Number of heads in transformer")
parser.add_argument('--dim_feedforward', type=int, default=2048, help="Feedforward dimension in transformer")

# color jitter order
parser.add_argument('--fixed_color_jitter_order', action='store_true', help='fixed color jitter order for all images')
# random flip
parser.add_argument('--random_flip', action='store_true', help='apply random flip')
# use flash attention
parser.add_argument('--flash_att', action='store_true', help='apply flash attention')





def no_action_code(fixed_color_jitter_order=False):
    if not fixed_color_jitter_order:
        action_code = [0] + [0, 0, 1, 1] + [1., 1., 1., 0.] + torch.randperm(4).tolist() + [0, 0, 0]
    else:
        action_code = [0] + [0, 0, 1, 1] + [1., 1., 1., 0.] + [0, 0, 0]
    action_code = torch.tensor(action_code, dtype=torch.float32)
    return action_code


def load_model(model, model_path, k):
    checkpoint = torch.load(model_path,map_location='cpu')
    state_dict = {}
    for k, v in checkpoint[k].items():
        state_dict[k.replace('module.', '')] = v
    model.load_state_dict(state_dict)
    return model


def main():
    start_time = time.time()
    args = parser.parse_args()

    ## DDP init
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    model_key = ''
    if args.arch == 'res':
        model_key = f'res_ActChannels@{args.action_channels}_numLayer@{args.num_layer}_'
    elif args.arch == 'sep':
        model_key = f'sep_ActChannels@{args.action_channels}_numLayer@{args.num_layer}_'
    elif args.arch == 'vit':
        if not args.flash_att:
            model_key = f'vit_nhead@{args.nhead}_dimFeed@{args.dim_feedforward}_numLayer@{args.num_layer}_'
        else:
            model_key = f'flashVit_nhead@{args.nhead}_dimFeed@{args.dim_feedforward}_numLayer@{args.num_layer}_'

    exp_name = (f'noActionCode@{args.no_action_code}_transLoss@{args.trans_loss}_probAncView@{args.prob_orig_view}_'
                f'lr@{args.lr}_bs@{args.batch_size}_'
                f'epochs@{args.epochs}_decBlocks_{args.num_blocks}'+'_'+model_key)

    if args.name is not None:
        exp_name = args.name + '_' + exp_name
    exp_name = 'ALLACT_' + exp_name

    if args.local_rank == 0:
        # experiment = wandb.init(name=exp_name, project='FeatCondGenAllActions_1007')
        experiment = wandb.init(name=exp_name, project='FeatCondGenFixedColorJitter')
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
    #assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    assert missing_keys == [] and unexpected_keys == []
    print("\nSuccessfully loaded pre-trained view encoder")


    ## Decoder (ResNet-10)
    num_blocks = [args.num_blocks, args.num_blocks, args.num_blocks, args.num_blocks]
    decoder = ResNet18Dec(num_Blocks=num_blocks)

    ## Feature Transfer Model

    if args.arch == 'res':
        feature = FeatureTransModelNew(action_dim=16 if not args.fixed_color_jitter_order else 12,
                                       action_channels=args.action_channels, layer=args.num_layer)
    elif args.arch == 'sep':
        feature = SeparatedFeatureTransModel(action_dims=[1, 4, 8, 1, 1, 1] if not args.fixed_color_jitter_order else [1, 4, 4, 1, 1, 1],
            action_channels=args.action_channels, layer=args.num_layer)
    elif args.arch == 'vit':
        feature = ActionConditionedTransformer(action_code_dim=16 if not args.fixed_color_jitter_order else 12,
                                               num_layers=args.num_layer, nhead=args.nhead,
                                               dim_feedforward=args.dim_feedforward)
        # if not args.flash_att:
        #     feature = ActionConditionedTransformer(action_code_dim=16 if not args.fixed_color_jitter_order else 12,
        #         num_layers=args.num_layer, nhead=args.nhead, dim_feedforward=args.dim_feedforward)
        # else:
        #     feature = ActionConditionedTransformerFlash(action_code_dim=16 if not args.fixed_color_jitter_order else 12,
        #         num_layers=args.num_layer, nhead=args.nhead, dim_feedforward=args.dim_feedforward)

    # if args.load:
    #     # feature = load_model(feature, args.feature_model_path)
    #     print("Loading decoder only...")
    #     decoder = load_model(decoder, args.decoder_model_path)

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


    # size = int((256 / 224) * args.image_size)
    # preprocess = transforms.Compose([
    #     transforms.Resize(size),
    #     transforms.CenterCrop(args.image_size),
    #     ])

    # use the entire image as anchor
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        ])

    print("\nUsing pytorch image augmentations")

    action_generator = aug_with_action_code(size=args.image_size, p_flip=0 if not args.random_flip else 0.5,
                                            fixed_color_jitter_order=args.fixed_color_jitter_order)

    train_dataset = ActionCodeDataset(traindir, action_generator, preprocess)

    ### Validation Dataset
    val_dataset = ActionCodeDataset(valdir, action_generator, preprocess)

    ##DDP Sampler
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                num_workers=args.workers, pin_memory=True, sampler=val_sampler)





    print('\nStarting training...')
    scaler = GradScaler()

    for epoch in range(1, args.epochs+1):
        ## train for one epoch
        # DDP init
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        if epoch == 1:
            validate_recon(val_loader, encoder, decoder, feature, criterion, args, experiment, device)

        log_flag = True if epoch % 5 == 0 else False

        train_one_epoch(train_loader, encoder, decoder, feature, criterion, optimizer, epoch, args, experiment, device, scaler, log_flag)
        lr_scheduler.step()

        if epoch % 10 == 0:
            validate_recon(val_loader, encoder, decoder, feature, criterion, args, experiment, device)
        dist.barrier()

    test(valdir, encoder, decoder, feature, criterion, args, experiment, device, preprocess)


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

        with (autocast()):
            z = encoder(img.to(device))
            z_t = encoder(trans_img.to(device))

            z_t_prime = feature(z, action_code.to(device))

            # feature transfer loss
            loss_1 = criterion(z_t_prime, z_t)

            # reconstruction
            if args.no_action_code:
                dummy_code = []
                for _ in range(z.size(0)):
                    dummy_code.append(no_action_code(args.fixed_color_jitter_order))
                dummy_code = torch.stack(dummy_code)

                r = decoder(feature(z, dummy_code.to(device)))
            else:
                r = decoder(z)

            r_t = decoder(z_t_prime)
            z_prime = encoder(r)
            # reconstruction loss for original image
            loss_2 = criterion(z_prime, z)

            z_t_pp = encoder(r_t)
            # reconstruction loss for transformed image
            loss_3 = criterion(z_t_pp, z_t)

            if torch.rand(1) < args.prob_orig_view:
                loss_2_weight = 1
            else:
                loss_2_weight = 0

            loss = loss_1 + loss_2 * loss_2_weight + loss_3

            if args.trans_loss:
                if args.no_action_code:
                    dummy_code = []
                    for _ in range(z.size(0)):
                        dummy_code.append(no_action_code(args.fixed_color_jitter_order))
                    dummy_code = torch.stack(dummy_code)

                    direct_recons_trans = decoder(feature(z_t, dummy_code.to(device)))
                else:
                    direct_recons_trans = decoder(z_t)
                z_t_d = encoder(direct_recons_trans)
                loss_4 = criterion(z_t, z_t_d)
                loss += loss_4




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
            if args.trans_loss:
                logged_dict['train_recon_trans_feat_loss_direct'] = loss_4.item()


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


def validate_recon(val_loader, encoder, decoder, feature, criterion, args, exp, device, suffix=''):
    batch_time = AverageMeter()
    losses = torch.tensor(0.0).to(device)
    feature_recon_losses = torch.tensor(0.0).to(device)
    feature_trans_losses = torch.tensor(0.0).to(device)
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
            loss_1 = criterion(z_t_prime, z_t)
            feature_trans_losses += loss_1.item()

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

            feature_recon_losses += loss_3.item()

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
    dist.reduce(feature_recon_losses, 0, op=dist.ReduceOp.SUM)
    dist.reduce(feature_trans_losses, 0, op=dist.ReduceOp.SUM)
    dist.reduce(samples, 0, op=dist.ReduceOp.SUM)


    if args.local_rank == 0:
        log_dict['val_pixel_loss'] = losses.item() / samples.item()
        log_dict['val_feature_recon_loss'] = feature_recon_losses.item() / samples.item()
        log_dict['val_feature_trans_loss'] = feature_trans_losses.item() / samples.item()
        if suffix != '':
            # add suffix to the log_dict keys
            log_dict = {k + suffix: v for k, v in log_dict.items()}
        exp.log(log_dict)
        print('Validation Loss: ', losses.item() / samples.item())

    return


def test(valdir, encoder, decoder, feature, criterion, args, exp, device, preprocess):
    probs = {
        'p_flip': 0,
        'p_crop': 0,
        'p_color_jitter': 0,
        'p_grayscale': 0,
        'p_gaussian_blur': 0,
        'p_solarization': 0,
    }

    keys = list(probs.keys())
    for i in range(len(keys)+1):
        if i == len(keys):
            k = 'p_none'
        else:
            k = keys[i]
            probs[k] = 1

        if args.local_rank == 0:
            print(f"Testing {k}")
        action_generator = aug_with_action_code(size=args.image_size, fixed_color_jitter_order=args.fixed_color_jitter_order, **probs)
        test_dataset = ActionCodeDataset(valdir, action_generator, preprocess)
        test_sampler = DistributedSampler(test_dataset)
        test_sampler.set_epoch(0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 num_workers=args.workers, pin_memory=True, sampler=test_sampler)
        validate_recon(test_loader, encoder, decoder, feature, criterion, args, exp, device, suffix=f'_{k}')
        if k in keys:
            probs[k] = 0


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
