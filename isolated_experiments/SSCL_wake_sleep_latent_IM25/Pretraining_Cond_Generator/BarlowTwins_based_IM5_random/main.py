import argparse
import os, time

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torch.cuda.amp import GradScaler, autocast


from models import *
from augmentations import Episode_Transformations

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

from copy import deepcopy

parser = argparse.ArgumentParser(description='Conditional Generator pre-training - Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-5-random')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters (pretrained)
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--enc_pretrained_file_path', type=str, default='./../../Pretraining_Encoder/Barlow_Twins_IM5_random/output/Barlow_IM5random_offPRE_encprojstandard_views@12_epochs@100_lr@0.003_wd@0_bs@128_seed@0/view_encoder_epoch99.pth')
# Conditional Generator parameters
parser.add_argument('--gen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--dec_num_Blocks', type=list, default=[1,1,1,1])
parser.add_argument('--dec_num_out_channels', type=int, default=3)
parser.add_argument('--ft_feature_dim', type=int, default=512)
parser.add_argument('--ft_action_code_dim', type=int, default=11)
parser.add_argument('--ft_num_layers', type=int, default=2)
parser.add_argument('--ft_nhead', type=int, default=4)
parser.add_argument('--ft_dim_feedforward', type=int, default=256)
parser.add_argument('--ft_dropout', type=float, default=0.1)
# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--episode_batch_size', type=int, default=80)
parser.add_argument('--num_views', type=int, default=6) 
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--save_dir', type=str, default="output/run_cond_generator_pretraining")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations
parser.add_argument('--seed', type=int, default=0)

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
    args.lr = args.lr * args.episode_batch_size / 128
    print(args)
    if not os.path.exists(args.save_dir): # create save dir
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Seed everything
    seed_everything(seed=args.seed)

    ### Define tensoboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load data (Only use first task for pre-training)
    print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=True)
    train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)#, persistent_workers=True, drop_last=True)

    ### Load pretrained view_encoder
    print('\n==> Load pre-trained view encoder...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(torch.load(args.enc_pretrained_file_path), strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    view_encoder.fc = torch.nn.Identity() # remove last fc layer from view_encoder network (it is not trained)
    for param in view_encoder.parameters(): # freeze view_encoder
        param.requires_grad = False

    ### Load Conditional generator
    print('\n==> Load Conditional Generator')
    cond_generator = eval(args.gen_model_name)(dec_num_Blocks = args.dec_num_Blocks, 
                                               dec_num_out_channels = args.dec_num_out_channels, 
                                               ft_feature_dim = args.ft_feature_dim, 
                                               ft_action_code_dim = args.ft_action_code_dim, 
                                               ft_num_layers = args.ft_num_layers, 
                                               ft_nhead = args.ft_nhead, 
                                               ft_dim_feedforward = args.ft_dim_feedforward, 
                                               ft_dropout = args.ft_dropout)

    ### Print models
    print('\nView encoder')
    print(view_encoder)
    print('\nConditional Generator')
    print(cond_generator)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    cond_generator = torch.nn.DataParallel(cond_generator)
    view_encoder = view_encoder.to(args.device)
    cond_generator = cond_generator.to(args.device)

    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(cond_generator.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.MSELoss().to(args.device)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr*1e-6, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    ### Save one batch for plot purposes
    episodes_plot, _ = next(iter(train_loader))

    ### Train loop
    print('\n==> Training model')
    scaler = GradScaler()
    init_time = time.time()
    for epoch in range(args.epochs):
        print(f'\n==> Epoch {epoch}/{args.epochs}')
        start_time = time.time()

        ## Train STEP ##
        total_loss=0
        view_encoder.eval()
        cond_generator.train()

        for i, (batch_episode, _) in enumerate(train_loader):
            batch_episode_imgs = batch_episode[0].to(args.device)
            batch_episode_actions = batch_episode[1].to(args.device)

            # Pass obtained tensors through conditional generator
            batch_episode_tensors = torch.empty(0).to(args.device)
            batch_episode_gen_FTtensors = torch.empty(0).to(args.device)
            batch_episode_gen_DecEnctensors = torch.empty(0).to(args.device)
            batch_episode_gen_DecEnctensors_direct = torch.empty(0).to(args.device)
            # batch_episode_gen_FTtensors_noaction = torch.empty(0).to(args.device)
            # batch_episode_gen_DecEnctensors_noaction = torch.empty(0).to(args.device)

            for v in range(args.num_views):
                batch_imgs = batch_episode_imgs[:,v]
                batch_actions = batch_episode_actions[:,v]

                with autocast():
                    # Forward pass on view encoder
                    batch_tensors = view_encoder(batch_imgs)

                    if v==0:
                        batch_first_view_tensors = deepcopy(batch_tensors)
                        batch_no_actions = deepcopy(batch_actions)

                    # Conditional forward pass (When v=0, the action codes are "no action". Using first view to predict the same first view)
                    batch_gen_images, batch_gen_FTtensors = cond_generator(batch_first_view_tensors, batch_actions)
                    batch_gen_DecEnctensors = view_encoder(batch_gen_images)

                    # Direct forward pass (skip FTN to boost training of generator)
                    batch_gen_images_direct = cond_generator(batch_tensors, None, skip_FTN=True)
                    batch_gen_DecEnctensors_direct = view_encoder(batch_gen_images_direct)

                    # Conditional forward pass with "no action" for other views (not just the first view)
                    # batch_gen_images_noaction, batch_gen_FTtensors_noaction = cond_generator(batch_tensors, batch_no_actions)
                    # batch_gen_DecEnctensors_noaction = view_encoder(batch_gen_images_noaction)

                # Concatenate tensors
                batch_episode_tensors = torch.cat([batch_episode_tensors, batch_tensors.unsqueeze(1)], dim=1)
                batch_episode_gen_FTtensors = torch.cat([batch_episode_gen_FTtensors, batch_gen_FTtensors.unsqueeze(1)], dim=1)
                batch_episode_gen_DecEnctensors = torch.cat([batch_episode_gen_DecEnctensors, batch_gen_DecEnctensors.unsqueeze(1)], dim=1)
                batch_episode_gen_DecEnctensors_direct = torch.cat([batch_episode_gen_DecEnctensors_direct, batch_gen_DecEnctensors_direct.unsqueeze(1)], dim=1)
                # batch_episode_gen_FTtensors_noaction = torch.cat([batch_episode_gen_FTtensors_noaction, batch_gen_FTtensors_noaction.unsqueeze(1)], dim=1)
                # batch_episode_gen_DecEnctensors_noaction = torch.cat([batch_episode_gen_DecEnctensors_noaction, batch_gen_DecEnctensors_noaction.unsqueeze(1)], dim=1)

            # conditional loss (FT) (first view + action --> other views)
            lossgen_1 = criterion(batch_episode_gen_FTtensors, batch_episode_tensors)
            # conditional loss (DecEnc) (first view + action --> other views)
            lossgen_2 = criterion(batch_episode_gen_DecEnctensors, batch_episode_tensors)
            # direct loss (DecEnc) (views --> views)
            lossgen_3 = criterion(batch_episode_gen_DecEnctensors_direct, batch_episode_tensors)
            # conditional loss (FT) (other views + no action --> other views)
            # lossgen_4 = criterion(batch_episode_gen_FTtensors_noaction, batch_episode_tensors)
            # conditional loss (DecEnc) (other views + no action --> other views)
            # lossgen_5 = criterion(batch_episode_gen_DecEnctensors_noaction, batch_episode_tensors)
            # total loss
            loss = lossgen_1 + lossgen_2 + lossgen_3 #+ lossgen_4 + lossgen_5

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log results
            total_loss += loss.item()
            if i % args.print_frequency == 0:
                print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' + 
                      f'Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- ' +
                      f'lr: {scheduler.get_last_lr()[0]:.6f} -- ' +
                      f'Loss 1 (FT): {lossgen_1.item():.6f} -- ' +
                      f'Loss 2 (DecEnc): {lossgen_2.item():.6f} -- ' +
                      f'Loss 3 (DecEnc_direct): {lossgen_3.item():.6f} -- '
                    #   f'Loss 4 (FT_noaction): {lossgen_4.item():.6f} -- ' +
                    #   f'Loss 5 (DecEnc_noaction): {lossgen_5.item():.6f} -- ' +
                      f'Total Loss: {loss.item():.6f}'
                    )
            scheduler.step()
            writer.add_scalar('Loss_1_FT (per batch)', lossgen_1.item(), epoch*len(train_loader)+i)
            writer.add_scalar('Loss_2_DecEnc (per batch)', lossgen_2.item(), epoch*len(train_loader)+i)
            writer.add_scalar('Loss_3_DecEnc_direct (per batch)', lossgen_3.item(), epoch*len(train_loader)+i)
            # writer.add_scalar('Loss_4_FT_noaction (per batch)', lossgen_4.item(), epoch*len(train_loader)+i)
            # writer.add_scalar('Loss_5_DecEnc_noaction (per batch)', lossgen_5.item(), epoch*len(train_loader)+i)
            writer.add_scalar('Total Loss (per batch)', loss.item(), epoch*len(train_loader)+i)
        total_loss /= len(train_loader)
        writer.add_scalar('Total Loss (per epoch)', total_loss, epoch)
        print(f'Epoch [{epoch}] Total Train Loss per Epoch: {total_loss:.6f}')
        print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

        ### Save model
        if (epoch+1) % 10 == 0 or epoch==0:
            cond_generator_state_dict = cond_generator.module.state_dict()
            torch.save(cond_generator_state_dict, os.path.join(args.save_dir, f'cond_generator_epoch{epoch}.pth'))

        ### Plot reconctructions examples every 1 epochs
        if (epoch+1) % 5 == 0 or epoch==0:
            view_encoder.eval()
            cond_generator.train()
            n = 8
            episodes_plot_imgs = episodes_plot[0][:n].to(args.device)
            episodes_plot_actions = episodes_plot[1][:n].to(args.device)
            episodes_plot_gen_imgs = torch.empty(0)
            with torch.no_grad():
                first_view_tensors = view_encoder(episodes_plot_imgs[:,0])
                for v in range(args.num_views):
                    actions = episodes_plot_actions[:,v]
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

    return None

if __name__ == '__main__':
    main()