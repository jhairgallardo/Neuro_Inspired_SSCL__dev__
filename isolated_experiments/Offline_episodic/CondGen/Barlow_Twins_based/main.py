import argparse
import os, time

import torch
import torch.distributed
import torch.nn.functional as F
from torchvision import datasets
import torchvision
from torch.cuda.amp import GradScaler, autocast


from models_encGNMish_ftrelu_genGNMish import *
from augmentations import Episode_Transformations

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

parser = argparse.ArgumentParser(description='Conditional Generator pre-training - Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters (pretrained)
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--enc_pretrained_file_path', type=str)
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
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=2) 
parser.add_argument('--workers', type=int, default=64)
parser.add_argument('--save_dir', type=str, default="output/run_cond_generator_pretraining")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations
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
    else:
        raise ValueError("This code is doesn't support single GPU runs. Please use torch.distributed.launch to run on multiple GPUs.")

    # Create save dir folders and save args
    if args.local_rank == 0:
        print(args)
        if not os.path.exists(args.save_dir): # create save dir
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)


    # Calculate batch size per GPU
    args.episode_batch_size_per_gpu = int(args.episode_batch_size / torch.distributed.get_world_size())
    # Calculate number of workers per GPU
    args.workers_per_gpu = int(args.workers / torch.distributed.get_world_size())        
    
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
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=True)
    train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, 
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True)

    ### Load pretrained view_encoder
    if args.local_rank == 0:
        print('\n==> Load pre-trained view encoder...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(torch.load(args.enc_pretrained_file_path), strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    view_encoder.fc = torch.nn.Identity() # remove last fc layer from view_encoder network (it is not trained)
    for param in view_encoder.parameters(): # freeze view_encoder
        param.requires_grad = False
    view_encoder.eval()
    if args.local_rank == 0:
        print("\nSuccessfully loaded pre-trained view encoder")

    ### Load Conditional generator
    if args.local_rank == 0:
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
    if args.local_rank == 0:
        print('\nView encoder')
        print(view_encoder)
        print('\nConditional Generator')
        print(cond_generator)
        print('\n')

    ### Move models to device, apply SyncBatchNorm and DDP
    view_encoder = view_encoder.to(device) # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient
    cond_generator = cond_generator.to(device)
    cond_generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cond_generator)
    cond_generator = torch.nn.parallel.DistributedDataParallel(cond_generator, device_ids=[args.local_rank], output_device=args.local_rank)#, find_unused_parameters=True)


    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(cond_generator.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.MSELoss()
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    ### Save one batch for plot purposes
    if args.local_rank == 0:
        episodes_plot, _ = next(iter(train_loader))

    ### Train loop
    if args.local_rank == 0:
        print('\n==> Training model')
    scaler = GradScaler()
    init_time = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()
        if args.local_rank == 0:
            print(f'\n==> Epoch {epoch}/{args.epochs}')
        
        # DDP init
        train_sampler.set_epoch(epoch)

        ## Train STEP ##
        total_loss=0
        view_encoder.eval()
        cond_generator.train()

        for i, (batch_episode, _) in enumerate(train_loader):
            batch_episode_imgs = batch_episode[0].to(device)
            batch_episode_actions = batch_episode[1].to(device)

            batch_first_view_images = batch_episode_imgs[:,0]

            lossgen_1 = 0
            lossgen_2 = 0
            lossgen_3 = 0

            with (autocast()):

                batch_first_view_tensors = view_encoder(batch_first_view_images)

                for v in range(args.num_views):
                    batch_imgs = batch_episode_imgs[:,v]
                    batch_actions = batch_episode_actions[:,v]
                    
                    # Forward pass on view encoder
                    batch_tensors = view_encoder(batch_imgs)

                    # Conditional forward pass (When v=0, the action codes are "no action". Using first view to predict the same first view)
                    batch_gen_images, batch_gen_FTtensors = cond_generator(batch_first_view_tensors, batch_actions)
                    batch_gen_DecEnctensors = view_encoder(batch_gen_images)

                    # Direct forward pass (skip FTN to boost training of generator)
                    batch_gen_images_direct = cond_generator(batch_tensors, None, skip_FTN=True)
                    batch_gen_DecEnctensors_direct = view_encoder(batch_gen_images_direct)

                    lossgen_1 += criterion(batch_gen_FTtensors, batch_tensors)
                    lossgen_2 += criterion(batch_gen_DecEnctensors, batch_tensors)
                    lossgen_3 += criterion(batch_gen_DecEnctensors_direct, batch_tensors)

                lossgen_1 /= args.num_views
                lossgen_2 /= args.num_views
                lossgen_3 /= args.num_views
                loss = lossgen_1 + lossgen_2 + lossgen_3

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Log results per batch
            if args.local_rank == 0:
                total_loss += loss.item()
                writer.add_scalar('Loss_1_FT (per batch)', lossgen_1.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss_2_DecEnc (per batch)', lossgen_2.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss_3_DecEnc_direct (per batch)', lossgen_3.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Total Loss (per batch)', loss.item(), epoch*len(train_loader)+i)
                if i % args.print_frequency == 0:
                    print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' + 
                        f'Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- ' +
                        f'lr: {scheduler.get_last_lr()[0]:.6f} -- ' +
                        f'Loss 1 (FT): {lossgen_1.item():.6f} -- ' +
                        f'Loss 2 (DecEnc): {lossgen_2.item():.6f} -- ' +
                        f'Loss 3 (DecEnc_direct): {lossgen_3.item():.6f} -- '
                        f'Total Loss: {loss.item():.6f}'
                        )
                        
        torch.distributed.barrier() # wait for all processes to finish before moving to the next epoch

        # Log results per epoch, save model, and plot reconstructions
        if args.local_rank == 0:
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
                cond_generator.eval()
                n = 8
                episodes_plot_imgs = episodes_plot[0][:n].to(device)
                episodes_plot_actions = episodes_plot[1][:n].to(device)
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