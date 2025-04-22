import argparse
import os, time

import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torchvision

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental

from models_GNMish_lossgen1only import *
from loss_functions import KoLeoLoss
from augmentations import Episode_Transformations
from utils import MetricLogger, accuracy

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet2012')
parser.add_argument('--num_pretraining_classes', type=int, default=100)
parser.add_argument('--data_order_file_name', type=str, default='./IM1K_data_class_orders/imagenet_class_order_siesta.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--enc_pretrained_file_path', type=str)
# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--action_code_dim', type=int, default=12)
parser.add_argument('--ft_num_layers', type=int, default=8) # 2
parser.add_argument('--ft_nhead', type=int, default=8) # 4
parser.add_argument('--ft_dim_feedforward', type=int, default=1024) # 256
parser.add_argument('--ft_dropout', type=float, default=0.1)
parser.add_argument('--dec_num_Blocks', type=list, default=[1,1,1,1])
parser.add_argument('--dec_num_out_channels', type=int, default=3)
# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--episode_batch_size', type=int, default=104) 
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001) # 0.001
parser.add_argument('--wd', type=float, default=0) # 0.01
# Other parameters
parser.add_argument('--workers', type=int, default=48) # 8 for 1 gpu, 48 for 4 gpus
parser.add_argument('--save_dir', type=str, default="output/Pretrained_condgenerators/run_debug")
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
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=True)
    train_dataset_continuum = ImageFolderDataset(traindir)

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
    train_dataset = train_tasks[0] # Create the train dataset taking only the first task (the first 100 classes)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True if train_sampler is None else False,
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True)
    
    # ### Plot some images as examples
    # if args.local_rank == 0:
    #     print('\n==> Plotting some images as examples...')
    #     import matplotlib.pyplot as plt
    #     import torchvision.utils as vutils
    #     import numpy as np
    #     batch_episodes_imgs, batch_labels, _ = next(iter(train_loader))
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
        print('\n==> Prepare models...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual=True, output_before_avgpool=True)
    cond_generator = eval(args.condgen_model_name)(action_code_dim=args.action_code_dim,
                                            feature_dim=view_encoder.fc.weight.shape[1],
                                            ft_num_layers=args.ft_num_layers,
                                            ft_nhead=args.ft_nhead,
                                            ft_dim_feedforward=args.ft_dim_feedforward,
                                            ft_dropout=args.ft_dropout,
                                            dec_num_Blocks=args.dec_num_Blocks,
                                            dec_num_out_channels=args.dec_num_out_channels)
    # Load pretrained view encoder
    missing_keys, unexpected_keys = view_encoder.load_state_dict(torch.load(args.enc_pretrained_file_path), strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    view_encoder.fc = torch.nn.Identity() # remove the head of the encoder
    for param in view_encoder.parameters(): # freeze view_encoder
        param.requires_grad = False
    view_encoder.eval()
                                                  
    ### Print models
    if args.local_rank == 0:
        print('\nView encoder')
        print(view_encoder)
        print('\nClassifier')
        print(cond_generator)
        print('\n')

    ### Dataparallel and move models to device
    view_encoder = view_encoder.to(device)
    cond_generator = cond_generator.to(device)
    if args.ddp: # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient
        cond_generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cond_generator)
        cond_generator = torch.nn.parallel.DistributedDataParallel(cond_generator, device_ids=[args.local_rank], output_device=args.local_rank)

    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(cond_generator.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.MSELoss()
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1e-6/args.lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    ### Save one batch for plot purposes
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
        
        ##################
        ### Train STEP ###
        ##################
        loss_total_log = MetricLogger('Loss Total')
        loss_gen1_log = MetricLogger('Loss Gen1')
        loss_gen2_log = MetricLogger('Loss Gen2')
        loss_gen3_log = MetricLogger('Loss Gen3')

        view_encoder.eval()
        cond_generator.train()
        for i, (batch_episodes, _, _) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes[0].to(device, non_blocking=True) # (B, V, C, H, W)
            batch_episodes_actions = batch_episodes[1].to(device, non_blocking=True) # (B, V, A)

            ## Forward pass
            loss_gen1 = 0
            # loss_gen2 = 0
            # loss_gen3 = 0
            batch_first_view_images = batch_episodes_imgs[:,0] # (B, C, H, W)

            # Sanity check no action run
            # batch_no_actions = batch_episodes_actions[:,0] 
            with (autocast()):
                batch_first_view_tensors = view_encoder(batch_first_view_images)
                for v in range(args.num_views):
                    batch_imgs = batch_episodes_imgs[:,v]
                    batch_actions = batch_episodes_actions[:,v]

                    # Forward pass on view encoder
                    batch_tensors = view_encoder(batch_imgs)

                    # Sanity check no action run
                    # batch_gen_images, batch_gen_FTtensors = cond_generator(batch_tensors, batch_no_actions) 

                    # Conditional forward pass (Special case: When v=0, the action codes are "no action", meaning it uses the first view to predict the same first view)
                    batch_gen_images, batch_gen_FTtensors = cond_generator(batch_first_view_tensors, batch_actions)
                    # batch_gen_DecEnctensors = view_encoder(batch_gen_images)

                    # Direct forward pass (skip FTN to boost training of generator)
                    # batch_gen_images_direct = cond_generator(batch_tensors, None, skip_conditioning=True)
                    # batch_gen_DecEnctensors_direct = view_encoder(batch_gen_images_direct)

                    # Calculate loss and accumulate across views
                    loss_gen1 += criterion(batch_gen_FTtensors, batch_tensors)
                    # loss_gen2 += criterion(batch_gen_DecEnctensors, batch_tensors)
                    # loss_gen3 += criterion(batch_gen_DecEnctensors_direct, batch_tensors)

            # Normalize loss across views
            loss_gen1 /= args.num_views
            # loss_gen2 /= args.num_views
            # loss_gen3 /= args.num_views
 
            # Calculate Total loss for the batch
            loss_total = loss_gen1 #+ loss_gen2 + loss_gen3

            ## Backward pass with clip norm
            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss_total.backward()
            # torch.nn.utils.clip_grad_norm_(cond_generator.parameters(), 1.0)
            # optimizer.step()
            scheduler.step()

            ## Track losses for per epoch plotting
            loss_total_log.update(loss_total.item(), batch_episodes_imgs.size(0))
            loss_gen1_log.update(loss_gen1.item(), batch_episodes_imgs.size(0))
            # loss_gen2_log.update(loss_gen2.item(), batch_episodes_imgs.size(0))
            # loss_gen3_log.update(loss_gen3.item(), batch_episodes_imgs.size(0))

            if (args.local_rank == 0) and ((i % args.print_frequency) == 0):
                print(
                    f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                    f'lr: {scheduler.get_last_lr()[0]:.6f} -- ' +
                    f'Loss Gen1: {loss_gen1.item():.6f} -- ' +
                    # f'Loss Gen2: {loss_gen2.item():.6f} -- ' +
                    # f'Loss Gen3: {loss_gen3.item():.6f} -- ' +
                    f'Loss Total: {loss_total.item():.6f}'
                    )
                # Print the max gradient value of the network
                # all_gradients = []
                # for param in cond_generator.parameters():
                #     if param.grad is not None:
                #         all_gradients.append(param.grad.data.abs().max())
                # if len(all_gradients) > 0:
                #     max_grad = torch.max(torch.stack(all_gradients))
                #     print(f'Max gradient: {max_grad.item():.6f}')

                #     avg_grad = torch.mean(torch.stack(all_gradients))
                #     print(f'Avg gradient: {avg_grad.item():.6f}')

            if args.local_rank == 0:
                writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Gen1', loss_gen1.item(), epoch*len(train_loader)+i)
                # writer.add_scalar('Loss Gen2', loss_gen2.item(), epoch*len(train_loader)+i)
                # writer.add_scalar('Loss Gen3', loss_gen3.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Total', loss_total.item(), epoch*len(train_loader)+i)
        
        # Train Epoch metrics
        if args.ddp:
            loss_total_log.all_reduce()
            loss_gen1_log.all_reduce()
            # loss_gen2_log.all_reduce()
            # loss_gen3_log.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Total (per epoch)', loss_total_log.avg, epoch)
            writer.add_scalar('Loss Gen1 (per epoch)', loss_gen1_log.avg, epoch)
            # writer.add_scalar('Loss Gen2 (per epoch)', loss_gen2_log.avg, epoch)
            # writer.add_scalar('Loss Gen3 (per epoch)', loss_gen3_log.avg, epoch)
            # print(f'Epoch [{epoch}] Loss Gen1: {loss_gen1_log.avg:.6f} -- Loss Gen2: {loss_gen2_log.avg:.6f} -- Loss Gen3: {loss_gen3_log.avg:.6f} -- Loss Total: {loss_total_log.avg:.6f}')
            print(f'Epoch [{epoch}] Loss Gen1: {loss_gen1_log.avg:.6f} -- Loss Total: {loss_total_log.avg:.6f}')


        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes to finish the training epoch

        ### Save model ###
        if (args.local_rank == 0) and (((epoch+1) % 10) == 0) or epoch==0:
            if args.ddp:
                cond_generator_state_dict = cond_generator.module.state_dict()
            else:
                cond_generator_state_dict = cond_generator.state_dict()
            torch.save(cond_generator_state_dict, os.path.join(args.save_dir, f'cond_generator_epoch{epoch}.pth'))

        ### Plot reconstructions examples ###
        # if args.local_rank == 0:
        #     if (epoch+1) % 5 == 0 or epoch==0:
        #         view_encoder.eval()
        #         cond_generator.eval()
        #         n = 8
        #         episodes_plot_imgs = episodes_plot[0][:n].to(device, non_blocking=True)
        #         episodes_plot_actions = episodes_plot[1][:n].to(device, non_blocking=True)
        #         episodes_plot_gen_imgs = torch.empty(0)
        #         with torch.no_grad():
        #             first_view_tensors = view_encoder(episodes_plot_imgs[:,0])
        #             for v in range(args.num_views):
        #                 actions = episodes_plot_actions[:,v]
        #                 gen_images, _ = cond_generator(first_view_tensors, actions)
        #                 episodes_plot_gen_imgs = torch.cat([episodes_plot_gen_imgs, gen_images.unsqueeze(1).detach().cpu()], dim=1)
        #         episodes_plot_imgs = episodes_plot_imgs.detach().cpu()
        #         # plot each episode
        #         for i in range(n):
        #             episode_i_imgs = episodes_plot_imgs[i]
        #             episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_imgs]
        #             episode_i_imgs = torch.stack(episode_i_imgs, dim=0)

                    # episode_i_gen_imgs = episodes_plot_gen_imgs[i]
                    # min_vals = episode_i_gen_imgs.amin(dim=(-1,-2), keepdim=True)
                    # max_vals = episode_i_gen_imgs.amax(dim=(-1,-2), keepdim=True)
                    # episode_i_gen_imgs = (episode_i_gen_imgs - min_vals) / (max_vals - min_vals)

        #             grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs], dim=0), nrow=args.num_views)
        #             grid = grid.permute(1, 2, 0).cpu().numpy()
        #             grid = (grid * 255).astype(np.uint8)
        #             grid = Image.fromarray(grid)
        #             image_name = f'epoch{epoch}_episode{i}.png'
        #             save_plot_dir = os.path.join(args.save_dir, 'gen_plots')
        #             # create folder if it doesn't exist
        #             if not os.path.exists(save_plot_dir):
        #                 os.makedirs(save_plot_dir)
        #             grid.save(os.path.join(save_plot_dir, image_name))

        if args.local_rank == 0:
            print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    return None

if __name__ == '__main__':
    main()