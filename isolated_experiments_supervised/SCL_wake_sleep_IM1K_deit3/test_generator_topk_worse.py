import argparse
import os, time

import torch
import torch.nn.functional as F
from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental
import torchvision
from torch.cuda.amp import GradScaler, autocast


from models_deit3 import *
from augmentations import Episode_Transformations, collate_function
# from augmentations_randcrop import Episode_Transformations, collate_function
# from augmentations_hflip import Episode_Transformations, collate_function
# from augmentations_colorjitter import Episode_Transformations, collate_function
# from augmentations_grayscale import Episode_Transformations, collate_function
# from augmentations_blur import Episode_Transformations, collate_function
# from augmentations_solarization import Episode_Transformations, collate_function


from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

from copy import deepcopy

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet2012')
parser.add_argument('--num_pretraining_classes', type=int, default=10)
parser.add_argument('--data_order_file_name', type=str, default='./IM1K_data_class_orders/imagenet_class_order_siesta.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
# parser.add_argument('--enc_pretrained_file_path', type=str, default='./output/Pretrained_encoders/PreEnc100c_deit_tiny_patch16_LS_views@4no1stview_epochs@100_lr@0.0032_wd@0.05_bs@512_koleo@0.01_droppath@0.0125_seed@0/view_encoder_epoch99.pth')
parser.add_argument('--enc_pretrained_file_path', type=str, default='./output/Pretrained_condgen_AND_enc/deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.001wd@0.05droppath@0.0125_CONDGEN_lr@0.001wd@0layers@8heads@8dimff@1024dropout@0_seed@0/view_encoder_epoch99.pth')

# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
# parser.add_argument('--condgen_pretrained_file_path', type=str, default='./output/Pretrained_condgenerators/FOR_PreEnc100c_deit_tiny_patch16_LS_views@4no1stview_epochs@100_lr@0.0032_wd@0.05_bs@512_koleo@0.01_droppath@0.0125_seed@0/MultTokenV2_2.5tanh0.4237_PreCondGen10c_8layers1024dim8nheads_views@2_epochs@100warm5_lr@0.001_wd@0_bs@104_seed@0_dropout@0/cond_generator_epoch99.pth')
parser.add_argument('--condgen_pretrained_file_path', type=str, default='./output/Pretrained_condgen_AND_enc/deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.001wd@0.05droppath@0.0125_CONDGEN_lr@0.001wd@0layers@8heads@8dimff@1024dropout@0_seed@0/cond_generator_epoch99.pth')
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
parser.add_argument('--episode_batch_size', type=int, default=1024)#256)
parser.add_argument('--num_views', type=int, default=2)
# Other parameters
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--save_dir', type=str, default="output/condgen_testing_allaug")
# parser.add_argument('--save_dir', type=str, default="output/condgen_testing_randcrop")
# parser.add_argument('--save_dir', type=str, default="output/condgen_testing_hflip")
# parser.add_argument('--save_dir', type=str, default="output/condgen_testing_colorjitter")
# parser.add_argument('--save_dir', type=str, default="output/condgen_testing_grayscale")
# parser.add_argument('--save_dir', type=str, default="output/condgen_testing_blur")
# parser.add_argument('--save_dir', type=str, default="output/condgen_testing_solarization")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations.
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

def _update_top(top_list, k, loss_val, payload):
    """
    Keep the k highest-loss payloads in `top_list`.
    `top_list` is a list of (loss_value, payload_dict) tuples.
    """
    if len(top_list) < k:
        top_list.append((loss_val, payload))
    else:
        # replace the current minimum if the new loss is larger
        min_idx = min(range(len(top_list)), key=lambda i: top_list[i][0])
        if loss_val > top_list[min_idx][0]:
            top_list[min_idx] = (loss_val, payload)

def plot_episode(args, payload, title):
    V = payload["orig_imgs"].size(0)

    # row‑0 originals (undo normalisation)
    orig = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in payload["orig_imgs"]]
    orig = torch.stack(orig, dim=0) # (V,C,H,W)        

    # row‑1 generated (tanh → [-1,1] → original colour space)
    gen  = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in payload["gen_imgs"]]
    gen = torch.stack(gen, dim=0) # (V,C,H,W)
    gen = torch.clamp(gen, 0, 1)   # (V,C,H,W)

    # row‑2 heat‑maps (already (V,14,14))
    maps = payload["loss_maps"]
    vmin, vmax = maps.min().item(), maps.max().item()     # global range

    fig, ax = plt.subplots(3, V, figsize=(3*V, 9))
    if V == 1:                                              #  keep dims uniform
        ax = np.expand_dims(ax, 1)

    im_ref = None                                         # ref for colour‑bar
    for v in range(V):
        # --- first row -----------------------------------------------------
        ax[0, v].imshow(orig[v].permute(1,2,0))
        ax[0, v].axis("off")
        ttl = f"loss={payload['losses'][v]:.4f}"
        if v == payload["view_idx"]:
            ax[0, v].set_title(ttl, fontsize=8, fontweight="bold", color='red')
        else:
            ax[0, v].set_title(ttl, fontsize=8)

        # --- second row ----------------------------------------------------
        ax[1, v].imshow(gen[v].permute(1,2,0))
        ax[1, v].axis("off")

        # --- third row -----------------------------------------------------
        im = ax[2, v].imshow(maps[v], cmap="hot", vmin=vmin, vmax=vmax, origin="upper")
        ax[2, v].axis("off")
        if im_ref is None:          # remember first image for colour‑bar
            im_ref = im

    # single colour‑bar for the whole bottom row
    cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.24])
    fig.colorbar(im_ref, cax=cbar_ax)

    fig.suptitle(f"{title}  ·  worst episode score = {payload['loss']:.4f}  "
                f"(worst-view = {payload['view_idx']})", fontsize=14)
    plt.tight_layout(rect=[0,0,0.9,1])   # leave room for colour‑bar
    plt.savefig(os.path.join(args.save_dir, f'{title}.png'), bbox_inches='tight')
    plt.close(fig)

def _plot_hist(args, per_view_lists, title, fname):
    V = args.num_views
    
    global_min = min(min(lst) for lst in per_view_lists)
    global_max = max(max(lst) for lst in per_view_lists)

    fig, ax = plt.subplots(1, V, figsize=(4*V, 4), sharey=True)
    if V == 1:         # keep dimensions uniform when V == 1
        ax = [ax]
    for v in range(V):
        ax[v].hist(per_view_lists[v], bins=50)
        ax[v].set_xlim(global_min, global_max)
        ax[v].set_title(f"view {v}")
        ax[v].set_xlabel("MSE")
        if v == 0:
            ax[v].set_ylabel("count")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, fname), dpi=300)
    plt.close(fig)

def _plot_mean_maps(args, mean_maps, title, fname):
    V = args.num_views
    vmin, vmax = mean_maps.min().item(), mean_maps.max().item()

    fig, ax = plt.subplots(1, V, figsize=(3*V, 3))
    if V == 1:                                     # keep dimension uniform
        ax = [ax]

    img_ref = None
    for v in range(V):
        im = ax[v].imshow(mean_maps[v], cmap='hot',
                          vmin=vmin, vmax=vmax, origin='upper')
        ax[v].axis('off')
        ax[v].set_title(f'view {v}')
        if img_ref is None:
            img_ref = im

    # one colour-bar for all panels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img_ref, cax=cbar_ax)

    fig.suptitle(title)
    plt.tight_layout(rect=[0,0,0.9,1])
    plt.savefig(os.path.join(args.save_dir, fname), dpi=300)
    plt.close(fig)

def main():

    ### Parse arguments
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir): # create save dir
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Seed everything
    seed_everything(seed=args.seed)

    ### Load data
    print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size, shuffle=True, collate_fn=collate_function,
                                               num_workers=args.workers)#, drop_last=True)
    
    ### Load pretrained view_encoder
    print('\n==> Load pre-trained view encoder...')
    view_encoder = eval(args.enc_model_name)(output_before_pool = True)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(torch.load(args.enc_pretrained_file_path), strict=False)
    assert missing_keys == ['head.weight', 'head.bias'] and unexpected_keys == []
    feature_dim = view_encoder.head.weight.shape[1]
    view_encoder.head = torch.nn.Identity() # remove last fc layer from view_encoder network (it is not trained)
    for param in view_encoder.parameters(): # freeze view_encoder
        param.requires_grad = False

    ### Load Conditional generator
    print('\n==> Load pre-trained Conditional Generator')
    cond_generator = eval(args.condgen_model_name)(img_num_tokens=args.img_num_tokens,
                                                img_feature_dim = feature_dim,
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
    missing_keys, unexpected_keys = cond_generator.load_state_dict(torch.load(args.condgen_pretrained_file_path), strict=True)
    for param in cond_generator.parameters():
        param.requires_grad = False

    ### Print models
    print('\nView encoder')
    print(view_encoder)
    print('\nConditional Generator')
    print(cond_generator)
    print('\n')

    ### Dataparallel and move models to device
    # view_encoder = torch.nn.DataParallel(view_encoder)
    # cond_generator = torch.nn.DataParallel(cond_generator)
    view_encoder = view_encoder.to(args.device)
    cond_generator = cond_generator.to(args.device)

    ### Load optimizer and criterion
    criterion = torch.nn.MSELoss(reduction='none')

    ### Testing
    print('\n==> Testing model')

    ## Val STEP ##
    view_encoder.eval()
    cond_generator.eval()

    top_eps_gen1 = []          # keeps tuples (episode_loss , payload)
    top_eps_gen2 = []

    max1_payload = None   # will keep a dict with everything we need to plot
    max2_payload = None

    lossgen1_view_hist = [[] for _ in range(args.num_views)]
    lossgen2_view_hist = [[] for _ in range(args.num_views)]

    sum_map_gen1 = torch.zeros(args.num_views, 14, 14, device=args.device)
    sum_map_gen2 = torch.zeros(args.num_views, 14, 14, device=args.device)
    total_count   = 0                         # number of episodes we’ve seen

    accum_loss=0
    with torch.no_grad():
        for i, (batch_episode, _, _) in enumerate(train_loader):
            batch_episodes_imgs = batch_episode[0].to(args.device, non_blocking=True)
            batch_episodes_actions = batch_episode[1]

            # Pass obtained tensors through conditional generator
            batch_episodes_tensors = torch.empty(0).to(args.device)
            batch_episodes_gen_FTtensors = torch.empty(0).to(args.device)
            batch_episodes_gen_DecEnctensors = torch.empty(0).to(args.device)
            batch_episodes_gen_DecEnctensors_direct = torch.empty(0).to(args.device)
            batch_episodes_gen_images = torch.empty(0).to(args.device)

            batch_first_view_tensors = view_encoder(batch_episodes_imgs[:,0])[:, 1:, :]
            for v in range(args.num_views):
                batch_imgs = batch_episodes_imgs[:,v]
                batch_actions = [batch_episodes_actions[j][v] for j in range(batch_imgs.shape[0])] # (B, A)

                # Forward pass on view encoder
                batch_tensors = view_encoder(batch_imgs)[:, 1:, :]

                # Conditional forward pass (When v=0, the action codes are "no action". Using first view to predict the same first view)
                batch_gen_images, batch_gen_FTtensors = cond_generator(batch_first_view_tensors, batch_actions)
                batch_gen_DecEnctensors = view_encoder(batch_gen_images)[:, 1:, :]

                # Direct forward pass (skip FTN to boost training of generator)
                batch_gen_images_direct = cond_generator(batch_tensors, None, skip_conditioning=True)
                batch_gen_DecEnctensors_direct = view_encoder(batch_gen_images_direct)[:, 1:, :]

                # Concatenate tensors
                batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_tensors.unsqueeze(1)], dim=1)
                batch_episodes_gen_FTtensors = torch.cat([batch_episodes_gen_FTtensors, batch_gen_FTtensors.unsqueeze(1)], dim=1)
                batch_episodes_gen_DecEnctensors = torch.cat([batch_episodes_gen_DecEnctensors, batch_gen_DecEnctensors.unsqueeze(1)], dim=1)
                batch_episodes_gen_DecEnctensors_direct = torch.cat([batch_episodes_gen_DecEnctensors_direct, batch_gen_DecEnctensors_direct.unsqueeze(1)], dim=1)
                batch_episodes_gen_images = torch.cat([batch_episodes_gen_images, batch_gen_images.unsqueeze(1)], dim=1)

            # conditional loss (FT) (first view + action --> other views)
            batch_episodes_lossgen1_raw = criterion(batch_episodes_gen_FTtensors, batch_episodes_tensors) # shape [batch_size, num_views, num_tokens, feature_dim]
            batch_episodes_lossgen1_raw = batch_episodes_lossgen1_raw.permute(0, 1, 3, 2) # shape [batch_size, num_views, feature_dim, num_tokens]
            batch_episodes_lossgen1_raw = batch_episodes_lossgen1_raw.reshape(batch_episodes_lossgen1_raw.shape[0], batch_episodes_lossgen1_raw.shape[1], batch_episodes_lossgen1_raw.shape[2], 14, 14) # shape [batch_size, num_views, feature_dim, 14, 14]
            batch_episodes_lossgen1_spatial_per_view = batch_episodes_lossgen1_raw.mean(dim=2) # shape [batch_size, num_views, 14, 14]
            batch_episodes_lossgen1_mean_per_view = batch_episodes_lossgen1_raw.mean(dim=(2,3,4)) # shape [batch_size, num_views]
            lossgen_1 = batch_episodes_lossgen1_raw.mean()

            # conditional loss (DecEnc) (first view + action --> other views)
            batch_episodes_lossgen2_raw = criterion(batch_episodes_gen_DecEnctensors, batch_episodes_tensors)
            batch_episodes_lossgen2_raw = batch_episodes_lossgen2_raw.permute(0, 1, 3, 2) # shape [batch_size, num_views, feature_dim, num_tokens]
            batch_episodes_lossgen2_raw = batch_episodes_lossgen2_raw.reshape(batch_episodes_lossgen2_raw.shape[0], batch_episodes_lossgen2_raw.shape[1], batch_episodes_lossgen2_raw.shape[2], 14, 14) # shape [batch_size, num_views, feature_dim, 14, 14]
            batch_episodes_lossgen2_spatial_per_view = batch_episodes_lossgen2_raw.mean(dim=(2))
            batch_episodes_lossgen2_mean_per_view = batch_episodes_lossgen2_raw.mean(dim=(2,3,4)) # shape [batch_size, num_views]
            lossgen_2 = batch_episodes_lossgen2_raw.mean()

            # direct loss (DecEnc) (views --> views)
            batch_episodes_lossgen3_raw = criterion(batch_episodes_gen_DecEnctensors_direct, batch_episodes_tensors)
            batch_episodes_lossgen3_raw = batch_episodes_lossgen3_raw.permute(0, 1, 3, 2) # shape [batch_size, num_views, feature_dim, num_tokens]
            batch_episodes_lossgen3_raw = batch_episodes_lossgen3_raw.reshape(batch_episodes_lossgen3_raw.shape[0], batch_episodes_lossgen3_raw.shape[1], batch_episodes_lossgen3_raw.shape[2], 14, 14) # shape [batch_size, num_views, feature_dim, 14, 14]
            batch_episodes_lossgen3_spatial_per_view = batch_episodes_lossgen3_raw.mean(dim=(2))
            batch_episodes_lossgen3_mean_per_view = batch_episodes_lossgen3_raw.mean(dim=(2,3,4)) # shape [batch_size, num_views]
            lossgen_3 = batch_episodes_lossgen3_raw.mean()

            # total loss
            loss = lossgen_1 + lossgen_2 + lossgen_3

            # accum total loss
            accum_loss += loss.item() # accumulate loss

            # Loss results per batch
            print(f'Batch {i+1}/{len(train_loader)} -- ' +
                f'Loss 1 (FT): {lossgen_1.item():.6f} -- ' +
                f'Loss 2 (DecEnc): {lossgen_2.item():.6f} -- ' +
                f'Loss 3 (DecEnc_direct): {lossgen_3.item():.6f} -- '
                f'Loss: {loss.item():.6f}'
                )
            
            B = batch_episodes_imgs.size(0)               # batch size

            # episode-level mean error (across views) --------------------------
            ep_loss1_vals = batch_episodes_lossgen1_mean_per_view.mean(dim=1)   # (B,)
            ep_loss2_vals = batch_episodes_lossgen2_mean_per_view.mean(dim=1)   # (B,)
            
            TOP_K=5
            for b in range(B):
                # ------------------ payload for loss-gen1 ---------------------
                view_idx1 = batch_episodes_lossgen1_mean_per_view[b].argmax().item()
                payload1  = {
                    "orig_imgs" : batch_episodes_imgs[b].cpu(),                 # (V,C,H,W)
                    "gen_imgs"  : batch_episodes_gen_images[b].cpu(),
                    "losses"    : batch_episodes_lossgen1_mean_per_view[b].cpu(),   # (V,)
                    "loss_maps" : batch_episodes_lossgen1_spatial_per_view[b].cpu(),# (V,14,14)
                    "view_idx"  : view_idx1,
                    "loss"      : ep_loss1_vals[b].item(),
                }
                _update_top(top_eps_gen1, TOP_K, ep_loss1_vals[b].item(), payload1)

                # ------------------ payload for loss-gen2 ---------------------
                view_idx2 = batch_episodes_lossgen2_mean_per_view[b].argmax().item()
                payload2  = {
                    "orig_imgs" : batch_episodes_imgs[b].cpu(),
                    "gen_imgs"  : batch_episodes_gen_images[b].cpu(),
                    "losses"    : batch_episodes_lossgen2_mean_per_view[b].cpu(),
                    "loss_maps" : batch_episodes_lossgen2_spatial_per_view[b].cpu(),
                    "view_idx"  : view_idx2,
                    "loss"      : ep_loss2_vals[b].item(),
                }
                _update_top(top_eps_gen2, TOP_K, ep_loss2_vals[b].item(), payload2)

            for v in range(args.num_views):                                     
                lossgen1_view_hist[v].extend(                                   
                    batch_episodes_lossgen1_mean_per_view[:, v].cpu().tolist()) 
                lossgen2_view_hist[v].extend(                                   
                    batch_episodes_lossgen2_mean_per_view[:, v].cpu().tolist())
                
            sum_map_gen1 += batch_episodes_lossgen1_spatial_per_view.sum(dim=0)   # (V,14,14)
            sum_map_gen2 += batch_episodes_lossgen2_spatial_per_view.sum(dim=0)
            total_count  += batch_episodes_lossgen1_spatial_per_view.size(0)       # add B

    accum_loss /= len(train_loader)
    print(f'Total Train Loss: {accum_loss:.6f}')

    # sort the lists so index 0 is the very worst episode
    top_eps_gen1 = sorted(top_eps_gen1, key=lambda x: x[0], reverse=True)
    top_eps_gen2 = sorted(top_eps_gen2, key=lambda x: x[0], reverse=True)

    # save every episode plot ------------------------------------------------
    for rank, (_, payload) in enumerate(top_eps_gen1, start=1):
        plot_episode(args, payload, f"worse_Top{rank}_episode_loss-gen1")

    for rank, (_, payload) in enumerate(top_eps_gen2, start=1):
        plot_episode(args, payload, f"worse_Top{rank}_episode_loss-gen2")

    _plot_hist(args, lossgen1_view_hist, "Error distribution per view (loss‑gen1)", "hist_lossgen1.png")
    _plot_hist(args, lossgen2_view_hist, "Error distribution per view (loss‑gen2)", "hist_lossgen2.png")

    mean_map_gen1 = (sum_map_gen1 / total_count).cpu()    # (V,14,14)
    mean_map_gen2 = (sum_map_gen2 / total_count).cpu()

    _plot_mean_maps(args, mean_map_gen1, 'Average spatial error per view (loss-gen1)', 'mean_heatmap_gen1.png')
    _plot_mean_maps(args, mean_map_gen2, 'Average spatial error per view (loss-gen2)', 'mean_heatmap_gen2.png')

    return None

if __name__ == '__main__':
    main()