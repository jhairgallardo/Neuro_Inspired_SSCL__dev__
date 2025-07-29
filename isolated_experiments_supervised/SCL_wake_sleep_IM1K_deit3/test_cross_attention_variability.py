import argparse
import os, time

import torch
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torchvision

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental

from models_deit3_projcos import *
from augmentations import Episode_Transformations, collate_function
from utils import MetricLogger, accuracy, time_duration_print

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

import matplotlib.pyplot as plt

folder = 'projcosOPTI_augaggV0normal64d_3tanh_deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.0008wd@0.05droppath@0.0125_CONDGEN_lr@0.0008wd@0layers@8heads@8dimff@1024dropout@0_seed@0'

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet1K')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--num_pretraining_classes', type=int, default=10)
parser.add_argument('--data_order_file_name', type=str, default='./IM1K_data_class_orders/imagenet_class_order_siesta.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--enc_pretrained_file_path', type=str, default=f'./output/Pretrained_condgen_AND_enc/{folder}/view_encoder_epoch99.pth')
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
parser.add_argument('--classifier_pretrained_file_path', type=str, default=f'./output/Pretrained_condgen_AND_enc/{folder}/classifier_epoch99.pth')
# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--condgen_pretrained_file_path', type=str, default=f'./output/Pretrained_condgen_AND_enc/{folder}/cond_generator_epoch99.pth')
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
# Conditional generator loss weight
parser.add_argument('--gen_alpha', type=float, default=1.0)
# Training parameters
parser.add_argument('--episode_batch_size', type=int, default=20)
parser.add_argument('--num_views', type=int, default=4)
# Other parameters
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/Pretrained_condgen_AND_enc/test")#_cross_attention_variability_normal")
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
    val_tasks_ep = ClassIncremental(val_dataset_continuum, increment=1, initial_increment=args.num_pretraining_classes, transformations=[train_transform], class_order=data_class_order)
    train_dataset = train_tasks[0] # Create the train dataset taking only the first task (the first 100 classes)
    val_dataset = val_tasks[0] # Create the val dataset taking only the first task (the first 100 classes)
    val_dataset_ep = val_tasks_ep[0] # Create the val dataset taking only the first task (the first 100 classes)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_sampler_ep = torch.utils.data.distributed.DistributedSampler(val_dataset_ep, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        val_sampler_ep = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True if train_sampler is None else False,
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True,
                                               collate_fn=collate_function)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                             sampler=val_sampler, num_workers=args.workers_per_gpu, pin_memory=True)
    val_loader_ep = torch.utils.data.DataLoader(val_dataset_ep, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                                sampler=val_sampler_ep, num_workers=args.workers_per_gpu, pin_memory=True,
                                               collate_fn=collate_function)

    ### Load models
    if args.local_rank == 0:
        print('\n==> Prepare models...')

    view_encoder = eval(args.enc_model_name)(output_before_pool=True)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(torch.load(args.enc_pretrained_file_path), strict=False)
    assert missing_keys == ['head.weight', 'head.bias'] and unexpected_keys == []

    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.embed_dim, num_classes=args.num_classes)
    missing_keys, unexpected_keys = classifier.load_state_dict(torch.load(args.classifier_pretrained_file_path), strict=True)

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
    missing_keys, unexpected_keys = cond_generator.load_state_dict(torch.load(args.condgen_pretrained_file_path), strict=True)

    view_encoder.head = torch.nn.Identity() # remove the head of the encoder

    # Freeze parameters of each network
    for param in view_encoder.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = False
    for param in cond_generator.parameters():
        param.requires_grad = False
                                                  
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

    ### Get batch of data to test the model on
    seed_everything(final_seed)  # Reset seed to ensure reproducibility for the batch
    if args.local_rank == 0:
        episodes_trainset, episodes_trainset_labels, _ = next(iter(train_loader))
        episodes_valepset, episodes_valepset_labels, _ = next(iter(val_loader_ep))

    view_encoder.eval()
    classifier.eval()
    cond_generator.eval()

    ### Test cross-attention variability (episodes_trainset)
    batch_episodes_imgs = episodes_trainset[0].to(device, non_blocking=True)  # (B, V, C, H, W)
    batch_episodes_labels = episodes_trainset_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True)  # (B, V)
    batch_episodes_actions = episodes_trainset[1]  # (B, V, A)
    B, V, C, H, W = batch_episodes_imgs.shape
    with torch.no_grad():
        # Flatten the batch and views
        flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W)
        flat_feats_and_cls = view_encoder(flat_imgs)  # (B*V, 1+T, D)
        # Reshape and get first view features
        all_feats = flat_feats_and_cls.view(B, V, flat_feats_and_cls.size(1), -1)  # (B, V, 1+T, D)
        first_view_feats = all_feats[:, 0, 1:, :].detach()  # (B, T, D) # Discard the CLS token. Shape is (B, T, D)
        # Reshape to get the CLS token and features
        flat_tensors = all_feats[:, :, 1:, :].reshape(B * V, -1, all_feats.size(-1))  # → (B·V, T, D)
        flat_cls = all_feats[:, :, 0, :].reshape(B * V, all_feats.size(-1))  # → (B·V, D)
        # Reshape and expand the first view features
        flat_first_feats = first_view_feats.unsqueeze(1)  # (B, 1,  T, D)
        flat_first_feats = flat_first_feats.expand(-1, V, -1, -1)  # (B, V,  T, D)
        flat_first_feats = flat_first_feats.reshape(B * V, *first_view_feats.shape[1:])  # (B*V, T, D)
        # flat_first_feats = random_mask_tokens(flat_first_feats, mask_ratio=0.1)  # (B*V, T, D)
        # Get actions
        flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V

        # Run the conditional generator
        aug_tok, pad_mask = cond_generator.conditioning_network.aug_tokeniser(flat_actions)
        aug_tok = aug_tok +  cond_generator.conditioning_network.pe_aug(aug_tok.size(1), append_zeros_dim= cond_generator.conditioning_network.dim_linparam)     # (B,L,D)
        memory  =  cond_generator.conditioning_network.aug_enc(aug_tok, pad_mask)            # (B,L,D)
        memory  =  cond_generator.conditioning_network.aug_mlp_out(memory)                       # (B,L,d)

        tgt = cond_generator.conditioning_network.feature_mlp(flat_first_feats)
        tgt = tgt + cond_generator.conditioning_network.pe_img(tgt.size(1))                              # (B,196,d)

        layer_idx = 0
        dec_layer = cond_generator.conditioning_network.transformer_decoder.layers[layer_idx]

        attn_out, attn_w = dec_layer.multihead_attn(
                                        query=tgt, key=memory, value=memory,
                                        key_padding_mask=pad_mask,          # memory length > 1 ( Put None here is memory length = 1 )
                                        need_weights=True, attn_mask=None
                                    )
        # attn_w is already (B, P, S)
        attn_w = attn_w.detach().cpu()            # (B, P, S)

    # a) visualize first example’s heatmap
    import seaborn as sns
    num_episodes = 4
    for img_index in range(args.num_views*num_episodes):
        # Get actions applied in the episode
        action_ = flat_actions[img_index]
        action_names = [action_[i][0] for i in range(len(action_))]
        sns.heatmap(attn_w[img_index],
                    cmap='viridis',
                    cbar_kws={'label': 'attention weight'})
        plt.xlabel("aug token index")
        plt.ylabel("patch index")
        plt.title(f"cross-attention (patch × aug_token)\n{action_names}")
        plt.savefig(os.path.join(args.save_dir, f'cross_attention_heatmap_episode_{img_index // args.num_views}_view_{img_index % args.num_views}.png'))
        plt.close()

        # for action_index in range(len(action_)):
        #     # Reshape the patches from P to pxp (where P is 196 and pxp is 14x14)
        #     # Then show the attention heatmap on the patches

        #     # Get the attention map for the specific action
        #     attn_img_act = attn_w[img_index][:,action_index] # (P) 

        #     # Reshape the attention weights to a square grid (pxpxs)
        #     patch_size = int(attn_w[img_index].shape[0]**0.5)  # Assuming square patches
        #     attn_img_act = attn_img_act.reshape(patch_size, patch_size).detach().cpu().numpy()
        #     # Normalize the attention weights to [0, 1]
        #     # attn_img_act = (attn_img_act - attn_img_act.min()) / (attn_img_act.max() - attn_img_act.min())
        #     # Create a heatmap
        #     plt.imshow(attn_img_act, cmap='viridis', interpolation='nearest')
        #     plt.colorbar(label='Attention Weight')
        #     plt.title(f"Attention Heatmap for Action: {action_names[action_index]}\nEpisode {img_index // args.num_views}, View {img_index % args.num_views}")
        #     plt.xlabel("Patch X")
        #     plt.ylabel("Patch Y")
        #     plt.savefig(os.path.join(args.save_dir, f'cross_attention_heatmap_episode_{img_index // args.num_views}_view_{img_index % args.num_views}_action_{action_index}.png'))
        #     plt.close()
            

    # b) quantify patch-variance
    # var over Patches dim=1, then mean over batch & aug_tokens
    patch_var = attn_w.var(dim=1).mean().item()
    print(f"Avg attention variance across patches: {patch_var:.5f}")

    ### Plot reconstructions examples ###
    if args.local_rank == 0:
        view_encoder.eval()
        cond_generator.eval()
        episodes_plot_imgs = episodes_trainset[0][:num_episodes].to(device, non_blocking=True)
        episodes_plot_actions = episodes_trainset[1][:num_episodes]
        episodes_plot_gen_imgs = torch.empty(0)
        with torch.no_grad():
            first_view_tensors = view_encoder(episodes_plot_imgs[:,0])[:, 1:, :] # Discard the CLS token. Shape is (B, T, D)
            for v in range(args.num_views):
                actions = [episodes_plot_actions[j][v] for j in range(episodes_plot_imgs.shape[0])]
                gen_images, _ = cond_generator(first_view_tensors, actions)
                episodes_plot_gen_imgs = torch.cat([episodes_plot_gen_imgs, gen_images.unsqueeze(1).detach().cpu()], dim=1)
        episodes_plot_imgs = episodes_plot_imgs.detach().cpu()
        # plot each episode
        for i in range(num_episodes):
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
            image_name = f'episode_{i}.png'
            save_plot_dir = os.path.join(args.save_dir, 'gen_plots')
            # create folder if it doesn't exist
            if not os.path.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            grid.save(os.path.join(save_plot_dir, image_name))

    return None

if __name__ == '__main__':
    main()