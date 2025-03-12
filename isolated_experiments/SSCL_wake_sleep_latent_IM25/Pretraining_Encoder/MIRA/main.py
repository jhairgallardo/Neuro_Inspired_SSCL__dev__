import argparse
import os, time

import torch
import torch.nn.functional as F
from torchvision import transforms #### I may want to erase datasets when not using
from torch.cuda.amp import GradScaler

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental

from models import *
from loss_functions import SwapLossViewExpanded
from augmentations import Episode_Transformations
import utils

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

# turn off warnings
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-25')
parser.add_argument('--num_classes', type=int, default=25)
parser.add_argument('--num_pretraining_classes', type=int, default=5)
parser.add_argument('--data_order_file_name', type=str, default='./../IM25_data_class_orders/IM25_data_class_order0.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Network parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--semantic_model_name', type=str, default='Semantic_Memory_Model')
parser.add_argument('--lr', type=float, default=0.008)
parser.add_argument('--wd', type=float, default=1e-6)
parser.add_argument('--proj_dim', type=int, default=2048)
parser.add_argument('--out_dim', type=int, default=1024)
parser.add_argument('--tau_t', type=float, default=0.225)
parser.add_argument('--tau_s', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.75)
parser.add_argument('--num_pseudoclasses', type=int, default=25)
# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=12) 
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/run_encoder_pretraining")
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

@torch.no_grad()
def mira(k: torch.Tensor,
         tau: float,
         beta: float,
         iters: int):
    bs = k.size(0) #* dist.get_world_size()  # total batch-size

    # fixed point iteration
    k = F.softmax(k / tau / (1 - beta), dim=1)
    temp = k.sum(dim=0)
    # dist.all_reduce(temp)
    v = (temp / bs).pow(1 - beta)
    for _ in range(iters):
        temp = k / (v.pow(- beta / (1 - beta)) * k).sum(dim=1, keepdim=True)
        temp = temp.sum(dim=0)
        # dist.all_reduce(temp)
        v = (temp / bs).pow(1 - beta)
    temp = v.pow(- beta / (1 - beta)) * k
    target = temp / temp.sum(dim=1, keepdim=True)
    # if there is nan in the target, return k
    if torch.isnan(target).any():
        # error
        raise ValueError('Nan in target')
    return target

@torch.no_grad()
def mira_pseudolabeling(logits, num_views, tau, beta, iters):
    targets = torch.empty(0).to(logits.device)
    for t in range(num_views):
        targets_t = mira(logits[:,t], tau, beta, iters)
        targets = torch.cat([targets, targets_t.unsqueeze(1)], dim=1)
    return targets

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
    print('\n==> Preparing data...')
    episode_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=False)
    train_tranform = [episode_transform]
    train_parent_dataset = ImageFolderDataset(data_path = os.path.join(args.data_path, 'train'))
    print(f'\n==> Loading data class order from file {args.data_order_file_name}...')
    data_class_order = list(np.loadtxt(args.data_order_file_name, dtype=int))
    # only grab the first task (pretraining classes) --> [0]
    train_dataset = ClassIncremental(train_parent_dataset, increment=1, initial_increment=args.num_pretraining_classes, transformations=train_tranform, class_order=data_class_order)[0]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
    print(f'\n==> Classes being used for pretraining: {np.array(data_class_order)[list(train_dataset.get_classes())]}')
    del train_parent_dataset, train_dataset

    ### Load view_encoder and semantic_memory
    print('\n==> Preparing model...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    semantic_memory = eval(args.semantic_model_name)(view_encoder.fc.weight.shape[1], 
                                                     num_pseudoclasses = args.num_pseudoclasses, 
                                                     proj_dim = args.proj_dim,
                                                     out_dim = args.out_dim)
    view_encoder.fc = torch.nn.Identity() # remove last fc layer from view_encoder network (we don't train it)
    print('\nView encoder')
    print(view_encoder)
    print('\nSemantic Memory')
    print(semantic_memory)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    semantic_memory = torch.nn.DataParallel(semantic_memory)
    view_encoder = view_encoder.to(args.device)
    semantic_memory = semantic_memory.to(args.device)
    param_groups = [
                    {'params': view_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': semantic_memory.parameters(), 'lr': args.lr, 'weight_decay': args.wd}
                   ]

    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(param_groups, lr=0, weight_decay=0)
    criterion = SwapLossViewExpanded(num_views=args.num_views).to(args.device)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr*1e-6, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    ### Train loop
    print('\n==> Training model')
    init_time = time.time()
    for epoch in range(args.epochs):
        print(f'\n==> Epoch {epoch}/{args.epochs}')
        start_time = time.time()

        ## Train STEP
        total_loss=0
        view_encoder.train()
        semantic_memory.train()
        for i, (batch_episode_imgs, _, _) in enumerate(train_loader):
            batch_episode_imgs = batch_episode_imgs.to(args.device)
            batch_episode_logits = torch.empty(0).to(args.device)
            for v in range(args.num_views):
                batch_imgs = batch_episode_imgs[:,v]
                batch_features = view_encoder(batch_imgs)
                batch_logits = semantic_memory(batch_features)
                batch_episode_logits = torch.cat([batch_episode_logits, batch_logits.unsqueeze(1)], dim=1)
            batch_episode_labels = mira_pseudolabeling(logits = batch_episode_logits, 
                                                num_views = args.num_views,
                                                tau=args.tau_t, 
                                                beta=args.beta, 
                                                iters=30)
            loss = criterion(batch_episode_logits/args.tau_s, batch_episode_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % args.print_frequency == 0:
                ps = F.softmax(batch_episode_logits[:,0] / args.tau_s, dim=1).detach().cpu()
                pt = F.softmax(batch_episode_logits[:,0] / args.tau_t, dim=1).detach().cpu()
                _, _, mi_ps = utils.statistics(ps)
                _, _, mi_pt = utils.statistics(pt)
                writer.add_scalar('MI_ps', mi_ps, epoch*len(train_loader)+i)
                writer.add_scalar('MI_pt', mi_pt, epoch*len(train_loader)+i)
                print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- lr: {scheduler.get_last_lr()[0]:.6f} -- CrossEntropySwap Loss: {loss.item():.6f} -- MI_ps: {mi_ps:.6f} -- MI_pt: {mi_pt:.6f}')
            scheduler.step()
            writer.add_scalar('Loss (per batch)', loss.item(), epoch*len(train_loader)+i)
        total_loss /= len(train_loader)
        writer.add_scalar('Loss (per epoch)', total_loss, epoch)
        print(f'Epoch [{epoch}] Total Train Loss per Epoch: {total_loss:.6f}')
        print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

        ### Save model
        if (epoch+1) % 10 == 0 or epoch==0:
            view_encoder_state_dict = view_encoder.module.state_dict()
            semantic_memory_state_dict = semantic_memory.module.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(semantic_memory_state_dict, os.path.join(args.save_dir, f'semantic_memory_epoch{epoch}.pth'))

    return None

if __name__ == '__main__':
    main()