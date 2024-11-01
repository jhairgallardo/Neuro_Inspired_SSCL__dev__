import argparse
import os, time

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import math
from typing import List

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental, InstanceIncremental

from models import *
from optimizer import LARS
from loss_functions import SwapLossViewExpanded, ConsistLossCARLViewExpanded, KoLeoLossViewExpanded, ConsistLossMSEViewExpanded
from augmentations import Episode_Transformations
from wake_sleep_trainer import Wake_Sleep_trainer
import utils

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

# turn off warnings
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSCL Pseudolabeling Wake-Sleep veridical')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--iid', action='store_true')

parser.add_argument('--model_name', type=str, default='resnet18')
parser.add_argument('--proj_dim', type=int, default=2048)
parser.add_argument('--num_pseudoclasses', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.01) #  LARS: 0.15, 0.3 # AdamW 0.001
parser.add_argument('--wd', type=float, default=1e-6)
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=12) 
parser.add_argument('--num_episodes_per_sleep', type=int, default=12800*5) # 12800 *5 comes from number of types of augmentations

parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/run_CSSL")
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
    args.num_episodes_batch_per_sleep = int(np.ceil(args.num_episodes_per_sleep/args.episode_batch_size))
    
    ### Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    ### Print and save args
    print(args)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Seed everything
    seed_everything(seed=args.seed)

    ### Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Define tensoboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load all data tasks
    print('\n==> Preparing data...')
    episode_transform = Episode_Transformations(num_views = args.num_views)
    args.mean = episode_transform.mean
    args.std = episode_transform.std
    train_tranform = [episode_transform]
    val_transform = [transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = args.mean, std = args.std)])]
    train_parent_dataset = ImageFolderDataset(data_path = os.path.join(args.data_path, 'train'))
    val_parent_dataset = ImageFolderDataset(data_path = os.path.join(args.data_path, 'val'))
    data_class_order = list(np.arange(args.num_classes)+1)
    if args.iid: # iid data
        train_tasks = InstanceIncremental(train_parent_dataset, 
                                          nb_tasks=args.num_tasks, 
                                          transformations=train_tranform)
        val_tasks = InstanceIncremental(val_parent_dataset, 
                                        nb_tasks=args.num_tasks, 
                                        transformations=val_transform)
    else: # non-iid data (class incremental)
        assert args.num_classes % args.num_tasks == 0, "Number of classes must be divisible by number of tasks"
        class_increment = int(args.num_classes // args.num_tasks)
        train_tasks = ClassIncremental(train_parent_dataset, 
                                       increment = class_increment, 
                                       transformations = train_tranform, 
                                       class_order = data_class_order)
        val_tasks = ClassIncremental(val_parent_dataset, 
                                     increment = class_increment, 
                                     transformations = val_transform, 
                                     class_order = data_class_order)

    ### Define SSL network model
    print('\n==> Preparing model...')
    encoder = eval(args.model_name)(zero_init_residual = True)
    model = eval('Semantic_Memory_Model')(encoder, 
                                          num_pseudoclasses = args.num_pseudoclasses, 
                                          proj_dim = args.proj_dim)
    print(model)

    ### Dataparallel and move model to device
    model = torch.nn.DataParallel(model)
    model.to(device)

    ### Define wake-sleep trainer
    WS_trainer = Wake_Sleep_trainer(model, episode_batch_size=args.episode_batch_size, args=args)

    ### Loop over tasks
    print('\n==> Start wake-sleep training')
    init_time = time.time()
    for task_id in range(len(train_tasks)):
        start_time = time.time()

        print(f"\n------ Task {task_id+1}/{len(train_tasks)} ------")

        ## Get tasks train loader
        train_dataset = train_tasks[task_id]
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size = args.episode_batch_size, 
                                                   shuffle = True, 
                                                   num_workers = args.workers, 
                                                   pin_memory = True, 
                                                   drop_last = True)
        
        ### WAKE PHASE ###
        print("Wake Phase...")
        WS_trainer.wake_phase(train_loader)
        del train_dataset#, train_loader

        ### SLEEP PHASE ###
        print("Sleep Phase...")
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.wd)
        criterion_crossentropyswap = SwapLossViewExpanded(num_views = args.num_views).to(device)
        criterion_consistencycarl = ConsistLossCARLViewExpanded(num_views = args.num_views).to(device)
        criterion_koleo = KoLeoLossViewExpanded(num_views = args.num_views).to(device)
        criterion_consistencymse = ConsistLossMSEViewExpanded(num_views = args.num_views).to(device)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr = args.lr, 
                                                        steps_per_epoch = args.num_episodes_batch_per_sleep, 
                                                        epochs = 1,
                                                        pct_start=0.02)
        WS_trainer.sleep_phase(num_episodes_per_sleep = args.num_episodes_per_sleep,
                               optimizer = optimizer, 
                               criterions = [criterion_crossentropyswap, 
                                             criterion_consistencycarl, 
                                             criterion_koleo,
                                             criterion_consistencymse], 
                               scheduler = scheduler,
                               device = device,
                               classes_list = data_class_order,
                               writer = writer, 
                               task_id = task_id)

        ### Evaluate model on validation set (seen so far)
        val_dataset = val_tasks[:task_id+1]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        print("\nEvaluate model on seen validation data...")
        WS_trainer.evaluate_model(val_loader, 
                                  device = device, 
                                  plot_clusters = True, 
                                  save_dir_clusters = os.path.join(args.save_dir,'pseudo_classes_clusters_seen_data'), 
                                  task_id = task_id, 
                                  mean = args.mean, 
                                  std = args.std, 
                                  calc_cluster_acc = False,
                                  num_pseudoclasses = args.num_pseudoclasses)
        
        ### Evaluate model on validation set (all data)
        val_dataset = val_tasks[:]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        print('\nEvaluating model on all validation data...')
        WS_trainer.evaluate_model(val_loader, 
                                  device = device, 
                                  plot_clusters = True, 
                                  save_dir_clusters = os.path.join(args.save_dir,'pseudo_classes_clusters_all_data'), 
                                  task_id = task_id,
                                  mean = args.mean,
                                  std = args.std,
                                  calc_cluster_acc = args.num_classes==args.num_pseudoclasses,
                                  num_pseudoclasses = args.num_pseudoclasses)

        ### Save encoder
        encoder_state_dict = model.module.encoder.state_dict()
        torch.save(encoder_state_dict, os.path.join(args.save_dir, f'encoder_taskid_{task_id}.pth'))

        ### Print time
        print(f'Task {task_id} Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    # Close tensorboard writer
    writer.close()

    print('\n==> END')

    return None


if __name__ == '__main__':
    main()