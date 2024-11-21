import argparse
import os, time

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental, InstanceIncremental

from models import *
from loss_functions import SwapLossViewExpanded, CrossCosineSimilarityExpanded, EntropyRegularizerExpanded, KoLeoLossViewExpanded
from augmentations import Episode_Transformations
from wake_sleep_trainer import Wake_Sleep_trainer

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

# turn off warnings
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSCL Pseudolabeling Wake-Sleep latent pretrained encoder')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--iid', action='store_true')
# View encoder parameters
parser.add_argument('--pretrained_enc_path', type=str, default='pretrained_models/MIRA_episodic_offline_IN10_encGNWS_projBNWS_100epochs_12views_0.1lr_128bs_seed0/encoder_epoch100.pth')
parser.add_argument('--enc_model_name', type=str, default='resnet18')
# Semantic memory parameters
parser.add_argument('--sm_lr', type=float, default=0.01) #  LARS: 0.15, 0.3 # AdamW 0.001
parser.add_argument('--sm_wd', type=float, default=1e-6)
parser.add_argument('--proj_dim', type=int, default=2048)
parser.add_argument('--tau_t', type=float, default=0.225)
parser.add_argument('--tau_s', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.75)
parser.add_argument('--num_pseudoclasses', type=int, default=10)
# Training parameters
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=12) 
parser.add_argument('--num_episodes_per_sleep', type=int, default=12800*5) # 12800 *5 comes from number of types of augmentations
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/run_CSSL")
parser.add_argument('--data_order_path', type=str, default='data_class_order/IN10')
parser.add_argument('--data_order_file_name', type=str, default='data_class_order_seed0.txt')
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

    ### Create data order directory
    if not os.path.exists(args.data_order_path):
        os.makedirs(args.data_order_path)
        
    ### Print and save args
    print(args)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Seed everything
    seed_everything(seed=args.seed)

    ### Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

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

    data_order_file = os.path.join(args.data_order_path, args.data_order_file_name)
    if not os.path.exists(data_order_file):
        print(f'\n==> Creating data class order and saving on {data_order_file}...')
        data_class_order = np.random.permutation(args.num_classes) + 1
        np.savetxt(os.path.join(data_order_file), 
                data_class_order, fmt='%d')
    else:
        print(f'\n==> Loading data class order from file {data_order_file}...')
        data_class_order = np.loadtxt(data_order_file, dtype=int)
    
    data_class_order = list(data_class_order)

    # Create tasks
    if args.iid: # iid data
        train_tasks = InstanceIncremental(train_parent_dataset, 
                                          nb_tasks=args.num_tasks, 
                                          transformations=train_tranform)
        val_tasks = InstanceIncremental(val_parent_dataset, 
                                        nb_tasks=args.num_tasks, 
                                        transformations=val_transform)
    else: # non-iid data (class incremental)
        assert args.num_classes % args.num_tasks == 0, "Number of classes must be divisible by number of tasks"
        args.class_increment = int(args.num_classes // args.num_tasks)
        train_tasks = ClassIncremental(train_parent_dataset, 
                                       increment = args.class_increment, 
                                       transformations = train_tranform, 
                                       class_order = data_class_order)
        val_tasks = ClassIncremental(val_parent_dataset, 
                                     increment = args.class_increment, 
                                     transformations = val_transform, 
                                     class_order = data_class_order)

    ### Load view_encoder and semantic_memory
    print('\n==> Preparing model...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True)
    semantic_memory = eval('Semantic_Memory_Model')(view_encoder.fc.weight.shape[1], num_pseudoclasses = args.num_pseudoclasses, proj_dim = args.proj_dim )
    print('\nView encoder')
    print(view_encoder)
    print('\nSemantic Memory')
    print(semantic_memory)
    print('\n')

    ### Load pretrained view encoder
    encoder_state_dict = torch.load(args.pretrained_enc_path)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(encoder_state_dict, strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    view_encoder.fc = torch.nn.Identity()
    # freeze view_encoder
    for param in view_encoder.parameters():
        param.requires_grad = False

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    semantic_memory = torch.nn.DataParallel(semantic_memory)
    view_encoder = view_encoder.to(device)
    semantic_memory = semantic_memory.to(device)

    ### Define wake-sleep trainer
    WS_trainer = Wake_Sleep_trainer(view_encoder, semantic_memory, episode_batch_size=args.episode_batch_size, args=args)

    ### Loop over tasks
    print('\n==> Start wake-sleep training')
    init_time = time.time()
    saved_metrics = {'Train_metrics':{}, 'Val_metrics_seen_data':{}, 'Val_metrics_all_data':{}}
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
        del train_dataset, train_loader

        ### SLEEP PHASE ###
        print("Sleep Phase...")
        param_groups = [
            {'params': semantic_memory.parameters(), 'lr': args.sm_lr, 'weight_decay': args.sm_wd}
            ]
        optimizer = torch.optim.AdamW(param_groups, lr = 0, weight_decay = 0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr = [args.sm_lr], 
                                                        steps_per_epoch = args.num_episodes_batch_per_sleep, 
                                                        epochs = 1)#,
                                                        # pct_start=0.02)
        criterion_crossentropyswap = SwapLossViewExpanded(num_views = args.num_views).to(device)
        criterion_crosscosinesim = CrossCosineSimilarityExpanded(num_views = args.num_views).to(device)
        critetion_entropyreg = EntropyRegularizerExpanded(num_views = args.num_views).to(device)
        criterion_koleo = KoLeoLossViewExpanded(num_views = args.num_views).to(device)
        train_metrics = WS_trainer.sleep_phase(num_episodes_per_sleep = args.num_episodes_per_sleep,
                                                    optimizer = optimizer, 
                                                    criterions = [criterion_crossentropyswap,
                                                                    criterion_crosscosinesim,
                                                                    critetion_entropyreg,
                                                                    criterion_koleo,
                                                                    ],
                                                    scheduler = scheduler,
                                                    classes_list = data_class_order,
                                                    writer = writer, 
                                                    task_id = task_id)

        ### Evaluate model on validation set (seen so far)
        val_dataset = val_tasks[:task_id+1]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        print("\nEvaluate model on seen validation data...")
        val_seendata_metrics = WS_trainer.evaluate_model(val_loader,
                                                        plot_clusters = True, 
                                                        save_dir_clusters = os.path.join(args.save_dir,'pseudo_classes_clusters_seen_data'), 
                                                        task_id = task_id, 
                                                        mean = args.mean, 
                                                        std = args.std, 
                                                        calc_cluster_acc = False)
        
        ### Evaluate model on validation set (all data)
        val_dataset = val_tasks[:]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        print('\nEvaluating model on all validation data...')
        val_alldata_metrics = WS_trainer.evaluate_model(val_loader,
                                                        plot_clusters = True, 
                                                        save_dir_clusters = os.path.join(args.save_dir,'pseudo_classes_clusters_all_data'), 
                                                        task_id = task_id,
                                                        mean = args.mean,
                                                        std = args.std,
                                                        calc_cluster_acc = args.num_classes==args.num_pseudoclasses)
        
        ### Save metrics
        saved_metrics['Train_metrics'][f'Task_{task_id}'] = train_metrics
        saved_metrics['Val_metrics_seen_data'][f'Task_{task_id}'] = val_seendata_metrics
        saved_metrics['Val_metrics_all_data'][f'Task_{task_id}'] = val_alldata_metrics
        with open(os.path.join(args.save_dir, 'saved_metrics.json'), 'w') as f:
            json.dump(saved_metrics, f, indent=2)

        ### Save semantic memory
        semantic_memory_state_dict = semantic_memory.module.state_dict()
        torch.save(semantic_memory_state_dict, os.path.join(args.save_dir, f'semantic_memory_taskid_{task_id}.pth'))

        ### Print time
        print(f'Task {task_id} Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    # Close tensorboard writer
    writer.close()

    print('\n==> END')

    return None


if __name__ == '__main__':
    main()