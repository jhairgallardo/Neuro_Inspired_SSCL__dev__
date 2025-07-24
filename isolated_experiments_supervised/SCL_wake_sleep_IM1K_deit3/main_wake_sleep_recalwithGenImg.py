import argparse
import os, time

import torch
import torch.distributed
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torchvision

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental

from models_deit3_projcos import *

# from augmentations import Episode_Transformations, collate_function
# from wake_sleep_trainer import Wake_Sleep_trainer, eval_classification_performance

from augmentationsV2 import Episode_Transformations, collate_function
from wake_sleep_trainer_logan import Wake_Sleep_trainer, eval_classification_performance

from utils import MetricLogger, accuracy, time_duration_print, file_broadcast_tensor, file_broadcast_list, plot_generated_images_hold_set

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

parser = argparse.ArgumentParser(description='Wake-Sleep Training - Supervised')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet2012')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--num_pretraining_classes', type=int, default=10)#10 #100) # initial increment
parser.add_argument('--class_increment', type=int, default=5) #5 #100 # increment per task after pretraining
parser.add_argument('--num_tasks_to_run', type=int, default=5) #5 #10  # None, number of tasks to run trainining for (optional, only for testing)
parser.add_argument('--data_order_file_name', type=str, default='./IM1K_data_class_orders/imagenet_class_order_siesta.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Pre-trained folder
parser.add_argument('--pretrained_folder', type=str, default='./output/Pretrained_condgen_AND_enc/projcosOPTI_augmV2_augaggV0normal64d_antialias_reflect_biastrue_3tanh_deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.0008wd@0.05droppath@0.0125_CONDGEN_lr@0.0008wd@0layers@8heads@8dimff@1024dropout@0_seed@0')

# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--enc_model_checkpoint', type=str, default='view_encoder_epoch99.pth')
parser.add_argument('--enc_lr', type=float, default=0.0003)
parser.add_argument('--enc_wd', type=float, default=0.05)
parser.add_argument('--drop_path', type=float, default=0.0125) # 0.0125 for tiny, 0.05 for small, 0.2 for base
# Classifier parameters
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
parser.add_argument('--classifier_model_checkpoint', type=str, default='classifier_epoch99.pth')
parser.add_argument('--classifier_lr', type=float, default=0.0003)
parser.add_argument('--classifier_wd', type=float, default=0)
# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--condgen_model_checkpoint', type=str, default='cond_generator_epoch99.pth')
parser.add_argument('--condgen_lr', type=float, default=0.0003)
parser.add_argument('--condgen_wd', type=float, default=0)
parser.add_argument('--cond_dropout', type=float, default=0)
# Training parameters
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--num_episodes_per_sleep', type=int, default=128000)# 128000 670000
parser.add_argument('--episode_batch_size', type=int, default=80) # 16 80
parser.add_argument('--patience', type=int, default=40)
parser.add_argument('--threshold_NREM', type=float, default=1e-3)
parser.add_argument('--threshold_REM', type=float, default=1e-3)
parser.add_argument('--window', type=int, default=50)
parser.add_argument('--smooth_loss_alpha', type=float, default=0.3)
parser.add_argument('--sampling_method', type=str, default='uniform', choices=['uniform', 'uniform_class_balanced', 'GRASP']) # uniform, random, sequential
parser.add_argument('--logan', action='store_true', help='Use LOGAN for action code optimization')
# Other parameters
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/wake_sleep_recalwithGenImg/run_debug")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations.
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_ep_plot', type=int, default=10) # number of episodes to plot per task (for debugging purposes)
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

    ### Load pre-trained model args
    args_pretrained = json.load(open(os.path.join(args.pretrained_folder, 'args.json'), 'r'))
    args.img_num_tokens = args_pretrained['img_num_tokens']
    args.cond_num_layers = args_pretrained['cond_num_layers']
    args.cond_nhead = args_pretrained['cond_nhead']
    args.cond_dim_ff = args_pretrained['cond_dim_ff']
    args.aug_feature_dim = args_pretrained['aug_feature_dim']
    args.aug_num_tokens_max = args_pretrained['aug_num_tokens_max']
    args.aug_n_layers = args_pretrained['aug_n_layers']
    args.aug_n_heads = args_pretrained['aug_n_heads']
    args.aug_dim_ff = args_pretrained['aug_dim_ff']
    args.upsampling_num_Blocks = args_pretrained['upsampling_num_Blocks']
    args.upsampling_num_out_channels = args_pretrained['upsampling_num_out_channels']

    ### DDP init
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        args.ddp = True
        print(f"DDP used, local rank set to {args.local_rank}. {torch.distributed.get_world_size()} GPUs training.")
        torch.distributed.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.local_rank = 0
        args.ddp = False
        print("DDP not used, local rank set to 0. 1 GPU training.")

    args.is_main = (args.local_rank == 0)

    # Create save dir folders and save args
    if args.is_main:
        print(args)
        if not os.path.exists(args.save_dir): # create save dir
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    ### Calculate batch size, workers, and num_episodes_per_sleep per GPU
    if args.ddp:
        args.episode_batch_size = int(args.episode_batch_size / torch.distributed.get_world_size())
        args.workers = int(args.workers / torch.distributed.get_world_size())
        args.num_episodes_per_sleep = int(args.num_episodes_per_sleep / torch.distributed.get_world_size())
    
    ### Bugdet of batch episodes per sleep (for schedulers)
    args.num_batch_episodes_per_sleep = int(np.ceil(args.num_episodes_per_sleep / args.episode_batch_size))

    ### Seed everything
    final_seed = args.seed + args.local_rank
    seed_everything(seed=final_seed)

    ### Define tensoboard writer
    if args.is_main:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))
    else:
        writer = None

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
    if args.is_main:
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

    ### Load class order from file
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

    ### Create tasks
    train_tasks = ClassIncremental(train_dataset_continuum, increment=args.class_increment, initial_increment=args.num_pretraining_classes, transformations=[train_transform], class_order=data_class_order)
    val_tasks = ClassIncremental(val_dataset_continuum, increment=args.class_increment, initial_increment=args.num_pretraining_classes, transformations=[val_transform], class_order=data_class_order)

    ### Define num_tasks_to_tun if not set
    if args.num_tasks_to_run is None:
        args.num_tasks_to_run = len(train_tasks)

    ### Load Models
    if args.is_main:
        print('\n==> Prepare models...')
    view_encoder = eval(args.enc_model_name)(drop_path_rate=args.drop_path, output_before_pool=True)
    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.embed_dim, num_classes=args.num_classes)
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
    view_encoder.head = torch.nn.Identity() # remove the head of the encoder
    # Load pre-trained weights if available
    if args.enc_model_checkpoint is not None:
        if args.is_main:
            print(f'Loading view encoder from {args.pretrained_folder}/{args.enc_model_checkpoint}')
        view_encoder.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.enc_model_checkpoint), map_location=device), strict=True)
    if args.classifier_model_checkpoint is not None:
        if args.is_main:
            print(f'Loading classifier from {args.pretrained_folder}/{args.classifier_model_checkpoint}')
        classifier.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.classifier_model_checkpoint), map_location=device), strict=True)
    if args.condgen_model_checkpoint is not None:
        if args.is_main:
            print(f'Loading conditional generator from {args.pretrained_folder}/{args.condgen_model_checkpoint}')
        cond_generator.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.condgen_model_checkpoint), map_location=device), strict=True)
                                                  
    ### Print models
    if args.is_main:
        print('\nView encoder')
        print(view_encoder)
        print('\nClassifier')
        print(classifier)
        print('\nConditional generator')
        print(cond_generator)
        print('\n')

    ### Move models to device. If distributed data parallel (DDP) is not used, use data parallel
    if args.ddp:
        view_encoder   = view_encoder.to(device)
        classifier     = classifier.to(device)
        cond_generator = cond_generator.to(device)
        view_encoder   = torch.nn.SyncBatchNorm.convert_sync_batchnorm(view_encoder)
        view_encoder   = torch.nn.parallel.DistributedDataParallel(view_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        classifier     = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        classifier     = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
        cond_generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cond_generator)
        cond_generator = torch.nn.parallel.DistributedDataParallel(cond_generator, device_ids=[args.local_rank], output_device=args.local_rank)
    else: # Single GPU training
        view_encoder   = view_encoder.to(device)
        classifier     = classifier.to(device)
        cond_generator = cond_generator.to(device)

    ### Define wake-sleep trainer engine ###
    WS_trainer = Wake_Sleep_trainer(episode_batch_size = args.episode_batch_size,
                                    num_episodes_per_sleep = args.num_episodes_per_sleep,
                                    num_views = args.num_views,
                                    dataset_mean = args.mean,
                                    dataset_std = args.std,
                                    patience=args.patience,
                                    threshold_nrem=args.threshold_NREM,
                                    threshold_rem=args.threshold_REM,
                                    window=args.window,
                                    smooth_loss_alpha=args.smooth_loss_alpha,
                                    device = device,
                                    save_dir = args.save_dir,
                                    print_freq = args.print_frequency)
    
    ### Load criterion
    criterion_sup = torch.nn.CrossEntropyLoss()
    criterion_condgen = torch.nn.MSELoss()

    ### Save one batch for plot purposes (all tasks)
    seed_everything(seed=final_seed)  # Seed for reproducibility of the plot
    if args.is_main:
        episodes_plot_dict = {}
        for i in range(args.num_tasks_to_run):
            train_loader_aux = torch.utils.data.DataLoader(train_tasks[i], batch_size=args.num_ep_plot, shuffle=True, collate_fn=collate_function)
            episodes_plot, _, _ = next(iter(train_loader_aux))
            episodes_plot_dict[f"taskid_{i}"] = episodes_plot
            del train_loader_aux
    



    ##########################################################################
    ### Initialize episodic memory with the first task (pre-training data) ###
    ##########################################################################
    # This is done by doing one wake phase before starting the wake-sleep training
    # Wake phase is done only with rank=0. Then, I broadcast the episodic memory to all ranks. No sampler is needed for this.
    if args.is_main:
        print('\n==> Initializing episodic memory with the first task (pre-training data)...')
        print(f'==> Number of classes in the first task: {args.num_pretraining_classes}')
        train_loader_current = torch.utils.data.DataLoader(
            train_tasks[0], 
            batch_size=args.episode_batch_size, 
            shuffle=True, 
            num_workers=args.workers, 
            collate_fn=collate_function, 
            pin_memory=True
        )
        wake_view_encoder = view_encoder.module if args.ddp else view_encoder  # Get the module if DDP
        new_tensors_paths, new_labels_paths, new_actions_paths = WS_trainer.wake_phase(wake_view_encoder, train_loader_current)
        del train_loader_current
    else:
        new_tensors_paths, new_labels_paths, new_actions_paths = None, None, None
    # Broadcast new adquired data to all ranks if ddp
    if args.ddp: 
        torch.distributed.barrier()  # Wait for all processes
        new_tensors_paths = file_broadcast_list(new_tensors_paths, os.path.join(args.save_dir, 'new_tensors_paths.pt'), args.local_rank)
        new_labels_paths  = file_broadcast_list(new_labels_paths,  os.path.join(args.save_dir, 'new_labels_paths.pt'), args.local_rank)
        new_actions_paths = file_broadcast_list(new_actions_paths, os.path.join(args.save_dir, 'new_actions_paths.pt'), args.local_rank)
        torch.distributed.barrier()  # Wait for all processes
    # Append new adquired data to the episodic memory
    WS_trainer.append_memory(new_tensors_paths, new_labels_paths, new_actions_paths)

    # Plot generated images with pre-trained model (before wake-sleep training)
    if args.is_main:
        aux_view_encoder = view_encoder.module if args.ddp else view_encoder  # Get the module if DDP
        aux_cond_generator = cond_generator.module if args.ddp else cond_generator
        plot_generated_images_hold_set(aux_view_encoder, aux_cond_generator, episodes_plot_dict, 0, 
                                        args.mean, args.std, args.num_views, args.save_dir, device)
            




    ###########################################
    #### Wake-Sleep Learning with NREM-REM ####
    ###########################################

    if args.is_main:
        print(f'\n==> Number of tasks to run after first task: {args.num_tasks_to_run-1}')
        print(f'==> Number of classes per task: {args.class_increment}')

    if args.is_main:
        print('\n==> Start Training on incoming tasks')
    init_time = time.time()
    scaler = GradScaler()

    #### TASKS LOOP ####
    for task_id in range(1, args.num_tasks_to_run):
        task_start_time = time.time()

        # Get validation set for evaluation
        val_seen_dataset = val_tasks[:task_id+1]  # All tasks seen so far
        val_loader_seen_tasks = torch.utils.data.DataLoader(
            val_seen_dataset,
            batch_size=args.episode_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(val_seen_dataset, shuffle=False) if args.ddp else None
        )
        val_baseinit_dataset = val_tasks[0]
        val_loader_baseinit_task = torch.utils.data.DataLoader(
            val_baseinit_dataset,
            batch_size=args.episode_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(val_baseinit_dataset, shuffle=False) if args.ddp else None
        )
        val_current_dataset = val_tasks[task_id]  # Current task validation set
        val_loader_current_task = torch.utils.data.DataLoader(
            val_current_dataset,
            batch_size=args.episode_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(val_current_dataset, shuffle=False) if args.ddp else None
        )

        if args.is_main:
            print(f"\n\n\n##------ Learning Task {task_id}/{args.num_tasks_to_run-1} ------##")
            if args.ddp: total_num_seen_episodes_reduced = WS_trainer.total_num_seen_episodes*torch.distributed.get_world_size()
            else: total_num_seen_episodes_reduced = WS_trainer.total_num_seen_episodes
            writer.add_scalar('Task_boundaries', task_id, total_num_seen_episodes_reduced)

        day=1 # For now I am only doing 1 day per task, but this can be changed later if needed. (I would introduce a loop for days)

        if args.is_main:
            print(f"\n##### Day {day} #####")

        #----------------#
        ### WAKE PHASE ###
        #----------------#
        if args.is_main:
            print(f"\n=== WAKE PHASE ===")
            train_loader_current_task = torch.utils.data.DataLoader(
                train_tasks[task_id],
                batch_size=args.episode_batch_size,
                shuffle=True,
                num_workers=args.workers,
                collate_fn=collate_function,
                pin_memory=True
            )
            wake_view_encoder = view_encoder.module if args.ddp else view_encoder  # Get the module if DDP
            new_tensors_paths, new_labels_paths, new_actions_paths = WS_trainer.wake_phase(wake_view_encoder, train_loader_current_task)
            del train_loader_current_task
        else:
            new_tensors_paths, new_labels_paths, new_actions_paths = None, None, None
        # Broadcast new adquired data to all ranks if ddp
        if args.ddp: 
            torch.distributed.barrier()  # Wait for all processes
            new_tensors_paths = file_broadcast_list(new_tensors_paths, os.path.join(args.save_dir, 'new_tensors_paths.pt'), args.local_rank)
            new_labels_paths  = file_broadcast_list(new_labels_paths,  os.path.join(args.save_dir, 'new_labels_paths.pt'), args.local_rank)
            new_actions_paths = file_broadcast_list(new_actions_paths, os.path.join(args.save_dir, 'new_actions_paths.pt'), args.local_rank)
            torch.distributed.barrier()  # Wait for all processes
        # Append new adquired data to the episodic memory
        WS_trainer.append_memory(new_tensors_paths, new_labels_paths, new_actions_paths)
        # Reset sleep counter to start sleep session
        WS_trainer.reset_sleep_counter()


        #-----------------#
        ### SLEEP PHASE ###
        #-----------------#
        if args.is_main:
            print(f"\n=== SLEEP PHASE ===")
        optimizer_encoder = torch.optim.AdamW(view_encoder.parameters(), lr=args.enc_lr, weight_decay=args.enc_wd)
        optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_wd)
        optimizer_condgen = torch.optim.AdamW(cond_generator.parameters(), lr=args.condgen_lr, weight_decay=args.condgen_wd)
        scheduler_encoder = OneCycleLR(optimizer_encoder, max_lr=args.enc_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        scheduler_classifier = OneCycleLR(optimizer_classifier, max_lr=args.classifier_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        scheduler_condgen = OneCycleLR(optimizer_condgen, max_lr=args.condgen_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)

        ### Sample indexes
        WS_trainer.sampling_idxs_for_sleep(args.num_episodes_per_sleep, sampling_method=args.sampling_method)
        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes

        ### NREM-REM cycles ###
        nrem_rem_cycle_counter = 1 # For printing purposes
        while WS_trainer.sleep_episode_counter < args.num_episodes_per_sleep:

            # ### Sample indexes
            # WS_trainer.sampling_idxs_for_sleep(args.num_episodes_per_sleep, sampling_method=args.sampling_method)
            # if args.ddp:
            #     torch.distributed.barrier()  # Wait for all processes

            ### NREM 
            if args.is_main:
                print(f"=== NREM - sleep cycle {nrem_rem_cycle_counter}")
            WS_trainer.NREM_sleep(view_encoder = view_encoder,
                                  classifier = classifier, 
                                  cond_generator = cond_generator,
                                  optimizers = [optimizer_encoder, optimizer_classifier, optimizer_condgen], 
                                  schedulers = [scheduler_encoder, scheduler_classifier, scheduler_condgen],
                                  criterions = [criterion_sup, criterion_condgen],
                                  task_id = task_id,
                                  scaler = scaler,
                                  writer = writer,
                                  is_main = args.is_main,
                                  ddp = args.ddp,
                                  mean = args.mean,
                                  std = args.std,
                                  save_dir = args.save_dir)
            if args.is_main:
                print(f'Validation metrics')
            eval_classification_performance(view_encoder, classifier, val_loader_seen_tasks, criterion_sup, writer, 
                                            "Val_Seen", args.ddp, args.is_main, device, WS_trainer.total_num_seen_episodes)
            eval_classification_performance(view_encoder, classifier, val_loader_baseinit_task, criterion_sup, writer,
                                            "Val_BaseInit", args.ddp, args.is_main, device, WS_trainer.total_num_seen_episodes)
            eval_classification_performance(view_encoder, classifier, val_loader_current_task, criterion_sup, writer,
                                            "Val_Current", args.ddp, args.is_main, device, WS_trainer.total_num_seen_episodes)
            
            # ### Sample indexes
            # WS_trainer.sampling_idxs_for_sleep(args.num_episodes_per_sleep, sampling_method=args.sampling_method)
            # if args.ddp:
            #     torch.distributed.barrier()  # Wait for all processes
            
            ### REM
            if args.is_main:
                print(f"=== REM - sleep cycle {nrem_rem_cycle_counter}")
            WS_trainer.REM_sleep(view_encoder = view_encoder,
                                classifier = classifier, 
                                cond_generator = cond_generator,
                                optimizers = [optimizer_encoder, optimizer_classifier, optimizer_condgen], 
                                schedulers = [scheduler_encoder, scheduler_classifier, scheduler_condgen],
                                criterions = [criterion_sup, criterion_condgen],
                                task_id = task_id,
                                scaler = scaler,
                                writer = writer,
                                is_main = args.is_main,
                                ddp = args.ddp,
                                mean = args.mean,
                                std = args.std,
                                save_dir = args.save_dir,
                                logan_flag=args.logan
                                )
            if args.is_main:
                print(f'Validation metrics')
            eval_classification_performance(view_encoder, classifier, val_loader_seen_tasks, criterion_sup, writer, 
                                            "Val_Seen", args.ddp, args.is_main, device, WS_trainer.total_num_seen_episodes)
            eval_classification_performance(view_encoder, classifier, val_loader_baseinit_task, criterion_sup, writer,
                                            "Val_BaseInit", args.ddp, args.is_main, device, WS_trainer.total_num_seen_episodes)
            eval_classification_performance(view_encoder, classifier, val_loader_current_task, criterion_sup, writer,
                                            "Val_Current", args.ddp, args.is_main, device, WS_trainer.total_num_seen_episodes)
            # Helper to avoid representational drift (do it after a phase that updates the view encoder)
            # Here, I will update the episodic_memory_tensors by generating their inputs again and pass them through view encoder
            if args.is_main:
                print(f"=== Recalculate episodic memory with Generated Images")
                aux_view_encoder = view_encoder.module if args.ddp else view_encoder  # Get the module if DDP
                aux_cond_generator = cond_generator.module if args.ddp else cond_generator  # Get the module if DDP
                WS_trainer.recalculate_episodic_memory_with_gen_imgs(aux_view_encoder, aux_cond_generator)
            # Broadcast the new episodic memory tensors to all ranks
            if args.ddp:
                torch.distributed.barrier()  # Wait for all processes
        
            nrem_rem_cycle_counter += 1  # Increment cycle counter

        # Task finished, print time taken for the task
        if args.is_main:
            task_duration = time.time() - task_start_time
            print(f"\nTask {task_id} finished in {time_duration_print(task_duration)}")
            print(f"Total time taken so far: {time_duration_print(time.time() - init_time)}")

        # Save models after each task
        if args.is_main:
            print(f"\n==> Saving models after task {task_id}...")
            torch.save(view_encoder.module.state_dict() if args.ddp else view_encoder.state_dict(), os.path.join(args.save_dir, f'view_encoder_task{task_id}.pth'))
            torch.save(classifier.module.state_dict() if args.ddp else classifier.state_dict(), os.path.join(args.save_dir, f'classifier_task{task_id}.pth'))
            torch.save(cond_generator.module.state_dict() if args.ddp else cond_generator.state_dict(), os.path.join(args.save_dir, f'cond_generator_task{task_id}.pth'))
            print("Models saved.")

        # Plot
        if args.is_main:
            aux_view_encoder = view_encoder.module if args.ddp else view_encoder  # Get the module if DDP
            aux_cond_generator = cond_generator.module if args.ddp else cond_generator
            plot_generated_images_hold_set(aux_view_encoder, aux_cond_generator, episodes_plot_dict, task_id, 
                                           args.mean, args.std, args.num_views, args.save_dir, device)
            

if __name__ == '__main__':
    main()