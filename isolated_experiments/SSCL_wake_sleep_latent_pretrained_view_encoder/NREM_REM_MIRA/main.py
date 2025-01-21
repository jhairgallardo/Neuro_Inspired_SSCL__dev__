import argparse
import os, time

import torch
from torchvision import transforms, datasets #### I may want to erase datasets when not using
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental, InstanceIncremental

from models import *
from loss_functions import SwapLossViewExpanded
from augmentations import Episode_Transformations
from wake_sleep_trainer import Wake_Sleep_trainer

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

# turn off warnings
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSCL Pseudolabeling Wake-Sleep latent pretrained encoder conditional generator (NREM)')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--iid', action='store_true')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters
parser.add_argument('--pretrained_enc_path', type=str)#, default='pretrained_models/MIRA_episodic_offline_IN10_encGN_projGN_100epochs_12views_0.008lr_128bs_seed0_nolinfreeze/encoder_epoch100.pth')
parser.add_argument('--enc_model_name', type=str, default='resnet18')
# Generator parameters
parser.add_argument('--generator_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--generator_lr', type=float, default=1e-3)
parser.add_argument('--generator_wd', type=float, default=1e-2)
parser.add_argument('--dec_num_Blocks', type=list, default=[1,1,1,1])
parser.add_argument('--dec_num_out_channels', type=int, default=3)
parser.add_argument('--ft_feature_dim', type=int, default=512)
parser.add_argument('--ft_action_code_dim', type=int, default=11)
parser.add_argument('--ft_num_layers', type=int, default=2)
parser.add_argument('--ft_nhead', type=int, default=4)
parser.add_argument('--ft_dim_feedforward', type=int, default=256)
parser.add_argument('--ft_dropout', type=float, default=0.1)
# Semantic memory parameters
parser.add_argument('--semantic_model_name', type=str, default='Semantic_Memory_Model')
parser.add_argument('--sm_lr', type=float, default=0.01) #  LARS: 0.15, 0.3 # AdamW 0.001
parser.add_argument('--sm_wd', type=float, default=1e-6)
parser.add_argument('--proj_dim', type=int, default=2048)
parser.add_argument('--out_dim', type=int, default=1024)
parser.add_argument('--tau_t', type=float, default=0.225)
parser.add_argument('--tau_s', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.75)
parser.add_argument('--num_pseudoclasses', type=int, default=10)
# Training parameters
parser.add_argument('--episode_batch_size', type=int, default=80)
parser.add_argument('--num_views', type=int, default=6) 
parser.add_argument('--num_episodes_per_sleep', type=int, default=12800*5) # 12800 *5 comes from number of types of augmentations
parser.add_argument('--patience', type=int, default=40)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--save_dir', type=str, default="output/run_CSSL")
parser.add_argument('--data_order_path', type=str, default='data_class_order/IN10')
parser.add_argument('--data_order_file_name', type=str, default='data_class_order.txt')
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

    ### Scale sm_lr according to episode_batch_size
    args.sm_lr = args.sm_lr * args.episode_batch_size / 128
    
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
    episode_transform = Episode_Transformations(num_views = args.num_views, 
                                                mean = args.mean, 
                                                std = args.std, 
                                                return_actions=True)
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
    # Note: To make continuum work with action_codes, I edited the following file:
    # /home/jhair/anaconda3/envs/py39gpu/lib/python3.9/site-packages/continuum/tasks/image_array_task_set.py
    # Specifically, line 112. The idea is to be able to pass a tuple variable on x, where the img is on x[0]
    # Maybe I should create a fork version of continuum with that change (and install with setup.py)
    if args.iid: # iid data
        train_tasks = InstanceIncremental(train_parent_dataset, nb_tasks=args.num_tasks, transformations=train_tranform)
        val_tasks = InstanceIncremental(val_parent_dataset, nb_tasks=args.num_tasks, transformations=val_transform)
    else: # non-iid data (class incremental)
        assert args.num_classes % args.num_tasks == 0, "Number of classes must be divisible by number of tasks"
        args.class_increment = int(args.num_classes // args.num_tasks)
        train_tasks = ClassIncremental(train_parent_dataset, increment = args.class_increment, transformations = train_tranform, class_order = data_class_order)
        val_tasks = ClassIncremental(val_parent_dataset, increment = args.class_increment, transformations = val_transform, class_order = data_class_order)

    ### Load view_encoder and semantic_memory
    print('\n==> Preparing model...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    conditional_generator = eval(args.generator_model_name)(dec_num_Blocks = args.dec_num_Blocks,
                                                            dec_num_out_channels = args.dec_num_out_channels,
                                                            ft_feature_dim=args.ft_feature_dim, 
                                                            ft_action_code_dim=args.ft_action_code_dim, 
                                                            ft_num_layers=args.ft_num_layers, 
                                                            ft_nhead=args.ft_nhead, 
                                                            ft_dim_feedforward=args.ft_dim_feedforward, 
                                                            ft_dropout=args.ft_dropout)
    semantic_memory = eval(args.semantic_model_name)(view_encoder.fc.weight.shape[1], 
                                                     num_pseudoclasses = args.num_pseudoclasses, 
                                                     proj_dim = args.proj_dim,
                                                     out_dim = args.out_dim)
    print('\nView encoder')
    print(view_encoder)
    print('\nConditional Generator')
    print(conditional_generator)
    print('\nSemantic Memory')
    print(semantic_memory)
    print('\n')

    ### Load pretrained view encoder
    encoder_state_dict = torch.load(args.pretrained_enc_path)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(encoder_state_dict, strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    view_encoder.fc = torch.nn.Identity()
    # freeze view_encoder (Erase this when training view_encoder. Manage freezing on each phase!!)
    for param in view_encoder.parameters():
        param.requires_grad = False

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    conditional_generator = torch.nn.DataParallel(conditional_generator)
    semantic_memory = torch.nn.DataParallel(semantic_memory)
    view_encoder = view_encoder.to(device)
    conditional_generator = conditional_generator.to(device)
    semantic_memory = semantic_memory.to(device)

    ### Define wake-sleep trainer
    WS_trainer = Wake_Sleep_trainer(view_encoder, 
                                    conditional_generator,
                                    semantic_memory, 
                                    episode_batch_size = args.episode_batch_size, 
                                    tau_t = args.tau_t,
                                    tau_s = args.tau_s,
                                    beta = args.beta,
                                    num_episodes_per_sleep = args.num_episodes_per_sleep,
                                    device = device,
                                    num_views = args.num_views,
                                    save_dir = args.save_dir,
                                    dataset_mean = args.mean,
                                    dataset_std = args.std,
                                    num_pseudoclasses = args.num_pseudoclasses)

    ### Loop over tasks
    print('\n==> Start wake-sleep training')

    init_time = time.time()
    scaler = GradScaler()
    # Load all validation data
    val_loader_all = torch.utils.data.DataLoader(val_tasks[:], batch_size = 128, shuffle = False, num_workers = args.workers)
    # Set variable to save metrics
    semantic_metrics_seen_data_val = {'Metric_mode': 'semantic_seen_data_val'}
    semantic_metrics_all_data_val = {'Metric_mode': 'semantic_all_data_val'}
    semantic_metrics_seen_data_train = {'Metric_mode': 'semantic_seen_data_train'}
    generator_metrics_seen_data_train = {'Metric_mode': 'generator_seen_data_train'}

    # Start task training
    cycle_generalcounter = 1
    for task_id in range(len(train_tasks)):
        print(f"\n\n\n------ Task {task_id+1}/{len(train_tasks)} ------")
        start_time = time.time()
        
        ## Create variable to track metrics of current task
        semantic_metrics_seen_data_val[f'task_id_{task_id}'] = {'NREM_REM_indicator':[], 'Cycle':[], 'Cycle_General':[], 'Episodes_seen':[], 'NMI':[], 'AMI':[], 'ARI':[], 'F':[], 'ACC':[], 'ACC-Top5':[]}
        semantic_metrics_all_data_val[f'task_id_{task_id}'] = {'NREM_REM_indicator':[], 'Cycle':[], 'Cycle_General':[], 'Episodes_seen':[], 'NMI':[], 'AMI':[], 'ARI':[], 'F':[], 'ACC':[], 'ACC-Top5':[]}
        semantic_metrics_seen_data_train[f'task_id_{task_id}'] = {'NREM_REM_indicator':[], 'Cycle':[], 'Cycle_General':[], 'Episodes_seen':[], 'NMI':[], 'AMI':[], 'ARI':[], 'F':[], 'ACC':[], 'ACC-Top5':[]}
        generator_metrics_seen_data_train[f'task_id_{task_id}'] = {'NREM_REM_indicator':[], 'Cycle':[], 'Cycle_General':[], 'Episodes_seen':[], 'FTtensor_loss':[], 'GENtensor_loss':[], 'UncondGENtensor_loss':[]}
        ## Get current task train loader
        train_loader = torch.utils.data.DataLoader(train_tasks[task_id], batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
        ## Get seen tasks val loader
        val_loader_seen = torch.utils.data.DataLoader(val_tasks[:task_id+1], batch_size = 128, shuffle = False, num_workers = args.workers)
        


        ###### WAKE PHASE ######
        print("\n#### Wake Phase... ####")
        WS_trainer.wake_phase(train_loader)
        del train_loader



        ###### SLEEP PHASE ######
        print("\n#### Sleep Phase ... ####")
        param_groups = [
            {'params': conditional_generator.parameters(), 'lr': args.generator_lr, 'weight_decay': args.generator_wd},
            {'params': semantic_memory.parameters(), 'lr': args.sm_lr, 'weight_decay': args.sm_wd}
            ]
        criterion_crossentropyswap = SwapLossViewExpanded(num_views = args.num_views).to(device)
        criterion_mse = torch.nn.MSELoss().to(device)
        
        # NREM-REM cycles
        cycle_innercounter = 1
        while WS_trainer.sleep_episode_counter < args.num_episodes_per_sleep:
            
            #### NREM ####
            print(f"\n## NREM Sleep -- task {task_id+1} cycle {cycle_innercounter} ##")
            optimizer = torch.optim.AdamW(param_groups, lr = 0, weight_decay = 0)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr = [args.generator_lr, args.sm_lr], 
                                                        steps_per_epoch = args.num_episodes_batch_per_sleep, 
                                                        epochs = 1)
            train_stats = WS_trainer.NREM_sleep(optimizer = optimizer, 
                                                criterions = [criterion_crossentropyswap, criterion_mse],
                                                scheduler = scheduler,
                                                writer = writer, 
                                                task_id = task_id,
                                                scaler=scaler,
                                                patience=args.patience)
            seen_episodes = task_id*WS_trainer.num_episodes_per_sleep + WS_trainer.sleep_episode_counter
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep: # if sleep limit has been reached, save clusters results
                val_stats_seendata = WS_trainer.evaluate_semantic_memory(val_loader_seen, num_gt_classes = args.num_classes, plot_clusters = True, 
                                                            save_dir_clusters = os.path.join(args.save_dir,'Semantic_metrics_val_seen_data'), 
                                                            task_id = task_id, mean = args.mean, std = args.std)
                val_stats_alldata = WS_trainer.evaluate_semantic_memory(val_loader_all, num_gt_classes = args.num_classes, plot_clusters = True, 
                                                            save_dir_clusters = os.path.join(args.save_dir,'Semantic_metrics_val_all_data'), 
                                                            task_id = task_id, mean = args.mean, std = args.std)
            else: # Only save stats
                val_stats_seendata = WS_trainer.evaluate_semantic_memory(val_loader_seen, plot_clusters = False)
                val_stats_alldata = WS_trainer.evaluate_semantic_memory(val_loader_all, plot_clusters = False)
            # Print stats
            print(f'\tTrain NREM metrics (episodes) -- task {task_id+1} cycle {cycle_innercounter}')
            print(f"\t\tNMI: {train_stats['NMI']:.4f}, AMI: {train_stats['AMI']:.4f}, ARI: {train_stats['ARI']:.4f}, ACC: {train_stats['ACC']:.4f}")
            print(f"\t\tMSE (FTtensor): {train_stats['FTtensor_loss']:.4f}, MSE (GENtensor): {train_stats['GENtensor_loss']:.4f}, MSE (UncondGENtensor): {train_stats['UncondGENtensor_loss']:.4f}")
            print(f'\tVal NREM seen data metrics (images) -- task {task_id+1} cycle {cycle_innercounter} ')
            print(f'\t\tNMI: {val_stats_seendata["NMI"]:.4f}, AMI: {val_stats_seendata["AMI"]:.4f}, ARI: {val_stats_seendata["ARI"]:.4f}, ACC: {val_stats_seendata["ACC"]:.4f}')
            print(f'\tVal NREM all data metrics (images) -- task {task_id+1} cycle {cycle_innercounter} ')
            print(f'\t\tNMI: {val_stats_alldata["NMI"]:.4f}, AMI: {val_stats_alldata["AMI"]:.4f}, ARI: {val_stats_alldata["ARI"]:.4f}, ACC: {val_stats_alldata["ACC"]:.4f}')
            # Acumulate stats
            semantic_metrics_seen_data_train = acumulate_metrics_semantics(semantic_metrics_seen_data_train, train_stats, 0, cycle_innercounter, cycle_generalcounter, seen_episodes, task_id)
            semantic_metrics_seen_data_val = acumulate_metrics_semantics(semantic_metrics_seen_data_val, val_stats_seendata, 0, cycle_innercounter, cycle_generalcounter, seen_episodes, task_id)
            semantic_metrics_all_data_val = acumulate_metrics_semantics(semantic_metrics_all_data_val, val_stats_alldata, 0, cycle_innercounter, cycle_generalcounter, seen_episodes, task_id)
            generator_metrics_seen_data_train = accumulate_metrics_generator(generator_metrics_seen_data_train, train_stats, 0, cycle_innercounter, cycle_generalcounter, seen_episodes, task_id)
            # check if sleep limit reached
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep:
                print('\n---Sleep limit reached. Waking up now---')
                cycle_generalcounter += 1
                break
            

            #### REM ####
            print(f"\n## REM Sleep -- task {task_id+1} cycle {cycle_innercounter} ##")
            optimizer = torch.optim.AdamW(param_groups, lr = 0, weight_decay = 0)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr = [args.generator_lr, args.sm_lr], 
                                                    steps_per_epoch = args.num_episodes_batch_per_sleep, 
                                                    epochs = 1)
            train_stats = WS_trainer.REM_sleep(optimizer = optimizer, 
                                                criterions = [criterion_crossentropyswap, criterion_mse],
                                                scheduler = scheduler,
                                                writer = writer, 
                                                task_id = task_id,
                                                scaler=scaler,
                                                patience=args.patience)
            seen_episodes = task_id*WS_trainer.num_episodes_per_sleep + WS_trainer.sleep_episode_counter
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep: # if sleep limit has been reached, save clusters results
                val_stats_seendata = WS_trainer.evaluate_semantic_memory(val_loader_seen, num_gt_classes = args.num_classes, plot_clusters = True, 
                                                            save_dir_clusters = os.path.join(args.save_dir,'Semantic_metrics_val_seen_data'), 
                                                            task_id = task_id, mean = args.mean, std = args.std)
                val_stats_alldata = WS_trainer.evaluate_semantic_memory(val_loader_all, num_gt_classes = args.num_classes, plot_clusters = True, 
                                                            save_dir_clusters = os.path.join(args.save_dir,'Semantic_metrics_val_all_data'), 
                                                            task_id = task_id, mean = args.mean, std = args.std)
            else: # Only save stats
                val_stats_seendata = WS_trainer.evaluate_semantic_memory(val_loader_seen, plot_clusters = False)
                val_stats_alldata = WS_trainer.evaluate_semantic_memory(val_loader_all, plot_clusters = False)
            # Print stats
            print(f'\tTrain REM metrics (episodes) -- task {task_id+1} cycle {cycle_innercounter}')
            print(f"\t\tNMI: {train_stats['NMI']:.4f}, AMI: {train_stats['AMI']:.4f}, ARI: {train_stats['ARI']:.4f}, ACC: {train_stats['ACC']:.4f}")
            print(f'\tVal REM seen data metrics (images) -- task {task_id+1} cycle {cycle_innercounter} ')
            print(f'\t\tNMI: {val_stats_seendata["NMI"]:.4f}, AMI: {val_stats_seendata["AMI"]:.4f}, ARI: {val_stats_seendata["ARI"]:.4f}, ACC: {val_stats_seendata["ACC"]:.4f}')
            print(f'\tVal REM all data metrics (images) -- task {task_id+1} cycle {cycle_innercounter} ')
            print(f'\t\tNMI: {val_stats_alldata["NMI"]:.4f}, AMI: {val_stats_alldata["AMI"]:.4f}, ARI: {val_stats_alldata["ARI"]:.4f}, ACC: {val_stats_alldata["ACC"]:.4f}')
            # Acumulate stats
            semantic_metrics_seen_data_train = acumulate_metrics_semantics(semantic_metrics_seen_data_train, train_stats, 1, cycle_innercounter, cycle_generalcounter, seen_episodes, task_id)
            semantic_metrics_seen_data_val = acumulate_metrics_semantics(semantic_metrics_seen_data_val, val_stats_seendata, 1, cycle_innercounter, cycle_generalcounter, seen_episodes, task_id)
            semantic_metrics_all_data_val = acumulate_metrics_semantics(semantic_metrics_all_data_val, val_stats_alldata, 1, cycle_innercounter, cycle_generalcounter, seen_episodes, task_id)
            # check if sleep limit reached
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep:
                print('\n---Sleep limit reached. Waking up now---')
                cycle_generalcounter += 1
                break
            
            # Update cycle counter
            cycle_innercounter += 1
            cycle_generalcounter += 1

        ## Plot some reconstructions using the training data (episodes)
        train_loader_all = torch.utils.data.DataLoader(train_tasks[:], batch_size = 128, shuffle = False, num_workers = args.workers)
        print("Plot generated images as examples (From training set since we have episodes and actions there)...")
        WS_trainer.evaluate_generator(train_loader_all, device=device,
                                      save_dir = os.path.join(args.save_dir,'Generator_examples_train'),
                                      task_id = task_id)

        ### Save all metrics
        with open(os.path.join(args.save_dir, 'semantic_metrics_seen_data_train.json'), 'w') as f:
            json.dump(semantic_metrics_seen_data_train, f, indent=2)
        with open(os.path.join(args.save_dir, 'semantic_metrics_seen_data_val.json'), 'w') as f:
            json.dump(semantic_metrics_seen_data_val, f, indent=2)
        with open(os.path.join(args.save_dir, 'semantic_metrics_all_data_val.json'), 'w') as f:
            json.dump(semantic_metrics_all_data_val, f, indent=2)
        with open(os.path.join(args.save_dir, 'generator_metrics_seen_data_train.json'), 'w') as f:
            json.dump(generator_metrics_seen_data_train, f, indent=2)

        ### Save semantic memory
        semantic_memory_state_dict = semantic_memory.module.state_dict()
        torch.save(semantic_memory_state_dict, os.path.join(args.save_dir, f'semantic_memory_taskid_{task_id}.pth'))

        ### Save conditional generator
        conditional_generator_state_dict = conditional_generator.module.state_dict()
        torch.save(conditional_generator_state_dict, os.path.join(args.save_dir, f'conditional_generator_taskid_{task_id}.pth'))
        
        ### Print time
        print(f'Task {task_id} Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    # Close tensorboard writer
    writer.close()

    print('\n==> END')

    return None

def acumulate_metrics_semantics(metric_dict, current_metrics, nrem_rem_indicator, cycle, general_cycle, episodes_seen, taskID):
    metric_dict[f'task_id_{taskID}']['NREM_REM_indicator'].append(nrem_rem_indicator)
    metric_dict[f'task_id_{taskID}']['Cycle'].append(cycle)
    metric_dict[f'task_id_{taskID}']['Cycle_General'].append(general_cycle)
    metric_dict[f'task_id_{taskID}']['Episodes_seen'].append(episodes_seen)
    metric_dict[f'task_id_{taskID}']['NMI'].append(current_metrics['NMI'])
    metric_dict[f'task_id_{taskID}']['AMI'].append(current_metrics['AMI'])
    metric_dict[f'task_id_{taskID}']['ARI'].append(current_metrics['ARI'])
    metric_dict[f'task_id_{taskID}']['F'].append(current_metrics['F'])
    metric_dict[f'task_id_{taskID}']['ACC'].append(current_metrics['ACC'])
    metric_dict[f'task_id_{taskID}']['ACC-Top5'].append(current_metrics['ACC-Top5'])

    return metric_dict

def accumulate_metrics_generator(metric_dict, current_metrics, nrem_rem_indicator, cycle, general_cycle, episodes_seen, taskID):
    metric_dict[f'task_id_{taskID}']['NREM_REM_indicator'].append(nrem_rem_indicator)
    metric_dict[f'task_id_{taskID}']['Cycle'].append(cycle)
    metric_dict[f'task_id_{taskID}']['Cycle_General'].append(general_cycle)
    metric_dict[f'task_id_{taskID}']['Episodes_seen'].append(episodes_seen)
    metric_dict[f'task_id_{taskID}']['FTtensor_loss'].append(current_metrics['FTtensor_loss'])
    metric_dict[f'task_id_{taskID}']['GENtensor_loss'].append(current_metrics['GENtensor_loss'])
    metric_dict[f'task_id_{taskID}']['UncondGENtensor_loss'].append(current_metrics['UncondGENtensor_loss'])
    return metric_dict


if __name__ == '__main__':
    main()