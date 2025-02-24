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
from wake_sleep_trainer import Wake_Sleep_trainer, evaluate_semantic_memory, evaluate_generator_batch

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

# turn off warnings
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Wake-Sleep latent -- pretrained fixed encoder -- pretrained conditional generator -- first_batch_pretraining')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--data_order_file_name', type=str, default='./../IM10_data_class_orders/IM10_data_class_order0.txt')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
parser.add_argument('--iid', action='store_true')
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--enc_pretrained_file_path', type=str, default='./../Pretraining_Encoder/output/MIRA_IN10_offPRE_viewEnc_encGNprojGN_preclasses@10_pseudoclasses@10_views@12_epochs@100_lr@0.008_wd@1e-6_bs@128_data_class_order0_seed@0/view_encoder_epoch99.pth')
# Generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--condgen_pretrained_file_path', default=None)#, default='./../Pretraining_Cond_Generator/output/MIRA_IN10_offPRE_cond_generator_preEncPseudoclasses@10_preclasses@10_views@6_epochs@100_lr@1e-3_wd@1e-2_bs@80_data_class_order0_seed@/cond_generator_epoch99.pth')
parser.add_argument('--condgen_lr', type=float, default=1e-3)
parser.add_argument('--condgen_wd', type=float, default=1e-2)
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
parser.add_argument('--num_pseudoclasses', type=int, default=10)
parser.add_argument('--sem_lr', type=float, default=0.01)
parser.add_argument('--sem_wd', type=float, default=1e-6)
parser.add_argument('--proj_dim', type=int, default=2048)
parser.add_argument('--out_dim', type=int, default=1024)
parser.add_argument('--tau_t', type=float, default=0.225)
parser.add_argument('--tau_s', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.75)
# Training parameters
parser.add_argument('--episode_batch_size', type=int, default=80)
parser.add_argument('--num_views', type=int, default=6) 
parser.add_argument('--num_episodes_per_sleep', type=int, default=64000)
parser.add_argument('--patience', type=int, default=40)
parser.add_argument('--threshold', type=float, default=1e-3)
parser.add_argument('--window', type=int, default=50)
parser.add_argument('--workers', type=int, default=16)
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
    args.num_batch_episodes_per_sleep = int(np.ceil(args.num_episodes_per_sleep/args.episode_batch_size))
    args.sem_lr = args.sem_lr * args.episode_batch_size / 128
    args.condgen_lr = args.condgen_lr * args.episode_batch_size / 128
    if not os.path.exists(args.save_dir): # Create save directory
        os.makedirs(args.save_dir)
    print(args)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f: # Save arguments
        json.dump(args.__dict__, f, indent=2)

    ### Seed everything, define device, define writer
    seed_everything(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load data tasks
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
    episode_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=True)
    train_tranform = [episode_transform]
    val_transform = [transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = args.mean, std = args.std)])]
    train_parent_dataset = ImageFolderDataset(data_path = os.path.join(args.data_path, 'train'))
    val_parent_dataset = ImageFolderDataset(data_path = os.path.join(args.data_path, 'val'))
    data_class_order = list(np.loadtxt(args.data_order_file_name, dtype=int))
    if args.iid: # iid
        train_tasks = InstanceIncremental(train_parent_dataset, nb_tasks=args.num_tasks, transformations=train_tranform)
        val_tasks = InstanceIncremental(val_parent_dataset, nb_tasks=args.num_tasks, transformations=val_transform)
    else: # non-iid (class incremental)
        assert args.num_classes % args.num_tasks == 0, "Number of classes must be divisible by number of tasks"
        args.class_increment = int(args.num_classes // args.num_tasks)
        train_tasks = ClassIncremental(train_parent_dataset, increment = args.class_increment, transformations = train_tranform, class_order = data_class_order)
        val_tasks = ClassIncremental(val_parent_dataset, increment = args.class_increment, transformations = val_transform, class_order = data_class_order)

    ### Load models
    print('\n==> Preparing models...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    conditional_generator = eval(args.condgen_model_name)(dec_num_Blocks = args.dec_num_Blocks,
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
    
    ### Load pretrained models (view_encoder, conditional_generator)
    print('\n==> Loading pretrained models (view_encoder, conditional_generator) ...')
    ## Load pretrained view_encoder
    encoder_state_dict = torch.load(args.enc_pretrained_file_path)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(encoder_state_dict, strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    view_encoder.fc = torch.nn.Identity()
    ## Freeze view_encoder (Erase this when training view_encoder. Manage freezing on each phase!!)
    for param in view_encoder.parameters():
        param.requires_grad = False
    ## Load pretrained conditional_generator
    if args.condgen_pretrained_file_path is not None:
        generator_state_dict = torch.load(args.condgen_pretrained_file_path)
        missing_keys, unexpected_keys = conditional_generator.load_state_dict(generator_state_dict, strict=False)
        assert missing_keys == [] and unexpected_keys == []

    ### Print models
    print('\nView encoder')
    print(view_encoder)
    print('\nConditional Generator')
    print(conditional_generator)
    print('\nSemantic Memory')
    print(semantic_memory)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    conditional_generator = torch.nn.DataParallel(conditional_generator)
    semantic_memory = torch.nn.DataParallel(semantic_memory)
    view_encoder = view_encoder.to(device)
    conditional_generator = conditional_generator.to(device)
    semantic_memory = semantic_memory.to(device)

    ### Define wake-sleep trainer
    print('\n==> Preparing Wake-Sleep trainer...')
    WS_trainer = Wake_Sleep_trainer(view_encoder = view_encoder, 
                                    conditional_generator= conditional_generator, 
                                    semantic_memory = semantic_memory, 
                                    episode_batch_size = args.episode_batch_size,
                                    num_episodes_per_sleep = args.num_episodes_per_sleep,
                                    num_views = args.num_views,
                                    tau_t = args.tau_t,
                                    tau_s = args.tau_s,
                                    beta = args.beta,
                                    dataset_mean = args.mean,
                                    dataset_std = args.std,
                                    device = device,
                                    save_dir = args.save_dir)

    ### Load a batch of data from all training data (one episode per class, so a total of 25 episodes)
    train_all_loader = torch.utils.data.DataLoader(train_tasks[:], batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers)
    gt_classes = list(range(args.num_classes))
    trplot_episodes_imgs = []
    trplot_episodes_actions = []
    trplot_episodes_labels = []
    for i, (batch, targets, _ ) in enumerate(train_all_loader):
        if len(gt_classes) == 0: break
        for j in range(args.episode_batch_size):
            if targets[j] in gt_classes:
                trplot_episodes_imgs.append(batch[0][j].unsqueeze(0))
                trplot_episodes_actions.append(batch[1][j].unsqueeze(0))
                trplot_episodes_labels.append(targets[j])
                gt_classes.remove(targets[j])
    trplot_episodes_imgs = torch.cat(trplot_episodes_imgs, dim=0)
    trplot_episodes_actions = torch.cat(trplot_episodes_actions, dim=0)
    trplot_episodes_labels = np.array(trplot_episodes_labels)
    trplot_batch = (trplot_episodes_imgs, trplot_episodes_actions, trplot_episodes_labels)

    ### Loop over tasks
    print('\n==> Start wake-sleep training')
    init_time = time.time()
    scaler = GradScaler()
    for task_id in range(len(train_tasks)):
        print(f"\n\n\n------ Task {task_id+1}/{len(train_tasks)} ------")
        start_time = time.time()
        writer.add_scalar('Task_boundaries', task_id+1, task_id*args.num_episodes_per_sleep)
        
        # Load current data
        train_loader_current = torch.utils.data.DataLoader(train_tasks[task_id], batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
        # Load seen validation data
        val_loader_seen = torch.utils.data.DataLoader(val_tasks[:task_id+1], batch_size = 128, shuffle = False, num_workers = args.workers, pin_memory = True)
        
        ###### WAKE PHASE ######
        print("\n#### Wake Phase... ####")
        print(f'Seeing classes: {train_tasks[task_id].get_classes()}')
        WS_trainer.wake_phase(train_loader_current)
        del train_loader_current

        ###### SLEEP PHASE ######
        print("\n#### Sleep Phase ... ####")
        # Set optimizer and schedulers
        optimizer_sem = torch.optim.AdamW(semantic_memory.parameters(), lr = args.sem_lr, weight_decay = args.sem_wd)
        scheduler_sem = torch.optim.lr_scheduler.OneCycleLR(optimizer_sem, max_lr = args.sem_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        optimizer_condgen = torch.optim.AdamW(conditional_generator.parameters(), lr = args.condgen_lr, weight_decay = args.condgen_wd)
        scheduler_condgen = torch.optim.lr_scheduler.OneCycleLR(optimizer_condgen, max_lr = args.condgen_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        # Set loss functions
        criterion_crossentropyswap = SwapLossViewExpanded(num_views = args.num_views).to(device)
        criterion_mse = torch.nn.MSELoss().to(device)

        cycle_innercounter = 1
        while WS_trainer.sleep_episode_counter < args.num_episodes_per_sleep:
            
            #### NREM ####
            print(f"\n## NREM Sleep -- task {task_id+1} cycle {cycle_innercounter} ##")
            WS_trainer.NREM_sleep(optimizers = [optimizer_sem, optimizer_condgen], 
                                schedulers = [scheduler_sem, scheduler_condgen],
                                criterions = [criterion_crossentropyswap, criterion_mse],
                                task_id = task_id,
                                patience=args.patience,
                                threshold=args.threshold,
                                window=args.window,
                                scaler=scaler,
                                writer = writer)
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep: plot_clusters=True
            else: plot_clusters=False
            val_seen_stats = evaluate_semantic_memory(val_loader_seen, view_encoder, semantic_memory, args.tau_s, args.num_pseudoclasses, 
                                                      task_id, device, plot_clusters, args.num_classes, args.save_dir)
            seen_episodes_so_far = task_id*args.num_episodes_per_sleep + WS_trainer.sleep_episode_counter
            writer.add_scalar('Val_seen_NMI', val_seen_stats['NMI'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_AMI', val_seen_stats['AMI'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_ARI', val_seen_stats['ARI'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_F', val_seen_stats['F'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_Top1_ACC', val_seen_stats['ACC'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_Top5_ACC', val_seen_stats['ACC-Top5'], seen_episodes_so_far)
            writer.add_scalar('NREM-REM_val_indicator', 0, seen_episodes_so_far)
            writer.add_scalar('Task_id_val_tag', task_id, seen_episodes_so_far)
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep:
                print('\n---Sleep limit reached. Waking up now---')
                break
            

            #### REM ####
            print(f"\n## REM Sleep -- task {task_id+1} cycle {cycle_innercounter} ##")
            WS_trainer.REM_sleep(optimizers = [optimizer_sem], 
                                schedulers = [scheduler_sem],
                                criterions = [criterion_crossentropyswap],
                                task_id = task_id,
                                patience=args.patience,
                                threshold=args.threshold,
                                window=args.window,
                                scaler=scaler,
                                writer = writer)
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep: plot_clusters=True
            else: plot_clusters=False
            val_seen_stats = evaluate_semantic_memory(val_loader_seen, view_encoder, semantic_memory, args.tau_s, args.num_pseudoclasses, 
                                                      task_id, device, plot_clusters, args.num_classes, args.save_dir)
            seen_episodes_so_far = task_id*args.num_episodes_per_sleep + WS_trainer.sleep_episode_counter
            writer.add_scalar('Val_seen_NMI', val_seen_stats['NMI'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_AMI', val_seen_stats['AMI'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_ARI', val_seen_stats['ARI'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_F', val_seen_stats['F'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_Top1_ACC', val_seen_stats['ACC'], seen_episodes_so_far)
            writer.add_scalar('Val_seen_Top5_ACC', val_seen_stats['ACC-Top5'], seen_episodes_so_far)
            writer.add_scalar('NREM-REM_val_indicator', 1, seen_episodes_so_far)
            writer.add_scalar('Task_id_val_tag', task_id, seen_episodes_so_far)
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep:
                print('\n---Sleep limit reached. Waking up now---')
                break

            cycle_innercounter += 1

        ### Evaluate conditional generator on saved training data batch (calculate loss and plot reconstructions)
        gen_losses = evaluate_generator_batch(trplot_batch, view_encoder, conditional_generator, criterion_mse,
                                              task_id, device, args.mean, args.std, args.save_dir)
        writer.add_scalar('Train_holdbatch_loss_gen1', gen_losses[0], task_id)
        writer.add_scalar('Train_holdbatch_loss_gen2', gen_losses[1], task_id)
        writer.add_scalar('Train_holdbatch_loss_gen3', gen_losses[2], task_id)
        writer.add_scalar('Train_holdbatch_loss_gentotal', gen_losses[3], task_id)

        ### Save view encoder
        view_encoder_state_dict = view_encoder.module.state_dict()
        torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_taskid_{task_id}.pth'))

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


if __name__ == '__main__':
    main()