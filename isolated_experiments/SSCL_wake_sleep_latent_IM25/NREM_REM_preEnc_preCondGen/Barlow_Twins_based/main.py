import argparse
import os, time

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental, InstanceIncremental

from models import *
from loss_functions import BarlowLossViewExpanded, KoLeoLossViewExpanded
from augmentations import Episode_Transformations
from wake_sleep_trainer import Wake_Sleep_trainer
from utils import knn_eval

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

parser = argparse.ArgumentParser(description='Wake-Sleep latent -- pretrained encoder -- pretrained conditional generator -- IM5random pre-trainining')
### Dataset parameters ###
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-25')
parser.add_argument('--data_order_file_name', type=str, default='./../../IM25_data_class_orders/IM25_data_class_order0.txt')
parser.add_argument('--num_classes', type=int, default=25)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
parser.add_argument('--iid', action='store_true')
### View encoder parameters ###
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--enc_pretrained_file_path', type=str, default='./../../Pretraining_Encoder/Barlow_Twins_IM5_random/output/Barlow_IM5random_offPRE_encprojstandard_views@12_epochs@100_lr@0.003_wd@0_bs@128_seed@0/view_encoder_epoch99.pth')
### Conditional Generator parameters ###
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--condgen_pretrained_file_path', type=str, default='./../../Pretraining_Cond_Generator/BarlowTwins_based_IM5_random/output/CondGen_IM5random_offPRE_encstandard_genGNReLU_preEncIM5random_views@6_epochs@100_lr@0.001_wd@0.01_bs@80__seed@0/cond_generator_epoch99.pth')
parser.add_argument('--dec_num_Blocks', type=list, default=[1,1,1,1])
parser.add_argument('--dec_num_out_channels', type=int, default=3)
parser.add_argument('--ft_feature_dim', type=int, default=512)
parser.add_argument('--ft_action_code_dim', type=int, default=11)
parser.add_argument('--ft_num_layers', type=int, default=2)
parser.add_argument('--ft_nhead', type=int, default=4)
parser.add_argument('--ft_dim_feedforward', type=int, default=256)
parser.add_argument('--ft_dropout', type=float, default=0.1)
### Representation learning head (barlow twins projector) ###
parser.add_argument('--proj_model_name', type=str, default='Projector_Model')
parser.add_argument('--proj_dim', type=int, default=2048)
### Training parameters ###
# Conditional generator
parser.add_argument('--condgen_lr', type=float, default=0.001)
parser.add_argument('--condgen_wd', type=float, default=0.01)
# Representation learning head
parser.add_argument('--rep_lr', type=float, default=0.003)
parser.add_argument('--rep_wd', type=float, default=0)
# NREM-REM swicth
parser.add_argument('--patience', type=int, default=80)
parser.add_argument('--threshold', type=float, default=1e-4)
parser.add_argument('--window', type=int, default=50)
parser.add_argument('--smooth_loss_alpha', type=float, default=0.3)
# Other
parser.add_argument('--koleo_gamma', type=float, default=0)
parser.add_argument('--episode_batch_size', type=int, default=80)
parser.add_argument('--num_views', type=int, default=6) 
parser.add_argument('--num_episodes_per_sleep', type=int, default=128000)
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
    args.rep_lr = args.rep_lr * args.episode_batch_size / 128
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

    ### Create train loader and val laoder for KNN
    train_tasks_knn = ClassIncremental(train_parent_dataset, increment = args.class_increment, transformations = val_transform, class_order = data_class_order)
    train_knn_dataloader = torch.utils.data.DataLoader(train_tasks_knn[:], batch_size = args.episode_batch_size, shuffle = False, num_workers = args.workers)
    val_knn_dataloader = torch.utils.data.DataLoader(val_tasks[:], batch_size = args.episode_batch_size, shuffle = False, num_workers = args.workers)

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
    projector_rep = eval(args.proj_model_name)(input_dim = view_encoder.fc.weight.shape[1], hidden_dim = args.proj_dim, output_dim = args.proj_dim)
    
    ### Load pretrained models (view_encoder, conditional_generator)
    print('\n==> Loading pretrained models (view_encoder, conditional_generator) ...')
    # Load pretrained view_encoder
    encoder_state_dict = torch.load(args.enc_pretrained_file_path)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(encoder_state_dict, strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    view_encoder.fc = torch.nn.Identity()
    # Load pretrained conditional_generator
    generator_state_dict = torch.load(args.condgen_pretrained_file_path)
    missing_keys, unexpected_keys = conditional_generator.load_state_dict(generator_state_dict, strict=True)

    ### Print models
    print('\nView encoder')
    print(view_encoder)
    print('\nConditional Generator')
    print(conditional_generator)
    print('\nRepresentation learning head (Barlow Twins projector)')
    print(projector_rep)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    conditional_generator = torch.nn.DataParallel(conditional_generator)
    projector_rep = torch.nn.DataParallel(projector_rep)
    view_encoder = view_encoder.to(device)
    conditional_generator = conditional_generator.to(device)
    projector_rep = projector_rep.to(device)

    ### Define wake-sleep trainer
    print('\n==> Preparing Wake-Sleep trainer...')
    WS_trainer = Wake_Sleep_trainer(episode_batch_size = args.episode_batch_size,
                                    num_episodes_per_sleep = args.num_episodes_per_sleep,
                                    num_views = args.num_views,
                                    dataset_mean = args.mean,
                                    dataset_std = args.std,
                                    device = device,
                                    save_dir = args.save_dir,
                                    koleo_gamma = args.koleo_gamma)
    
    ### KNN eval before training
    print('\n==> KNN evaluation before training')
    knn_val_all = knn_eval(train_knn_dataloader, val_knn_dataloader, view_encoder, device, k=10, num_classes=args.num_classes)
    print(f'KNN val accuracy on all classes: {knn_val_all}')
    writer.add_scalar('KNN_val_accuracy_all_seen_classes', knn_val_all, 0)

    ### Save initial models
    # Save view encoder at init
    view_encoder_state_dict = view_encoder.module.state_dict()
    torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_taskid_init.pth'))
    # Save conditional generator at init
    conditional_generator_state_dict = conditional_generator.module.state_dict()
    torch.save(conditional_generator_state_dict, os.path.join(args.save_dir, f'conditional_generator_taskid_init.pth'))
    # Save projector_rep at init
    projector_rep_state_dict = projector_rep.module.state_dict()
    torch.save(projector_rep_state_dict, os.path.join(args.save_dir, f'projector_rep_taskid_init.pth'))

    ### Save one batch for plot purposes with conditional generator
    train_loader_aux = torch.utils.data.DataLoader(train_tasks[:], batch_size = args.episode_batch_size, shuffle = True)
    episodes_plot, _, _ = next(iter(train_loader_aux))
    del train_loader_aux

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

        ########################
        ###### WAKE PHASE ######
        ########################
        print("\n#### Wake Phase... ####")
        print(f'Seen Tasks: {list(range(task_id+1))} -- Seen classes: {train_tasks[:task_id+1].get_classes()}')
        WS_trainer.wake_phase(view_encoder, train_loader_current)
        del train_loader_current

        #########################
        ###### SLEEP PHASE ######
        #########################
        print("\n#### Sleep Phase ... ####")
        param_groups_rep_learning = [{'params': view_encoder.parameters(), 'lr': args.rep_lr, 'weight_decay': args.rep_wd},
                                     {'params': projector_rep.parameters(), 'lr': args.rep_lr, 'weight_decay': args.rep_wd}]
        optimizer_rep = torch.optim.AdamW(param_groups_rep_learning, lr = 0, weight_decay = 0)
        scheduler_rep = torch.optim.lr_scheduler.OneCycleLR(optimizer_rep, max_lr = [args.rep_lr, args.rep_lr], steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        optimizer_condgen = torch.optim.AdamW(conditional_generator.parameters(), lr = args.condgen_lr, weight_decay = args.condgen_wd)
        scheduler_condgen = torch.optim.lr_scheduler.OneCycleLR(optimizer_condgen, max_lr = args.condgen_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        criterion_rep = BarlowLossViewExpanded(num_views = args.num_views).to(device)
        criterion_condgen = torch.nn.MSELoss().to(device)

        ### NREM-REM cycles
        cycle_innercounter = 1
        while WS_trainer.sleep_episode_counter < args.num_episodes_per_sleep:
            
            #### NREM ####
            print(f"\n## NREM Sleep -- task {task_id+1} cycle {cycle_innercounter} ##")
            WS_trainer.NREM_sleep(view_encoder, conditional_generator,
                                optimizers = [optimizer_condgen], 
                                schedulers = [scheduler_condgen],
                                criterions = [criterion_condgen],
                                task_id = task_id,
                                patience=args.patience,
                                threshold=args.threshold,
                                window=args.window,
                                smooth_loss_alpha=args.smooth_loss_alpha,
                                scaler=scaler,
                                writer = writer)
            seen_episodes_so_far = task_id*args.num_episodes_per_sleep + WS_trainer.sleep_episode_counter
            writer.add_scalar('NREM-REM_val_indicator', 0, seen_episodes_so_far)
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep:
                print('\n---Sleep limit reached. Waking up now---')
                break
            

            #### REM ####
            print(f"\n## REM Sleep -- task {task_id+1} cycle {cycle_innercounter} ##")
            WS_trainer.REM_sleep(view_encoder, projector_rep, conditional_generator,
                                optimizers = [optimizer_rep], 
                                schedulers = [scheduler_rep],
                                criterions = [criterion_rep],
                                task_id = task_id,
                                patience=args.patience,
                                threshold=args.threshold,
                                window=args.window,
                                scaler=scaler,
                                writer = writer)
            seen_episodes_so_far = task_id*args.num_episodes_per_sleep + WS_trainer.sleep_episode_counter
            writer.add_scalar('NREM-REM_val_indicator', 1, seen_episodes_so_far)
            # Temporary workaround to avoid drift in the episodic memory #
            # Since REM trains the view encoder. We need to update the tensor in the episodic memory to avoid drift.
            # I have saved the images of each episode. Here, we pass them through the view encoder and update the episodic memory
            WS_trainer.update_episodic_memory(view_encoder, device)
            if WS_trainer.sleep_episode_counter >= args.num_episodes_per_sleep:
                print('\n---Sleep limit reached. Waking up now---')
                break

            cycle_innercounter += 1

        # TODO #####################################################################################################
        ### Evaluate conditional generator on saved training data batch (calculate loss and plot reconstructions)
        gen_losses = evaluate_generator_batch(trplot_batch, view_encoder, conditional_generator, criterion_mse,
                                              task_id, device, args.mean, args.std, args.save_dir)
        writer.add_scalar('Train_holdbatch_loss_gen1', gen_losses[0], task_id)
        writer.add_scalar('Train_holdbatch_loss_gen2', gen_losses[1], task_id)
        writer.add_scalar('Train_holdbatch_loss_gen3', gen_losses[2], task_id)
        writer.add_scalar('Train_holdbatch_loss_gentotal', gen_losses[3], task_id)
        #############################################################################################################

        ### Save view encoder
        view_encoder_state_dict = view_encoder.module.state_dict()
        torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_taskid_{task_id}.pth'))

        ### Save representation learning head (Barlow Twins projector)
        projector_rep_state_dict = projector_rep.module.state_dict()
        torch.save(projector_rep_state_dict, os.path.join(args.save_dir, f'projector_rep_taskid_{task_id}.pth'))

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