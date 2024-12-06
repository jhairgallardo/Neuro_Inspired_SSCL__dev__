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
parser.add_argument('--pretrained_enc_path', type=str)
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

    ### Test episode transforms using normal pytorch dataset
    # train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=episode_transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size, 
    #                                         shuffle=True, num_workers=args.workers, pin_memory=True,
    #                                         persistent_workers=True, drop_last=True)
    # # Plot first espisode that has 12 images
    # # Get first batch with iter next
    # first_batch, _ = next(iter(train_loader))
    # first_batch_imgs = first_batch[0]
    # first_batch_actions = first_batch[1]
    # import matplotlib.pyplot as plt
    # import torchvision
    # # unnorm first batch images
    # episode_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in first_batch_imgs[0]]
    # plt.figure()
    # for i in range(12):
    #     view = episode_imgs[i]
    #     view = view.permute(1,2,0).cpu().numpy()
    #     plt.subplot(3,4,i+1)
    #     plt.imshow(view)
    #     plt.axis('off')
    # plt.savefig(os.path.join(args.save_dir, 'episode_transform_example.png'))
    # plt.close()

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
        
    # ### Test episodes using continuum
    # import matplotlib.pyplot as plt
    # import torch.nn.functional as F
    # for i in range(3): # check if action vectors for random crops are ok
    #     a = train_tasks[0][i]
    #     original_image = a[0][0][0]
    #     cropped_image = a[0][0][1]
    #     bbox = a[0][1][1]
    #     # make bbox values int
    #     bbox = [int(i*224) for i in bbox]
    #     cropped_image_manually = original_image[:, bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
    #     # resize cropped_image_manually to 224x224
    #     cropped_image_manually = F.interpolate(cropped_image_manually.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
    #     # Plot the 3 images (original, cropped, cropped_manually) in a grid of 1x3
    #     fig, ax = plt.subplots(1, 3)
    #     ax[0].imshow(original_image.permute(1,2,0))
    #     ax[0].set_title('Original Image')
    #     ax[1].imshow(cropped_image.permute(1,2,0))
    #     ax[1].set_title('Cropped Image')
    #     ax[2].imshow(cropped_image_manually.permute(1,2,0))
    #     plt.savefig(os.path.join(args.save_dir, f'episode_example_{i}.png'))
    #     plt.close()

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
    # freeze view_encoder
    for param in view_encoder.parameters(): # Erase this when training view_encoder. Manage freezing on each phase
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
                                    args = args)

    ### Loop over tasks
    print('\n==> Start wake-sleep training')
    init_time = time.time()
    saved_metrics = {'Train_metrics':{}, 'Val_metrics_seen_data':{}, 'Val_metrics_all_data':{}}
    scaler = GradScaler()
    for task_id in range(len(train_tasks)):
        start_time = time.time()

        print(f"\n------ Task {task_id+1}/{len(train_tasks)} ------")

        ## Get tasks train loader
        train_dataset = train_tasks[task_id]
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size = args.episode_batch_size, 
                                                   shuffle = True, 
                                                   num_workers = args.workers, 
                                                   pin_memory = True)
        
        ### WAKE PHASE ###
        print("Wake Phase...")
        WS_trainer.wake_phase(train_loader)
        del train_dataset, train_loader

        ### SLEEP PHASE ###
        print("Sleep Phase -- NREM ...")
        param_groups = [
            {'params': conditional_generator.parameters(), 'lr': args.generator_lr, 'weight_decay': args.generator_wd},
            {'params': semantic_memory.parameters(), 'lr': args.sm_lr, 'weight_decay': args.sm_wd}
            ]
        optimizer = torch.optim.AdamW(param_groups, lr = 0, weight_decay = 0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr = [args.generator_lr, args.sm_lr], 
                                                        steps_per_epoch = args.num_episodes_batch_per_sleep, 
                                                        epochs = 1)
        criterion_crossentropyswap = SwapLossViewExpanded(num_views = args.num_views).to(device)
        criterion_mse = torch.nn.MSELoss().to(device)
        train_metrics = WS_trainer.sleep_phase(num_episodes_per_sleep = args.num_episodes_per_sleep,
                                               optimizer = optimizer, 
                                               criterions = [criterion_crossentropyswap,
                                                            criterion_mse],
                                               scheduler = scheduler,
                                               classes_list = data_class_order,
                                               writer = writer, 
                                               task_id = task_id,
                                               scaler=scaler)
        
        ### Evaluate conditional generator on training data (check if reconstructions are ok)
        train_dataset_reconstructions = train_tasks[:task_id+1]
        train_loader_reconstructions = torch.utils.data.DataLoader(train_dataset_reconstructions, batch_size = 128, shuffle = False, num_workers = args.workers)
        WS_trainer.evaluate_generator(train_loader_reconstructions, device=device,
                                      save_dir = os.path.join(args.save_dir,'reconstructions_training_seen_data'),
                                      task_id = task_id)

        ### Evaluate model on validation set (seen so far)
        val_dataset = val_tasks[:task_id+1]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        print("\nEvaluate model on seen validation data...")
        val_seendata_metrics = WS_trainer.evaluate_semantic_memory(val_loader,
                                                                   plot_clusters = True, 
                                                                   save_dir_clusters = os.path.join(args.save_dir,'pseudo_classes_clusters_seen_data'), 
                                                                   task_id = task_id, 
                                                                   mean = args.mean, 
                                                                   std = args.std)
        
        ### Evaluate model on validation set (all data)
        val_dataset = val_tasks[:]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        print('\nEvaluating model on all validation data...')
        val_alldata_metrics = WS_trainer.evaluate_semantic_memory(val_loader,
                                                                 plot_clusters = True, 
                                                                 save_dir_clusters = os.path.join(args.save_dir,'pseudo_classes_clusters_all_data'), 
                                                                 task_id = task_id,
                                                                 mean = args.mean,
                                                                 std = args.std)
        
        ### Save metrics
        saved_metrics['Train_metrics'][f'Task_{task_id}'] = train_metrics
        saved_metrics['Val_metrics_seen_data'][f'Task_{task_id}'] = val_seendata_metrics
        saved_metrics['Val_metrics_all_data'][f'Task_{task_id}'] = val_alldata_metrics
        with open(os.path.join(args.save_dir, 'saved_metrics.json'), 'w') as f:
            json.dump(saved_metrics, f, indent=2)

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