import argparse
import os, time, random

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental, InstanceIncremental

from resnet_gn_mish import *
from wake_sleep_trainer import Wake_Sleep_trainer

from tensorboardX import SummaryWriter
from PIL import ImageFilter, ImageOps, ImageFilter
import einops
import numpy as np
import json

parser = argparse.ArgumentParser(description='SSCL Wake-Sleep veridical')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--iid', action='store_true')
# View encoder parameters
parser.add_argument('--view_encoder_model_name', type=str, default='resnet18')
parser.add_argument('--pretrained_view_encoder', type=str, default=None) 
# Semantic memory parameters
parser.add_argument('--proj_dim', type=int, default=2048)
parser.add_argument('--num_pseudoclasses', type=int, default=10)
# View generator parameters
parser.add_argument('--view_generator_model_name', type=str, default='resnet18dec')
parser.add_argument('--generator_lr', type=float, default=1e-3)
parser.add_argument('--generator_wd', type=float, default=1e-2)
# Other parameters
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--wd', type=float, default=1.5e-6)
parser.add_argument('--episode_batch_size', type=int, default=64)
parser.add_argument('--num_views', type=int, default=12)
parser.add_argument('--num_episodes_per_sleep', type=int, default=12800*5) # 1280 ----- 12800*5 comes from number of types of augmentations 
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--save_dir', type=str, default="output/run_CSSL")
parser.add_argument('--seed', type=int, default=0)

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
        train_tasks = InstanceIncremental(train_parent_dataset, nb_tasks=args.num_tasks, transformations=train_tranform)
        val_tasks = InstanceIncremental(val_parent_dataset, nb_tasks=args.num_tasks, transformations=val_transform)
    else: # non-iid data (class incremental)
        assert args.num_classes % args.num_tasks == 0, "Number of classes must be divisible by number of tasks"
        class_increment = int(args.num_classes // args.num_tasks)
        train_tasks = ClassIncremental(train_parent_dataset, increment = class_increment, 
                                    transformations = train_tranform, class_order = data_class_order)
        val_tasks = ClassIncremental(val_parent_dataset, increment = class_increment, 
                                    transformations = val_transform, class_order = data_class_order)

    ### Load view_encoder, semantic_memory, and view_generator
    print('\n==> Preparing model...')
    view_encoder = eval(args.view_encoder_model_name)(zero_init_residual = True, get_tensor_before_avgpool=True)
    semantic_memory = Semantic_Memory_Model(view_encoder.fc.weight.shape[1], num_pseudoclasses = args.num_pseudoclasses, proj_dim = args.proj_dim)
    view_generator = eval(args.view_generator_model_name)(num_Blocks=[1,1,1,1], nc=3)

    ### Load pretrained view_encoder
    if args.pretrained_view_encoder is not None:
        missing_keys, unexpected_keys = view_encoder.load_state_dict(torch.load(args.pretrained_view_encoder), strict=False)
        assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
        # freeze view_encoder
        for param in view_encoder.parameters():
            param.requires_grad = False

    ### All models as data parallel
    view_encoder = torch.nn.DataParallel(view_encoder)
    semantic_memory = torch.nn.DataParallel(semantic_memory)
    view_generator = torch.nn.DataParallel(view_generator)
    view_encoder = view_encoder.to(device)
    semantic_memory = semantic_memory.to(device)
    view_generator = view_generator.to(device)

    ### Define wake-sleep trainer
    WS_trainer = Wake_Sleep_trainer(view_encoder, semantic_memory, view_generator, args.episode_batch_size)

    ### Loop over tasks
    print('\n==> Start wake-sleep training')
    init_time = time.time()
    for task_id in range(len(train_tasks)):
        start_time = time.time()

        print(f"\n------ Task {task_id+1}/{len(train_tasks)} ------")

        ## Get tasks train loader
        train_dataset = train_tasks[task_id]
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.episode_batch_size, 
                                                   shuffle = True, num_workers = args.workers)
        
        ### WAKE PHASE ###
        print("Wake Phase...")
        WS_trainer.wake_phase(train_loader, device)
        del train_dataset, train_loader

        ### SLEEP PHASE ###
        print("Sleep Phase NREM...")
        optimizer_semantic_memory = torch.optim.AdamW(semantic_memory.parameters(), lr = args.lr, weight_decay = args.wd)
        criterion_semantic_memory = EntLoss(num_views = args.num_views).to(device)
        scheduler_semantic_memory = torch.optim.lr_scheduler.OneCycleLR(optimizer_semantic_memory, max_lr = args.lr,
                                                                        steps_per_epoch = args.num_episodes_batch_per_sleep, epochs = 1)
        
        optimizer_generator = torch.optim.AdamW(view_generator.module.get_parameters_controlled_wd(lr=args.generator_lr, wd=args.generator_wd), 
                                                lr = args.generator_lr, weight_decay = args.generator_wd)
        criterion_generator = torch.nn.MSELoss().to(device)
        scheduler_generator = torch.optim.lr_scheduler.OneCycleLR(optimizer_generator, max_lr = args.generator_lr, 
                                                                  steps_per_epoch = args.num_episodes_batch_per_sleep, epochs = 1)
        WS_trainer.sleep_phase_NREM(num_episodes_per_sleep = args.num_episodes_per_sleep,
                               optimizer_semantic_memory = optimizer_semantic_memory, 
                               criterion_semantic_memory = criterion_semantic_memory, 
                               scheduler_semantic_memory = scheduler_semantic_memory,
                               optimizer_generator = optimizer_generator,
                               criterion_generator = criterion_generator,
                               scheduler_generator = scheduler_generator,
                               device = device, 
                               writer = writer,
                               task_id=task_id)
        
        print("Sleep Phase REM...")
        optimizer_semantic_memory = torch.optim.AdamW(semantic_memory.parameters(), lr = args.lr, weight_decay = args.wd)
        criterion_semantic_memory = EntLoss(num_views = args.num_views).to(device)
        scheduler_semantic_memory = torch.optim.lr_scheduler.OneCycleLR(optimizer_semantic_memory, max_lr = args.lr,
                                                                        steps_per_epoch = args.num_episodes_batch_per_sleep, epochs = 1)
        WS_trainer.sleep_phase_REM(num_episodes_per_sleep = args.num_episodes_per_sleep,
                                 optimizer_semantic_memory = optimizer_semantic_memory, 
                                 criterion_semantic_memory = criterion_semantic_memory, 
                                 scheduler_semantic_memory = scheduler_semantic_memory,
                                 device = device, 
                                 writer = writer,
                                 task_id=task_id)

        ### Evaluate model on validation set (seen so far)
        val_dataset = val_tasks[:task_id+1]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        WS_trainer.evaluate_semantic_memory(val_loader, device=device, plot_clusters = True, 
                                            save_dir = os.path.join(args.save_dir,'pseudo_classes_clusters_seen_data'), 
                                            task_id = task_id, mean = args.mean, std = args.std)
        WS_trainer.evaluate_generator(val_loader, device=device, 
                                      save_dir = os.path.join(args.save_dir,'reconstructions_seen_data'),
                                      task_id = task_id, mean = args.mean, std = args.std)
        
        ### Evaluate model on validation set (all data)
        val_dataset = val_tasks[:]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        WS_trainer.evaluate_semantic_memory(val_loader, device=device, measure_cluster_acc=True, plot_clusters = True,
                                            save_dir = os.path.join(args.save_dir,'pseudo_classes_clusters_all_data'), 
                                            task_id = task_id, mean = args.mean, std = args.std)
        WS_trainer.evaluate_generator(val_loader, device=device,
                                      save_dir = os.path.join(args.save_dir,'reconstructions_all_data'),
                                      task_id = task_id, mean = args.mean, std = args.std)
        
        ### Save models
        # save semantic memory
        semantic_memory_state_dict = semantic_memory.module.state_dict()
        torch.save(semantic_memory_state_dict, os.path.join(args.save_dir, f'semantic_memory_taskid_{task_id}.pth'))
        # save view generator
        view_generator_state_dict = view_generator.module.state_dict()
        torch.save(view_generator_state_dict, os.path.join(args.save_dir, f'view_generator_taskid_{task_id}.pth'))

        ### Print time
        print(f'Task {task_id+1} Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    # Final evaluation to print clustering accuracy
    print('\n==> Final evaluation')
    WS_trainer.evaluate_semantic_memory(val_loader, device=device, measure_cluster_acc=True)

    # Close tensorboard writer
    writer.close()

    return None

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

class Episode_Transformations:
    def __init__(self, num_views, zca=False):
        self.num_views = num_views
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if zca: std = [1.0, 1.0, 1.0]
        self.mean = mean
        self.std = std

        # random flip function
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
        
        # function to create first view
        self.create_first_view = transforms.Compose([
                transforms.Resize((224,224)),
                ])
        
        # function to create other views
        self.create_view = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
                ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2)])

        # function to convert to tensor and normalize views
        self.tensor_normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ])
            
    def __call__(self, x):
        views = torch.zeros(self.num_views, 3, 224, 224) # initialize views tensor
        original_image = self.random_flip(x) # randomly flip original image first
        first_view = self.create_first_view(original_image) # create first view
        views[0] = self.tensor_normalize(first_view)
        for i in range(1, self.num_views): # create other views with augmentations
            views[i] = self.tensor_normalize(self.create_view(original_image))
        return views
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Semantic_Memory_Model(torch.nn.Module):
    def __init__(self, features_dim, num_pseudoclasses, proj_dim=4096):
        super().__init__()
        self.features_dim = features_dim
        self.num_pseudoclasses = num_pseudoclasses
        self.proj_dim = proj_dim
        # Global average pooling
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Projector (R) 
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.features_dim, self.proj_dim),
            torch.nn.GroupNorm(32, self.proj_dim),
            torch.nn.Mish(),
            torch.nn.Linear(self.proj_dim, self.proj_dim),
            torch.nn.GroupNorm(32, self.proj_dim),
            torch.nn.Mish()
            )
        # Linear head (F)
        self.linear_head = torch.nn.Linear(self.proj_dim, self.num_pseudoclasses, bias=True)
        self.norm = torch.nn.BatchNorm1d(self.num_pseudoclasses, affine=False)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.projector(x)
        x = self.linear_head(x)
        x = self.norm(x)
        return x
    
class EntLoss(torch.nn.Module):
    def __init__(self, num_views=4, tau=1):
        super(EntLoss, self).__init__()
        self.tau = tau
        self.N = num_views

    def forward(self, episodes_logits):
        episodes_probs = F.softmax(episodes_logits, dim=1)
        episodes_probs = einops.rearrange(episodes_probs, '(b v) c -> b v c', v=self.N).contiguous()
        episodes_sharp_probs = F.softmax(episodes_logits/self.tau, dim=1)
        episodes_sharp_probs = einops.rearrange(episodes_sharp_probs, '(b v) c -> b v c', v=self.N).contiguous()
        B = episodes_probs.size(0)

        consis_loss = 0
        sharp_loss = 0
        div_loss = 0

        for t in range(self.N):
            if t < self.N-1:
                SKL = 0.5 * (self.KL(episodes_probs[:,0], episodes_probs[:,t+1]) + self.KL(episodes_probs[:,t+1], episodes_probs[:,0])) # Simetrized KL anchor based
                consis_loss += SKL
            sharp_loss += self.entropy(episodes_sharp_probs[:,t]).mean() #### Sharpening loss
            mean_across_episodes = episodes_sharp_probs[:,t].mean(dim=0)
            div_loss += self.entropy(mean_across_episodes, dim=0) #### Diversity loss
        consis_loss = consis_loss / (self.N-1) # mean over views
        consis_loss = consis_loss.mean() # mean over episodes
        sharp_loss = sharp_loss / self.N
        div_loss = div_loss / self.N

        return consis_loss, sharp_loss, div_loss

    def KL(self, probs1, probs2, eps = 1e-5):
        kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum(dim=1)
        return kl

    def entropy(self, probs, eps = 1e-5, dim=1):
        H = - (probs * (probs + eps).log()).sum(dim=dim)
        return H


if __name__ == '__main__':
    main()