import argparse
import os, time, random

import torch
from my_transforms import transforms
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from my_continuum.datasets import ImageFolderDataset
from my_continuum import ClassIncremental, InstanceIncremental

from models import *
from wake_sleep_trainer import Wake_Sleep_trainer

from tensorboardX import SummaryWriter
from PIL import ImageFilter, ImageOps, ImageFilter, Image
import einops
import numpy as np
import json
import matplotlib.pyplot as plt

# turn off warnings
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='SSCL Wake-Sleep veridical')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--iid', action='store_true')
# View encoder parameters
parser.add_argument('--view_encoder_model_name', type=str, default='resnet18')
parser.add_argument('--pretrained_view_encoder', type=str, default='./pretrained_models/SSL_100epochs100stop_12views_0.02lr_128bs_seed0/encoder_epoch99.pth') ######## None
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
parser.add_argument('--episode_batch_size', type=int, default=56)
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
    # ori_img = Image.open('ouput/astronaut.jpg')
    episode_transform = Episode_Transformations(num_views = args.num_views)
    # episode_example, episode_action_bbox = episode_transform(ori_img)

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
        val_generator_tasks = InstanceIncremental(val_parent_dataset, nb_tasks=args.num_tasks, transformations=train_tranform)
    else: # non-iid data (class incremental)
        assert args.num_classes % args.num_tasks == 0, "Number of classes must be divisible by number of tasks"
        class_increment = int(args.num_classes // args.num_tasks)
        train_tasks = ClassIncremental(train_parent_dataset, increment = class_increment, 
                                    transformations = train_tranform, class_order = data_class_order)
        val_tasks = ClassIncremental(val_parent_dataset, increment = class_increment, 
                                    transformations = val_transform, class_order = data_class_order)
        val_generator_tasks = ClassIncremental(val_parent_dataset, increment = class_increment,
                                    transformations = train_tranform, class_order = data_class_order)

    # plot crops (unnormalization is missing)
    # for i in range(50):
    #     a = train_tasks[0][i]
    #     original_image = a[0][0][0]
    #     cropped_image = a[0][0][1]
    #     bbox = a[0][1][1]
    #     # make bbox values int
    #     bbox = [int(i) for i in bbox]
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
    #     plt.savefig(f'plot_images_{i}.png', bbox_inches='tight')
    #     plt.close()


    ### Load view_encoder, semantic_memory, and view_generator
    print('\n==> Preparing model...')
    view_encoder = eval(args.view_encoder_model_name)(zero_init_residual = True, get_tensor_before_avgpool=True)
    semantic_memory = eval('Semantic_Memory_Model')(view_encoder.fc.weight.shape[1], num_pseudoclasses = args.num_pseudoclasses, proj_dim = args.proj_dim)
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
        # del train_dataset, train_loader # Dump real images after getting embeddings

        ### SLEEP PHASE ###
        print("Sleep Phase NREM...")
        # Semantic memory optimizer, criterion, and scheduler
        optimizer_semantic_memory = torch.optim.AdamW(semantic_memory.parameters(), lr = args.lr, weight_decay = args.wd)
        criterion_semantic_memory = EntLoss(num_views = args.num_views).to(device)
        scheduler_semantic_memory = torch.optim.lr_scheduler.OneCycleLR(optimizer_semantic_memory, max_lr = args.lr,
                                                                        steps_per_epoch = args.num_episodes_batch_per_sleep, epochs = 1)
        # Generator optimizer, criterion, and scheduler
        optimizer_generator = torch.optim.AdamW(view_generator.parameters(), lr = args.generator_lr, weight_decay = args.generator_wd)
        criterion_generator = torch.nn.MSELoss().to(device)
        scheduler_generator = torch.optim.lr_scheduler.OneCycleLR(optimizer_generator, max_lr = args.generator_lr, 
                                                                  steps_per_epoch = args.num_episodes_batch_per_sleep, epochs = 1)
        # Sleep phase
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

        ### Evaluation 
        print("Evaluation...")
        # Evaluate model on validation set (seen so far)
        val_dataset = val_tasks[:task_id+1]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        WS_trainer.evaluate_semantic_memory(val_loader, device=device, calc_cluster_acc=False, plot_clusters = True,
                                            save_dir = os.path.join(args.save_dir,'pseudo_classes_clusters_seen_data'), 
                                            task_id = task_id, mean = args.mean, std = args.std)
        val_generator_dataset = val_generator_tasks[:task_id+1]
        val_generator_loader = torch.utils.data.DataLoader(val_generator_dataset, batch_size = 32, shuffle = False, num_workers = args.workers)
        WS_trainer.evaluate_generator(val_generator_loader, device=device, 
                                      save_dir = os.path.join(args.save_dir,'reconstructions_seen_data'),
                                      task_id = task_id, mean = args.mean, std = args.std)
        
        # Evaluate model on validation set (all data)
        val_dataset = val_tasks[:]
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = args.workers)
        WS_trainer.evaluate_semantic_memory(val_loader, device = device, calc_cluster_acc = args.num_classes==args.num_pseudoclasses, plot_clusters = True,
                                            save_dir = os.path.join(args.save_dir,'pseudo_classes_clusters_all_data'), 
                                            task_id = task_id, mean = args.mean, std = args.std)
        val_generator_dataset = val_generator_tasks[:]
        val_generator_loader = torch.utils.data.DataLoader(val_generator_dataset, batch_size = 32, shuffle = False, num_workers = args.workers)
        WS_trainer.evaluate_generator(val_generator_loader, device=device,
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

    # Close tensorboard writer
    writer.close()

    print('\n==> END')

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
        mean=[0.485, 0.456, 0.406] # ImageNet-1k mean
        std=[0.229, 0.224, 0.225] # ImageNet-1k std
        if zca: 
            std = [1.0, 1.0, 1.0]
        self.mean = mean
        self.std = std

        # random flip function
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
        
        # resize to 224 function
        self.resize_224 = transforms.Resize((224,224))
        
        # random resized crops function that yields back the bbox used
        self.random_resized_crop = transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3./4., 4./3.))

        # self.create_view = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                                 saturation=0.2, hue=0.1)],
        #         p=0.8
        #         ),
        #     transforms.RandomGrayscale(p=0.2),
        #     GaussianBlur(p=0.1),
        #     Solarization(p=0.2)])

        # Convert to tensor and normalize
        self.tensor_normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ])
            
    def __call__(self, x):

        # initialize views tensor, and actions tensor
        views = torch.zeros(self.num_views, 3, 224, 224)
        action_cropbb = torch.zeros(self.num_views, 4)

        # randomly flip original image first
        original_image = self.random_flip(x) 

        # create first view (unchanged original image resized to 224, action vectors indicating no transformation)
        first_view = self.resize_224(original_image) 
        views[0] = self.tensor_normalize(first_view)
        action_cropbb[0] = torch.tensor([0, 0, 224, 224])

        # create other views with augmentations from first view
        for i in range(1, self.num_views):
            aug_view, bbox = self.random_resized_crop(first_view)

            views[i] = self.tensor_normalize(aug_view)
            action_cropbb[i] = bbox

        return views, action_cropbb
    
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