import argparse
import os, time

import torch
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torchvision

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental

from models_deit3_projcos_causaltr import *
from augmentations import Episode_Transformations, collate_function
from utils import MetricLogger, accuracy, time_duration_print

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet2012')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--num_pretraining_classes', type=int, default=10)
parser.add_argument('--data_order_file_name', type=str, default='./IM1K_data_class_orders/imagenet_class_order_siesta.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Pre-trained folder 
# parser.add_argument('--pretrained_folder', type=str, default='./output/Pretrained_condgen_AND_enc/expts_causal/projcosOPTIcausalpos_3tanh_deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.0008wd@0.05droppath@0.0125_CONDGEN_lr@0.0008wd@0layers@8heads@8dimff@1024dropout@0_seed@0')
# parser.add_argument('--pretrained_folder', type=str, default='./output/Pretrained_condgen_AND_enc/expts_causal/projcosOPTIcausalpos_with1stclsview_3tanh_deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.0008wd@0.05droppath@0.0125_CONDGEN_lr@0.0008wd@0layers@8heads@8dimff@1024dropout@0_seed@0')
# parser.add_argument('--pretrained_folder', type=str, default='./output/Pretrained_condgen_AND_enc/expts_causal/projcosOPTIcausalpos_with1stclsview_Noise0.25drop0.25LS0.1_3tanh_deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.0008wd@0.05droppath@0.0125_CONDGEN_lr@0.0008wd@0layers@8heads@8dimff@1024dropout@0_seed@0')
parser.add_argument('--pretrained_folder', type=str, default='./output/Pretrained_condgen_AND_enc/expts_causal/projcosOPTIcausalpos_with1stclsview_Noise0.25drop0.5_3tanh_deit_tiny_patch16_LS_10c_views@4bs@80epochs100warm@5_ENC_lr@0.0008wd@0.05droppath@0.0125_CONDGEN_lr@0.0008wd@0layers@8heads@8dimff@1024dropout@0_seed@0')

# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--enc_model_checkpoint', type=str, default='view_encoder_epoch99.pth')
# Classifier parameters
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
parser.add_argument('--classifier_model_checkpoint', type=str, default='classifier_epoch99.pth')
# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--condgen_model_checkpoint', type=str, default='cond_generator_epoch99.pth')
# Testing parameters
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--episode_batch_size', type=int, default=80)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--save_dir', type=str, default="testing")
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

def test_on_episodic_data(args, view_encoder, classifier, cond_generator, ep_loader, criterion_sup, device):
    loss_sup_log = MetricLogger('Loss Sup')
    acc1_log = MetricLogger('Top1 ACC')
    acc5_log = MetricLogger('Top5 ACC')
    train_top1_views = {}
    train_top5_views = {}
    for v in range(args.num_views):
        train_top1_views[f"{v+1}views"] = MetricLogger(f'Train Top1 ACC {v+1} views')
        train_top5_views[f"{v+1}views"] = MetricLogger(f'Train Top5 ACC {v+1} views')

    cls_token_mean_val = MetricLogger('CLS Token Mean Value')
    cls_token_std_val = MetricLogger('CLS Token Std Value')
    cls_token_max_val = MetricLogger('CLS Token Max Value')
    cls_token_min_val = MetricLogger('CLS Token Min Value')
    with torch.no_grad():
        for i, (batch_episodes, batch_labels, _) in enumerate(tqdm(ep_loader)):
            batch_episodes_imgs = batch_episodes[0].to(device, non_blocking=True) # (B, V, C, H, W)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True) # (B, V)
            batch_episodes_actions = batch_episodes[1] # (B, V, A)

            B, V, C, H, W = batch_episodes_imgs.shape
            # Flat images
            flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)

            ### Forward pass view encoder
            flat_alltokens = view_encoder(flat_imgs) # (B*V, 1+T, D)
            noflat_alltokens = flat_alltokens.view(B, V, flat_alltokens.size(1), -1) # (B, V, 1+T, D)

            ### Forward pass classifier and calculate acc/loss values
            noflat_clstokens = noflat_alltokens[:, :, 0, :] # (B, V, D)
            noflat_logits = classifier(noflat_clstokens)
            flat_logits = noflat_logits.reshape(B * noflat_logits.size(1), -1)
            flat_labels = batch_episodes_labels.reshape(-1)
            loss_sup = criterion_sup(flat_logits, flat_labels)
            acc1, acc5 = accuracy(flat_logits, flat_labels, topk=(1, 5))
            # Calculate accuracy per view
            acc_per_view = {}
            for v in range(noflat_logits.shape[1]):
                view_logits = noflat_logits[:, v, :]
                view_labels = batch_episodes_labels[:, v]
                view_acc1, view_acc5 = accuracy(view_logits, view_labels, topk=(1, 5))
                acc_per_view[f"{v+1}views"] = [view_acc1.item(), view_acc5.item()]    
            # Accumulate metrics
            loss_sup_log.update(loss_sup.item(), batch_episodes_imgs.size(0))
            acc1_log.update(acc1.item(), batch_episodes_imgs.size(0))
            acc5_log.update(acc5.item(), batch_episodes_imgs.size(0))
            for v in range(args.num_views):
                train_top1_views[f"{v+1}views"].update(acc_per_view[f"{v+1}views"][0], batch_episodes_imgs.size(0))
                train_top5_views[f"{v+1}views"].update(acc_per_view[f"{v+1}views"][1], batch_episodes_imgs.size(0))

            # CLS token statistics
            clstoken_mean = noflat_clstokens.mean(dim=1).mean(dim=0) # (D,)
            clstoken_std = noflat_clstokens.std(dim=1).mean(dim=0) # (D,)
            clstoken_max = noflat_clstokens.max(dim=1).values.mean(dim=0) # (D,)
            clstoken_min = noflat_clstokens.min(dim=1).values.mean(dim=0) # (D,)
            cls_token_mean_val.update(clstoken_mean.mean().item(), batch_episodes_imgs.size(0))
            cls_token_std_val.update(clstoken_std.mean().item(), batch_episodes_imgs.size(0))
            cls_token_max_val.update(clstoken_max.mean().item(), batch_episodes_imgs.size(0))
            cls_token_min_val.update(clstoken_min.mean().item(), batch_episodes_imgs.size(0))

    print(f'\nCLS Token Mean Value: {cls_token_mean_val.avg:.6f}')
    print(f'CLS Token Std Value: {cls_token_std_val.avg:.6f}')
    print(f'CLS Token Max Value: {cls_token_max_val.avg:.6f}')
    print(f'CLS Token Min Value: {cls_token_min_val.avg:.6f}')

    return loss_sup_log, acc1_log, acc5_log, train_top1_views, train_top5_views

def test_on_standard_data(args, view_encoder, classifier, loader, criterion_sup, device):
    loss_sup_log = MetricLogger('Loss Sup')
    acc1_log = MetricLogger('Top1 ACC')
    acc5_log = MetricLogger('Top5 ACC')
    with torch.no_grad():
        for i, (batch_imgs, batch_labels, _) in enumerate(tqdm(loader)):
            batch_imgs = batch_imgs.to(device) # (B, C, H, W)
            batch_labels = batch_labels.to(device) # (B,)
            batch_tensors = view_encoder(batch_imgs) # (B, 1+T, D)
            batch_cls_tokens = batch_tensors[:, 0, :].unsqueeze(1) # (B, 1, D)
            batch_logits = classifier(batch_cls_tokens) # (B, 1, num_classes)
            batch_logits = batch_logits.squeeze(1) # (B, num_classes)
            loss_sup = criterion_sup(batch_logits, batch_labels)
            acc1, acc5 = accuracy(batch_logits, batch_labels, topk=(1, 5))
            # Accumulate metrics
            loss_sup_log.update(loss_sup.item(), batch_imgs.size(0))
            acc1_log.update(acc1.item(), batch_imgs.size(0))
            acc5_log.update(acc5.item(), batch_imgs.size(0))

    return loss_sup_log, acc1_log, acc5_log

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
    args.drop_path = args_pretrained['drop_path']
    args.cond_dropout = args_pretrained['cond_dropout']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make the save dir to be the pretrained folder + testing
    args.save_dir = os.path.join(args.pretrained_folder, args.save_dir)

    # Create save dir folder
    print(args)
    if not os.path.exists(args.save_dir): # create save dir
        os.makedirs(args.save_dir)

    # Calculate batch size per GPU
    args.episode_batch_size_per_gpu = args.episode_batch_size
    # Calculate number of workers per GPU
    args.workers_per_gpu = args.workers

    ### Seed everything
    final_seed = args.seed
    seed_everything(seed=final_seed)

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

    print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    train_dataset_continuum = ImageFolderDataset(traindir)
    val_dataset_continuum = ImageFolderDataset(valdir)
    
    train_ep_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
    val_ep_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std),
                        ])


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

    train_ep_tasks = ClassIncremental(train_dataset_continuum, increment=1, initial_increment=args.num_pretraining_classes, transformations=[train_ep_transform], class_order=data_class_order)
    val_ep_tasks = ClassIncremental(val_dataset_continuum, increment=1, initial_increment=args.num_pretraining_classes, transformations=[val_ep_transform], class_order=data_class_order)
    val_tasks = ClassIncremental(val_dataset_continuum, increment=1, initial_increment=args.num_pretraining_classes, transformations=[val_transform], class_order=data_class_order)
    # Test on the pre-training classes (first task)
    train_ep_dataset = train_ep_tasks[0]
    val_ep_dataset = val_ep_tasks[0]
    val_dataset = val_tasks[0]
    train_ep_loader = torch.utils.data.DataLoader(train_ep_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True,
                                               num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True,
                                               collate_fn=collate_function)
    val_ep_loader = torch.utils.data.DataLoader(val_ep_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                               num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True,
                                               collate_fn=collate_function)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                             num_workers=args.workers_per_gpu, pin_memory=True)

    ### Load Models
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
        print(f'Loading view encoder from {args.pretrained_folder}/{args.enc_model_checkpoint}')
        view_encoder.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.enc_model_checkpoint), map_location=device), strict=True)
    if args.classifier_model_checkpoint is not None:
        print(f'Loading classifier from {args.pretrained_folder}/{args.classifier_model_checkpoint}')
        classifier.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.classifier_model_checkpoint), map_location=device), strict=True)
    if args.condgen_model_checkpoint is not None:
        print(f'Loading conditional generator from {args.pretrained_folder}/{args.condgen_model_checkpoint}')
        cond_generator.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.condgen_model_checkpoint), map_location=device), strict=True)
                                                  
    # ### Print models
    # print('\nView encoder')
    # print(view_encoder)
    # print('\nClassifier')
    # print(classifier)
    # print('\nConditional generator')
    # print(cond_generator)
    # print('\n')

    ### Dataparallel and move models to device
    view_encoder = view_encoder.to(device)
    classifier = classifier.to(device)
    cond_generator = cond_generator.to(device)

    ### Save one batch for plot purposes
    seed_everything(final_seed)  # Reset seed to ensure reproducibility for the batch
    train_episodes_plot, _, _ = next(iter(train_ep_loader))
    val_episodes_plot, _, _ = next(iter(val_ep_loader))

    ### Loss functions
    criterion_sup = torch.nn.CrossEntropyLoss()
    criterion_condgen = torch.nn.MSELoss()

    ### Test step
    view_encoder.eval()
    classifier.eval()
    cond_generator.eval()

    # Train_ep_loader
    train_ep_losssuplog, train_ep_acc1log, train_ep_acc5log, train_ep_top1views, train_ep_top5views = test_on_episodic_data(args, view_encoder, classifier, cond_generator, train_ep_loader, criterion_sup, device)
    print('\nTraining_ep_loader')
    print(f'\tLoss Sup: {train_ep_losssuplog.avg:.6f} -- Top1 ACC: {train_ep_acc1log.avg/100.0:.3f} -- Top5 ACC: {train_ep_acc5log.avg/100.0:.3f}')
    for v in range(args.num_views):
        print(f'\t{v+1} views: Top1 {train_ep_top1views[f"{v+1}views"].avg/100.0:.3f} -- Top5 {train_ep_top5views[f"{v+1}views"].avg/100.0:.3f}')

    # Val_ep_loader
    val_ep_losssuplog, val_ep_acc1log, val_ep_acc5log, val_ep_top1views, val_ep_top5views = test_on_episodic_data(args, view_encoder, classifier, cond_generator, val_ep_loader, criterion_sup, device)
    print('\nValidation_ep_loader')
    print(f'\tLoss Sup: {val_ep_losssuplog.avg:.6f} -- Top1 ACC: {val_ep_acc1log.avg/100.0:.3f} -- Top5 ACC: {val_ep_acc5log.avg/100.0:.3f}')
    for v in range(args.num_views):
        print(f'\t{v+1} views: Top1 {val_ep_top1views[f"{v+1}views"].avg/100.0:.3f} -- Top5 {val_ep_top5views[f"{v+1}views"].avg/100.0:.3f}')

    # Val_loader
    val_losssuplog, val_acc1log, val_acc5log = test_on_standard_data(args, view_encoder, classifier, val_loader, criterion_sup, device)
    print('\nValidation_loader')
    print(f'\tLoss Sup: {val_losssuplog.avg:.6f} -- Top1 ACC: {val_acc1log.avg/100.0:.3f} -- Top5 ACC: {val_acc5log.avg/100.0:.3f}')

    # Save all results in a json file (round values to have 4 decimal places)
    print('\nSaving results...')
    results = {
        'train_ep_loss_sup': round(train_ep_losssuplog.avg, 4),
        'train_ep_top1_acc': round(train_ep_acc1log.avg / 100.0, 4),
        'train_ep_top5_acc': round(train_ep_acc5log.avg / 100.0, 4),
        'train_ep_top1_views': {k: round(v.avg / 100.0, 4) for k, v in train_ep_top1views.items()},
        'train_ep_top5_views': {k: round(v.avg / 100.0, 4) for k, v in train_ep_top5views.items()},
        'val_ep_loss_sup': round(val_ep_losssuplog.avg, 4),
        'val_ep_top1_acc': round(val_ep_acc1log.avg / 100.0, 4),
        'val_ep_top5_acc': round(val_ep_acc5log.avg / 100.0, 4),
        'val_ep_top1_views': {k: round(v.avg / 100.0, 4) for k, v in val_ep_top1views.items()},
        'val_ep_top5_views': {k: round(v.avg / 100.0, 4) for k, v in val_ep_top5views.items()},
        'val_loss_sup': round(val_losssuplog.avg, 4),
        'val_top1_acc': round(val_acc1log.avg / 100.0, 4),
        'val_top5_acc': round(val_acc5log.avg / 100.0, 4)
    }


    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)



    return None

if __name__ == '__main__':
    main()