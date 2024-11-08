import argparse
import os, time
import random
import einops
import json

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import utils
from models import *
from loss_functions import TwistLossViewExpanded
from augmentations import Episode_Transformations
from evaluate_cluster import evaluate as eval_pred

from PIL import ImageFilter
from PIL import Image, ImageOps, ImageFilter

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description='SSL on episodes offline Training')
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)

parser.add_argument('--model_name', type=str, default='resnet18')
parser.add_argument('--proj_dim', type=int, default=2048)
parser.add_argument('--num_pseudoclasses', type=int, default=10)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--wd', type=float, default=1.5e-6)
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=12)
parser.add_argument('--tau', type=float, default=0.8)

parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--save_frequency', type=int, default=10) # epochs
parser.add_argument('--print_frequency', type=int, default=10) # iterations
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="output/run_SSL_on_episodes")

def main():

    ### Parse arguments
    args = parser.parse_args()
    args.save_dir_clusters = os.path.join(args.save_dir, 'pseudo_classes_clusters')

    ### Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ### Print and save args
    print(args)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Seed everything
    utils.seed_everything(seed=args.seed)

    ### Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Define tensoboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load data
    print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    train_transform = Episode_Transformations(num_views = args.num_views)
    args.mean = train_transform.mean
    args.std = train_transform.std
    train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
    val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std),
                ])
    val_dataset = datasets.ImageFolder(valdir, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size, 
                                            shuffle=True, num_workers=args.workers, pin_memory=True,
                                            persistent_workers=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size, 
                                            shuffle=False, num_workers=args.workers, pin_memory=True)

    ### Define SSL network model
    print('\n==> Preparing model...')
    encoder = eval(args.model_name)(zero_init_residual = True)
    model = eval('Semantic_Memory_Model')(encoder, 
                                          num_pseudoclasses = args.num_pseudoclasses, 
                                          proj_dim = args.proj_dim)
    print(model)
    model = torch.nn.DataParallel(model)
    model.to(device)

    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = TwistLossViewExpanded(num_views=args.num_views, tau=args.tau).to(device)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr*1e-6, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    ### Perform validation on random model
    print('\n==> Validation on random model')
    epoch=0
    val_metrics = validate(args, model, val_loader, device, epoch=epoch, save_dir_clusters = args.save_dir_clusters)
    nmi = val_metrics['NMI']
    ami = val_metrics['AMI']
    ari = val_metrics['ARI']
    fscore = val_metrics['F']
    adjacc = val_metrics['ACC']
    top5 = val_metrics['ACC-Top5']
    print(f'Epoch [{epoch}]')
    print(f'Validation -- NMI: {nmi:.4f} -- AMI: {ami:.4f} -- ARI: {ari:.4f} -- F: {fscore:.4f} -- ACC: {adjacc:.4f} -- ACC-Top5: {top5:.4f}')
    writer.add_scalar('Metric NMI', nmi, epoch)
    writer.add_scalar('Metric AMI', ami, epoch)
    writer.add_scalar('Metric ARI', ari, epoch)
    writer.add_scalar('Metric F', fscore, epoch)
    writer.add_scalar('Metric ACC', adjacc, epoch)
    writer.add_scalar('Metric ACC-Top5', top5, epoch)
    # Save model at random init
    state_dict = model.module.state_dict()
    torch.save(state_dict, os.path.join(args.save_dir, f'episodicSSL_model_epoch{epoch}.pth'))
    encoder_state_dict = model.module.encoder.state_dict()
    torch.save(encoder_state_dict, os.path.join(args.save_dir, f'encoder_epoch{epoch}.pth'))

    ### Train model
    print('\n==> Training model')
    init_time = time.time()
    train_loss_all = []
    nmi_all = []
    ACC_all = []
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train_loss = train_step(args, model, train_loader, optimizer, criterion, scheduler, epoch, device, writer)
        val_metrics = validate(args, model, val_loader, device, epoch, save_dir_clusters = args.save_dir_clusters)

        # Print results
        nmi = val_metrics['NMI']
        ami = val_metrics['AMI']
        ari = val_metrics['ARI']
        fscore = val_metrics['F']
        adjacc = val_metrics['ACC']
        top5 = val_metrics['ACC-Top5']
        print(f'Epoch [{epoch}] Total Train Loss per Epoch: {train_loss:.6f}')
        print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')
        print(f'NMI: {nmi:.4f}, AMI: {ami:.4f}, ARI: {ari:.4f}, F: {fscore:.4f}, ACC: {adjacc:.4f}, ACC-Top5: {top5:.4f}')
        
        # Save model
        if (epoch % args.save_frequency == 0):
            state_dict = model.module.state_dict()
            torch.save(state_dict, os.path.join(args.save_dir, f'episodicSSL_model_epoch{epoch}.pth'))
            encoder_state_dict = model.module.encoder.state_dict()
            torch.save(encoder_state_dict, os.path.join(args.save_dir, f'encoder_epoch{epoch}.pth'))

        # Add to tensorboard
        writer.add_scalar('Total Loss per epoch', train_loss, epoch)
        writer.add_scalar('Metric NMI', nmi, epoch)
        writer.add_scalar('Metric AMI', ami, epoch)
        writer.add_scalar('Metric ARI', ari, epoch)
        writer.add_scalar('Metric F', fscore, epoch)
        writer.add_scalar('Metric ACC', adjacc, epoch)
        writer.add_scalar('Metric ACC-Top5', top5, epoch)

        # Save results
        train_loss_all.append(train_loss)
        nmi_all.append(nmi)
        ACC_all.append(adjacc)
        np.save(os.path.join(args.save_dir, 'train_loss.npy'), np.array(train_loss_all))
        np.save(os.path.join(args.save_dir, 'nmi.npy'), np.array(nmi_all))
        np.save(os.path.join(args.save_dir, 'ACC.npy'), np.array(ACC_all))

    writer.close()
    print('\n==> Training finished')

    return None

def train_step(args, model, train_loader, optimizer, criterion, scheduler, epoch, device, writer=None):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # forward pass
        episode_imgs, batch_labels = batch
        episode_imgs = episode_imgs.to(device)

        episode_imgs = einops.rearrange(episode_imgs, 'b v c h w -> (b v) c h w').contiguous() # all episodes views in one batch
        episodes_logits = model(episode_imgs)

        # loss calculation
        consis_loss, sharp_loss, div_loss = criterion(episodes_logits)
        loss = consis_loss + sharp_loss - div_loss

        # backward pass
        loss.backward()
        optimizer.step()

        # accumulate loss
        total_loss += loss.item()

        # print each loss and total loss
        if i % args.print_frequency == 0:
            print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- lr: {scheduler.get_last_lr()[0]:.6f} -- Consis: {consis_loss.item():.6f}' + 
                  f' -- Sharp: {sharp_loss.item():.6f} -- Div: {div_loss.item():.6f} -- Total: {loss.item():.6f}')
            if writer is not None:
                writer.add_scalar('Consistency Loss', consis_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Sharpness Loss', sharp_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Diversity Loss', div_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Total Loss', loss.item(), epoch*len(train_loader)+i)
        # scheduler step      
        scheduler.step()

    total_loss /= len(train_loader)
    return total_loss

def validate(args, model, val_loader, device, epoch=0, save_dir_clusters='clusters/'):
    model.eval()
    
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            imgs = images.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_logits.append(logits.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_labels.append(targets)
    all_logits = torch.cat(all_logits).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_indices = np.arange(len(all_preds))
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(all_labels.astype(int), all_preds.astype(int), calc_acc=(args.num_pseudoclasses==args.num_classes), total_probs=all_probs)

    if (epoch==0) or (epoch % args.save_frequency == 0):
        os.makedirs(save_dir_clusters, exist_ok=True)
        ### plot 25 images per cluster
        for i in range(10): # only for 10 pseudo classes
            pseudoclass_imgs_indices = all_indices[all_preds==i]
            if len(pseudoclass_imgs_indices) > 0:
                pseudoclass_imgs_indices = np.random.choice(pseudoclass_imgs_indices, min(25, len(pseudoclass_imgs_indices)), replace=False)
                pseudoclass_imgs = [val_loader.dataset[j][0] for j in pseudoclass_imgs_indices]
                # psudoclass images are the output of the transform already. So we need to reverse the normalization
                pseudoclass_imgs = [transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in pseudoclass_imgs]
                # stack images
                pseudoclass_imgs = torch.stack(pseudoclass_imgs, dim=0)
                # plot a grid of images 5x5
                grid = torchvision.utils.make_grid(pseudoclass_imgs, nrow=5)
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
            else: # Save a black image
                grid = Image.new('RGB', (224*5, 224*5), (0, 0, 0))
            image_name = f'pseudoclass_{i}_epoch_{epoch}.png'
            grid.save(os.path.join(args.save_dir, f'pseudo_classes_clusters', image_name))

        ### Plot all logits in a 2D space. (PCA, then, t-SNE).
        tsne = TSNE(n_components=2, random_state=0)
        all_logits_2d = tsne.fit_transform(all_logits)
        # Legend is cluster ID
        plt.figure(figsize=(8, 8))
        clusters_IDs = np.unique(all_preds)
        for i in clusters_IDs:
            indices = all_preds==i
            plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Cluster {i}', alpha=0.75, s=20, color=sns.color_palette("husl", args.num_pseudoclasses)[i])
        plt.title(f'Logits 2D space TaskID Epoch {epoch}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(save_dir_clusters, f'logits_2d_space_clusters_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()
        # Legend is GT class
        plt.figure(figsize=(8, 8))
        labels_IDs = np.unique(all_labels)
        for i in labels_IDs:
            indices = all_labels==i
            plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20)
        plt.title(f'Logits 2D space Epoch {epoch}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(save_dir_clusters, f'logits_2d_space_labels_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()

        ### Plot headmap showing cosine similarity matrix of each cluster weights
        clusters_weights = model.module.linear_head.weight.data.cpu().squeeze()
        clusters_weights = F.normalize(clusters_weights, p=2, dim=1)
        clusters_cosine_sim = torch.mm(clusters_weights, clusters_weights.T)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(clusters_cosine_sim, cmap='viridis', ax=ax, annot= args.num_pseudoclasses==10 , vmax=1, vmin=-1)
        plt.title(f'Cosine Similarity Matrix Epoch {epoch}')
        plt.xlabel('Cluster ID')
        plt.ylabel('Cluster ID')
        plt.savefig(os.path.join(save_dir_clusters, f'cosine_similarity_matrix_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()

        ### Plot number of samples per cluster with color per class
        dict_class_vs_clusters = {}
        for i in labels_IDs:
            dict_class_vs_clusters[f'Class {i}'] = []
            for j in range(args.num_pseudoclasses):
                indices = (all_labels==i) & (all_preds==j)
                dict_class_vs_clusters[f'Class {i}'].append(np.sum(indices))
        df = pd.DataFrame(dict_class_vs_clusters)
        df.plot.bar(stacked=True, figsize=(10, 8))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'Number of samples per cluster per class Epoch {epoch}')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of samples')
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(save_dir_clusters, f'number_samples_per_cluster_per_class_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()

    return {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5}



if __name__ == '__main__':
    main()