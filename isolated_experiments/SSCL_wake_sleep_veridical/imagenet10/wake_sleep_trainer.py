import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler

from evaluate_cluster import evaluate as eval_pred

import einops
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="whitegrid")

class Wake_Sleep_trainer:
    def __init__(self, model, episode_batch_size):
        self.model = model
        self.episodic_memory = torch.empty(0)
        self.episode_batch_size = episode_batch_size
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        aux_memory = {}
        for i, (episode_batch, _ , _ ) in enumerate(incoming_dataloader):
            aux_memory[i] = episode_batch

        aux_memory = list(aux_memory.values())
        aux_memory = torch.cat(aux_memory, dim=0)
        self.episodic_memory = torch.cat([self.episodic_memory, aux_memory], dim=0)

        return None
    
    def sleep_phase(self, num_episodes_per_sleep, optimizer, criterion, scheduler, device, writer=None, task_id=None):
        '''
        Train model on episodic memory
        '''
        self.model.train()

        # Sample sleep episodes idxs from episodic_memory (Uniform sampling with replacement)
        weights = torch.ones(len(self.episodic_memory))
        sampled_episodes_idxs = list(WeightedRandomSampler(weights, num_episodes_per_sleep, replacement=True))

        # Train model on sleep episodes a bacth at a time
        for i in range(0, num_episodes_per_sleep, self.episode_batch_size):
            batch_idxs = sampled_episodes_idxs[i:i+self.episode_batch_size]
            batch_episodes = self.episodic_memory[batch_idxs]
            batch_images = einops.rearrange(batch_episodes, 'b v c h w -> (b v) c h w').contiguous() # all episodes and all views in one batch of (b v)
            batch_images.to(device)
            batch_logits = self.model(batch_images)

            optimizer.zero_grad()
            consis_loss, sharp_loss, div_loss = criterion(batch_logits)
            # loss = consis_loss + sharp_loss - div_loss
            # loss = consis_loss + sharp_loss + div_loss  # I do + when doing div thershold stuff. The minus went inside the criterion function

            loss = consis_loss + sharp_loss + torch.max(torch.tensor(0), div_loss - criterion.div_entropy_upper) + torch.max(torch.tensor(0), -(div_loss - criterion.div_entropy_lower))

            loss.backward()
            optimizer.step()
            scheduler.step()

            if i==0 or (i//self.episode_batch_size) % 5 == 0:
                current_episode_idx = min(i+self.episode_batch_size,num_episodes_per_sleep)
                print(f'Episode [{current_episode_idx}/{num_episodes_per_sleep}] -- lr: {scheduler.get_last_lr()[0]:.6f} -- Consis: {consis_loss.item():.6f}' + 
                    f' -- Sharp: {sharp_loss.item():.6f} -- Div: {div_loss.item():.6f} -- Total: {loss.item():.6f}')
                
                if writer is not None and task_id is not None:
                    writer.add_scalar('Consistency Loss', consis_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Sharpness Loss', sharp_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Diversity Loss', div_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Total Loss', loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)

        return None
    
    def evaluate_model(self, val_loader, device, calc_cluster_acc=False,
                       plot_clusters=False, save_dir_clusters=None, task_id=None, mean=None, std=None):
        '''
        Evaluate model on validation set
        '''
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(val_loader):
                images = images.to(device)
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_labels.append(targets)
        
        all_preds = torch.cat(all_preds).numpy()
        all_indices = np.arange(len(all_preds))
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(all_labels.astype(int), all_preds.astype(int), calc_acc=calc_cluster_acc, total_probs=all_probs)
        if calc_cluster_acc: print(f'NMI: {nmi:.4f}, AMI: {ami:.4f}, ARI: {ari:.4f}, F: {fscore:.4f}, ACC: {adjacc:.4f}, ACC-Top5: {top5:.4f}')

        if plot_clusters:
            assert save_dir_clusters is not None
            assert task_id is not None
            assert mean is not None
            assert std is not None
            # plot 25 images per cluster
            os.makedirs(save_dir_clusters, exist_ok=True)
            for i in range(10): # Only for the first 10 pseudoclasses
                pseudoclass_imgs_indices = all_indices[all_preds==i]
                if len(pseudoclass_imgs_indices) > 0:
                    pseudoclass_imgs_indices = np.random.choice(pseudoclass_imgs_indices, min(25, len(pseudoclass_imgs_indices)), replace=False)
                    pseudoclass_imgs = [val_loader.dataset[j][0] for j in pseudoclass_imgs_indices]
                    # psudoclass images are the output of the transform already. So we need to reverse the normalization
                    pseudoclass_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in pseudoclass_imgs]
                    # stack images
                    pseudoclass_imgs = torch.stack(pseudoclass_imgs, dim=0)
                    # plot a grid of images 5x5
                    grid = torchvision.utils.make_grid(pseudoclass_imgs, nrow=5)
                    grid = grid.permute(1, 2, 0).cpu().numpy()
                    grid = (grid * 255).astype(np.uint8)
                    grid = Image.fromarray(grid)
                else: # Save a black image
                    grid = Image.new('RGB', (224*5, 224*5), (0, 0, 0))
                image_name = f'pseudoclass_{i}_taskid_{task_id}.png'
                grid.save(os.path.join(save_dir_clusters, image_name))

            # summary of clusters using a stripplot
            data_dict = {'Cluster ID': all_preds, 'GT Class': all_labels}
            plt.figure(figsize=(15, 5))
            sns.stripplot(data=data_dict, x='Cluster ID', y='GT Class', size=4, jitter=0.15, alpha=0.6) 
            ylist = np.unique(all_labels)
            plt.yticks(ticks=ylist, labels=ylist)
            if max(all_preds) > 10:
                plt.xticks(rotation=90)
            plt.grid()
            name = save_dir_clusters.split('/')[-1]
            plt.title(f'{name}\nCluster Summary TaskID: {task_id}')
            plt.savefig(os.path.join(save_dir_clusters, f'cluster_summary_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

        return None