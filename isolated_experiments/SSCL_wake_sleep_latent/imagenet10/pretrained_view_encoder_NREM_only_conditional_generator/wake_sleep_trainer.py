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

class Wake_Sleep_trainer:
    def __init__(self, view_encoder, semantic_memory, view_generator, episode_batch_size):

        self.view_encoder = view_encoder
        self.semantic_memory = semantic_memory
        self.view_generator = view_generator

        self.episodic_memory_tensors = torch.empty(0)
        self.episodic_memory_cropbboxs = torch.empty(0)

        self.episode_batch_size = episode_batch_size
    
    def wake_phase(self, incoming_dataloader, device):
        ''' 
        Collect tensors and action codes in episodic memory
        '''
        self.view_encoder.eval() 
        aux_memory_tensors = {}
        aux_memory_cropbboxs = {}
        for i, (episode_batch, _ , _ ) in enumerate(incoming_dataloader):
            episode_batch_imgs = episode_batch[0]
            episode_batch_cropbboxs = episode_batch[1]

            b = episode_batch_imgs.size(0)
            v = episode_batch_imgs.size(1)

            # forward pass to collect image representations as tensors
            episode_batch_imgs = episode_batch_imgs.to(device)
            episode_batch_imgs = einops.rearrange(episode_batch_imgs, 'b v c h w -> (b v) c h w').contiguous() # all episodes and all views in one batch of (b v)
            with torch.no_grad(): 
                episode_batch_tensors  = self.view_encoder(episode_batch_imgs)
            episode_batch_tensors = einops.rearrange(episode_batch_tensors, '(b v) c h w -> b v c h w', b=b, v=v).contiguous() # back to batch of episodes
            aux_memory_tensors[i] = episode_batch_tensors.cpu()

            # collect cropbboxs
            aux_memory_cropbboxs[i] = episode_batch_cropbboxs

        # Concatenate all tensors in episodic memory
        aux_memory_tensors = list(aux_memory_tensors.values())
        aux_memory_tensors = torch.cat(aux_memory_tensors, dim=0)
        self.episodic_memory_tensors = torch.cat([self.episodic_memory_tensors, aux_memory_tensors], dim=0)

        # Concatenate all cropbboxs in episodic memory
        aux_memory_cropbboxs = list(aux_memory_cropbboxs.values())
        aux_memory_cropbboxs = torch.cat(aux_memory_cropbboxs, dim=0)
        self.episodic_memory_cropbboxs = torch.cat([self.episodic_memory_cropbboxs, aux_memory_cropbboxs], dim=0)

        return None
    
    def sleep_phase_NREM(self, 
                        num_episodes_per_sleep, 
                        optimizer_semantic_memory, criterion_semantic_memory, scheduler_semantic_memory, 
                        optimizer_generator, criterion_generator, scheduler_generator, 
                        device, writer=None, task_id=None):
        '''
        Train model using episodic memory
        '''

        self.view_encoder.eval()
        self.view_generator.train()
        self.semantic_memory.train()
        
        # freeze view encoder
        for param in self.view_encoder.parameters():
            param.requires_grad = False
        # unfreeze generator
        for param in self.view_generator.parameters():
            param.requires_grad = True
        # unfreeze semantic memory
        for param in self.semantic_memory.parameters():
            param.requires_grad = True

        v = self.episodic_memory_tensors.size(1)

        # Sample sleep episodes idxs from episodic_memory (Uniform sampling with replacement)
        weights = torch.ones(len(self.episodic_memory_tensors))
        sampled_episodes_idxs = list(WeightedRandomSampler(weights, num_episodes_per_sleep, replacement=True))

        # Train model on sleep episodes a bacth at a time
        for i in range(0, num_episodes_per_sleep, self.episode_batch_size):
            batch_idxs = sampled_episodes_idxs[i:i+self.episode_batch_size]

            ## Forward pass on Semantic Memory
            # Gather tensors
            batch_tensors = self.episodic_memory_tensors[batch_idxs]
            batch_tensors = einops.rearrange(batch_tensors, 'b v c h w -> (b v) c h w').contiguous() # all episodes and all views in one batch of (b v)
            batch_tensors = batch_tensors.to(device)
            # Forward pass
            batch_logits = self.semantic_memory(batch_tensors)

            ## Forward pass on conditioned view generator
            # Gather all first view tensors and repeat them for all views in the episode
            batch_firstview_tensors = self.episodic_memory_tensors[batch_idxs][:,0,:,:,:]
            batch_firstview_tensors = batch_firstview_tensors.unsqueeze(1) # add view dimension
            batch_firstview_tensors = batch_firstview_tensors.repeat(1, v, 1, 1, 1) # repeat first view tensor for all views
            batch_firstview_tensors = einops.rearrange(batch_firstview_tensors, 'b v c h w -> (b v) c h w').contiguous() # all episodes and all views in one batch of (b v)
            batch_firstview_tensors = batch_firstview_tensors.to(device)
            # Gather action code cropbboxs
            batch_cropbboxs = self.episodic_memory_cropbboxs[batch_idxs]
            batch_cropbboxs = einops.rearrange(batch_cropbboxs, 'b v d -> (b v) d').contiguous() # all episodes and all views in one batch of (b v)
            batch_cropbboxs = batch_cropbboxs.to(device)
            # Forward pass
            batch_recon_imgs = self.view_generator(batch_firstview_tensors, batch_cropbboxs)
            # batch_recon_imgs = self.view_generator(batch_tensors, batch_cropbboxs) ################################## (sanity check)
            batch_recon_tensors = self.view_encoder(batch_recon_imgs)

            ## Backwad pass on semantic memory
            optimizer_semantic_memory.zero_grad()
            consis_loss, sharp_loss, div_loss = criterion_semantic_memory(batch_logits)
            semantic_loss = consis_loss + sharp_loss - div_loss
            semantic_loss.backward()
            optimizer_semantic_memory.step()
            scheduler_semantic_memory.step()

            ## Backward pass on conditioned view generator
            optimizer_generator.zero_grad()
            recon_loss = criterion_generator(batch_recon_tensors, batch_tensors)
            recon_loss.backward()
            optimizer_generator.step()
            scheduler_generator.step()

            if i==0 or (i//self.episode_batch_size) % 10 == 0:
                current_episode_idx = min(i+self.episode_batch_size,num_episodes_per_sleep)
                print(f'Episode [{current_episode_idx}/{num_episodes_per_sleep}] ' +
                      f'-- SM lr: {scheduler_semantic_memory.get_last_lr()[0]:.6f} -- Consis: {consis_loss.item():.6f} -- Sharp: {sharp_loss.item():.6f} -- Div: {div_loss.item():.6f} -- SM Total loss: {semantic_loss.item():.6f} ' + 
                      f'-- VG lr: {scheduler_generator.get_last_lr()[0]:.6f} -- VG Loss: {recon_loss.item():.6f}')
                
                if writer is not None and task_id is not None:
                    writer.add_scalar('Consistency Loss', consis_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Sharpness Loss', sharp_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Diversity Loss', div_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('SM Total Loss', semantic_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('VG Loss', recon_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)

        return None
    
    def evaluate_semantic_memory(self, val_loader, device, calc_cluster_acc=False,
                       plot_clusters=False, save_dir=None, task_id=None, mean=None, std=None):
        '''
        Evaluate semantic memory model on validation set
        '''
        self.semantic_memory.eval()
        self.view_encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(val_loader):
                images = images.to(device)
                logits = self.semantic_memory(self.view_encoder(images))
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
        if calc_cluster_acc: print(f'Metrics on all val data -- NMI: {nmi:.4f}, AMI: {ami:.4f}, ARI: {ari:.4f}, F: {fscore:.4f}, ACC: {adjacc:.4f}, ACC-Top5: {top5:.4f}')

        if plot_clusters:
            assert save_dir is not None
            assert task_id is not None
            assert mean is not None
            assert std is not None
            # plot 25 images per cluster
            os.makedirs(save_dir, exist_ok=True)
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
                grid.save(os.path.join(save_dir, image_name))

                # summary of clusters using a stripplot
                data_dict = {'Cluster ID': all_preds, 'GT Class': all_labels}
                plt.figure(figsize=(15, 5))
                sns.stripplot(data=data_dict, x='Cluster ID', y='GT Class', size=4, jitter=0.15, alpha=0.6) 
                ylist = np.unique(all_labels)
                plt.yticks(ticks=ylist, labels=ylist)
                if max(all_preds) > 10:
                    plt.xticks(rotation=90)
                plt.grid()
                name = save_dir.split('/')[-1]
                plt.title(f'{name}\nCluster Summary TaskID: {task_id}')
                plt.savefig(os.path.join(save_dir, f'cluster_summary_taskid_{task_id}.png'), bbox_inches='tight')
                plt.close()

    def evaluate_generator(self, loader, device, save_dir=None, task_id=None, mean=None, std=None):
        '''
        Evaluate view generator model
        '''
        self.view_generator.eval()
        self.view_encoder.eval()

        os.makedirs(save_dir, exist_ok=True)

        all_episodes = []
        all_cropbboxs = []
        all_labels = []
        classes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        with torch.no_grad():
            for i, (batch, targets, _ ) in enumerate(loader):
                if targets[0] in classes_to_plot:
                    all_episodes.append(batch[0][0].unsqueeze(0))
                    all_cropbboxs.append(batch[1][0].unsqueeze(0))
                    all_labels.append(targets[0])
                    classes_to_plot.remove(targets[0])

        all_episodes = torch.cat(all_episodes, dim=0)
        all_cropbboxs = torch.cat(all_cropbboxs, dim=0)
        all_labels = np.array(all_labels)

        num_views = all_episodes.size(1)
        for i in range(all_episodes.shape[0]):
            episode_imgs = all_episodes[i]
            episode_cropbboxs = all_cropbboxs[i]
            episode_gtclass = all_labels[i]

            episode_first_view_imgs = episode_imgs[0,:,:,:]
            episode_first_view_imgs = episode_first_view_imgs.unsqueeze(0) # add view dimension
            episode_first_view_imgs = episode_first_view_imgs.repeat(num_views, 1, 1, 1) # repeat first view tensor for all views
            episode_imgs_recon = self.view_generator(self.view_encoder(episode_first_view_imgs.to(device)), episode_cropbboxs.to(device)).cpu()
            # episode_imgs_recon = self.view_generator(self.view_encoder(episode_imgs.to(device)), episode_cropbboxs.to(device)).cpu() ################################## (sanity check)

            episode_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_imgs]
            episode_imgs_recon = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_imgs_recon]
            episode_imgs = torch.stack(episode_imgs, dim=0)
            episode_imgs_recon = torch.stack(episode_imgs_recon, dim=0)

            grid = torchvision.utils.make_grid(torch.cat([episode_imgs, episode_imgs_recon], dim=0), nrow=num_views)
            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            grid = Image.fromarray(grid)
            image_name = f'class_{episode_gtclass}_original_reconstructed_images_taskid_{task_id}.png'
            grid.save(os.path.join(save_dir, image_name))

        return None