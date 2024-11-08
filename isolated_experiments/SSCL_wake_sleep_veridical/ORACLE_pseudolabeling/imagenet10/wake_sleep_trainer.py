import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler

from evaluate_cluster import evaluate as eval_pred
from utils import intra_cluster_distance, inter_cluster_distance, encode_label

import einops
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

class Wake_Sleep_trainer:
    def __init__(self, model, episode_batch_size):
        self.model = model
        self.episodic_memory = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)
        self.episode_batch_size = episode_batch_size
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        aux_memory = {}
        aux_memory_labels = {}
        for i, (episode_batch, labels , _ ) in enumerate(incoming_dataloader):
            aux_memory[i] = episode_batch
            episode_labels = labels.unsqueeze(1).repeat(1, episode_batch.size(1))
            aux_memory_labels[i] = episode_labels


        aux_memory = list(aux_memory.values())
        aux_memory = torch.cat(aux_memory, dim=0)
        self.episodic_memory = torch.cat([self.episodic_memory, aux_memory], dim=0)

        aux_memory_labels = list(aux_memory_labels.values())
        aux_memory_labels = torch.cat(aux_memory_labels, dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        return None
    
    def sleep_phase(self, num_episodes_per_sleep, optimizer, criterions, scheduler, device, classes_list=None, writer=None, task_id=None):
        '''
        Train model on episodic memory
        '''
        self.model.train()

        criterion_twist = criterions[0]
        criterion_koleoproj = criterions[1]
        criterion_crossentropy = criterions[2]
        criterion_consistcarl = criterions[3]
        criterion_bce = criterions[4]

        # num_pseudo_labels is the number of output unist in the final linear head layer
        num_pseudoclasses = self.model.module.num_pseudoclasses

        # Sample sleep episodes idxs from episodic_memory (Uniform sampling with replacement)
        weights = torch.ones(len(self.episodic_memory)) # All with weight 1 (uniform)
        sampled_episodes_idxs = list(WeightedRandomSampler(weights, num_episodes_per_sleep, replacement=True))

        # Train model on sleep episodes a bacth at a time
        for i in range(0, num_episodes_per_sleep, self.episode_batch_size):

            batch_idxs = sampled_episodes_idxs[i:i+self.episode_batch_size]
            
            #### Data ####
            batch_episodes = self.episodic_memory[batch_idxs]
            batch_images = einops.rearrange(batch_episodes, 'b v c h w -> (b v) c h w').contiguous() # all episodes and all views in one batch of (b v)
            batch_images = batch_images.to(device)

            #### Forward pass to get logits ####
            batch_logits, batch_logits_proj = self.model(batch_images, proj_out=True)

            #### Get Pseudo-labels ####
            ## Oracle labels
            batch_episodes_labels = self.episodic_memory_labels[batch_idxs]
            batch_labels = einops.rearrange(batch_episodes_labels, 'b v -> (b v)').contiguous()
            batch_labels = batch_labels.to(device) # all episodes labels and all views in one batch of (b v)

            optimizer.zero_grad()

            #### Losses ####

            ## Consistency loss
            # consis_loss, _, _ = criterion_twist(batch_logits)
            consistcarl_loss = criterion_consistcarl(batch_logits)

            ## Regularization loss
            koleoproj_loss = criterion_koleoproj(batch_logits_proj)
            # sdloss = (batch_logits**2).mean()

            ## Pseudo-labeling loss
            crossentropy_loss = criterion_crossentropy(batch_logits, batch_labels)
            # if classes_list is not None:
            #     bce_loss = criterion_bce(batch_logits, encode_label(batch_labels, classes_list, num_pseudoclasses))
            # else:
            #     raise ValueError('Classes list is None')
            
            #### Total Loss ####
            # loss = crossentropy_loss
            # loss = bce_loss
            # loss = consis_loss + koleoproj_loss + crossentropy_loss
            loss = consistcarl_loss + koleoproj_loss + crossentropy_loss
            # loss = consistcarl_loss + koleoproj_loss + bce_loss
            # loss = consistcarl_loss + crossentropy_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            if i==0 or (i//self.episode_batch_size) % 5 == 0:
                current_episode_idx = min(i+self.episode_batch_size,num_episodes_per_sleep)
                print(f'Episode [{current_episode_idx}/{num_episodes_per_sleep}] -- lr: {scheduler.get_last_lr()[0]:.6f}' +
                      
                    #   f' -- Consis: {consis_loss.item():.6f}' +
                        f' -- ConsistCARL: {consistcarl_loss.item():.6f}' +

                      f' -- koleoproj_loss: {koleoproj_loss.item():.6f}' +
                      #   f' -- SD: {sdloss.item():.6f}' +

                      f' -- CrossEntropy: {crossentropy_loss.item():.6f}' +
                    #   f' -- BCE: {bce_loss.item():.6f}' +

                      f' -- Total: {loss.item():.6f}'
                      )
                
                if writer is not None and task_id is not None:
                    # get row and column std on lin output (similar to what they do in twists to track statistical caracteristics)
                    column_std = batch_logits.detach().std(dim=0, unbiased=False).mean().item()
                    row_std = batch_logits.detach().std(dim=1, unbiased=False).mean().item()

                    # writer.add_scalar('Consistency Loss', consis_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('ConsistCARL Loss', consistcarl_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)

                    writer.add_scalar('koleoproj_loss Loss', koleoproj_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    # writer.add_scalar('SD Loss', sdloss.item(), task_id*num_episodes_per_sleep + current_episode_idx)

                    writer.add_scalar('CrossEntropy Loss', crossentropy_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    # writer.add_scalar('BCE Loss', bce_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)

                    writer.add_scalar('Total Loss', loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)

                    writer.add_scalar('Column std', column_std, task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Row std', row_std, task_id*num_episodes_per_sleep + current_episode_idx)

        return None
    
    def evaluate_model(self, val_loader, device, calc_cluster_acc=False,
                       plot_clusters=False, save_dir_clusters=None, task_id=None, mean=None, std=None, num_pseudoclasses=10):
        '''
        Evaluate model on validation set
        '''
        self.model.eval()

        all_logits = []
        all_probs = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(val_loader):
                images = images.to(device)
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_logits.append(logits.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_labels.append(targets)
        
        all_logits = torch.cat(all_logits).numpy()
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

            ### plot 25 images per cluster
            os.makedirs(save_dir_clusters, exist_ok=True)
            for i in range(num_pseudoclasses): 
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
                if i == 19: # Only plot 20 clusters (from 0 to 19) if num_pseudoclasses > 20
                    break

            ### summary of clusters using a stripplot
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

            ### Calculate intra and inter cluster distances (also for labels)
            clusters_intra = intra_cluster_distance(all_logits, all_preds) # We want this to be low
            cluster_inter = inter_cluster_distance(all_logits, all_preds) # We want this to be high
            labels_intra = intra_cluster_distance(all_logits, all_labels)
            labels_inter = inter_cluster_distance(all_logits, all_labels)

            ### Plot all logits in a 2D space. (PCA, then, t-SNE).
            tsne = TSNE(n_components=2, random_state=0)
            all_logits_2d = tsne.fit_transform(all_logits)
            # Legend is cluster ID
            plt.figure(figsize=(8, 8))
            clusters_IDs = np.unique(all_preds)
            for i in clusters_IDs:
                indices = all_preds==i
                plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Cluster {i}', alpha=0.75, s=20, color=sns.color_palette("husl", num_pseudoclasses)[i]) ####################
            plt.title(f'{name}\nLogits 2D space TaskID: {task_id}\nIntra: {clusters_intra:.4f}, Inter: {cluster_inter:.4f}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(save_dir_clusters, f'logits_2d_space_clusters_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            labels_IDs = np.unique(all_labels)
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20)
            plt.title(f'{name}\nLogits 2D space TaskID: {task_id}\nIntra: {labels_intra:.4f}, Inter: {labels_inter:.4f}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(save_dir_clusters, f'logits_2d_space_labels_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

            ### Plot headmap showing cosine similarity matrix of each cluster weights
            clusters_weights = self.model.module.linear_head.weight.data.cpu()
            clusters_weights = F.normalize(clusters_weights, p=2, dim=1)
            clusters_cosine_sim = torch.mm(clusters_weights, clusters_weights.T)
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(clusters_cosine_sim, cmap='viridis', ax=ax, annot= num_pseudoclasses==10 , vmax=1, vmin=-1)
            plt.title(f'{name}\nCosine Similarity Matrix TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Cluster ID')
            plt.savefig(os.path.join(save_dir_clusters, f'cosine_similarity_matrix_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

            ### Plot number of samples per cluster with color per class
            dict_class_vs_clusters = {}
            for i in labels_IDs:
                dict_class_vs_clusters[f'Class {i}'] = []
                for j in range(num_pseudoclasses):
                    indices = (all_labels==i) & (all_preds==j)
                    dict_class_vs_clusters[f'Class {i}'].append(np.sum(indices))
            df = pd.DataFrame(dict_class_vs_clusters)
            df.plot.bar(stacked=True, figsize=(10, 8))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(f'{name}\nNumber of samples per cluster per class TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of samples')
            plt.xticks(rotation=0)
            plt.savefig(os.path.join(save_dir_clusters, f'number_samples_per_cluster_per_class_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

            ### Make plot with subplots where each subplot has the mean probability vector of each cluster
            if num_pseudoclasses<=10:
                plt.figure(figsize=(16, 8))
            else:
                plt.figure(figsize=(16*1.5, 8*1.5))
            rows = num_pseudoclasses//5
            cols = 5
            for i in range(num_pseudoclasses):
                indices = all_preds==i
                if np.sum(indices) > 0:
                    mean_probs = np.mean(all_probs[indices], axis=0)
                if np.sum(indices) == 0:
                    mean_probs = np.zeros(all_probs.shape[1])
                plt.subplot(rows, cols, i+1)
                plt.bar(np.arange(len(mean_probs)), mean_probs, label=f'Cluster {i}')
                plt.legend(loc=0)
                plt.xlabel('Cluster ID')
                if i % cols == 0:
                    plt.ylabel('Probability')
                plt.ylim(0, 1)
                plt.xticks(ticks=np.arange(len(mean_probs)), labels=np.arange(len(mean_probs)))
                plt.grid()
            plt.suptitle(f'Mean Probability Vector per Cluster TaskID: {task_id}')
            plt.savefig(os.path.join(save_dir_clusters, f'mean_prob_vector_clusters_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

            ### Make plot with subplots where each subplot has the mean probability vector of each class
            if num_pseudoclasses<=10:
                plt.figure(figsize=(16, 8))
            else:
                plt.figure(figsize=(16*1.5, 8*1.5))
            total_labels_IDs = np.max(labels_IDs)+1
            rows = 2
            cols = int(total_labels_IDs/2)
            for i in range(len(labels_IDs)):
                indices = all_labels==i
                if np.sum(indices) > 0:
                    mean_probs = np.mean(all_probs[indices], axis=0)
                if np.sum(indices) == 0:
                    mean_probs = np.zeros(all_probs.shape[1])
                plt.subplot(rows, cols, i+1)
                plt.bar(np.arange(len(mean_probs)), mean_probs, label=f'Class {i}')
                plt.legend(loc=0)
                plt.xlabel('Cluster ID')
                if i % cols == 0:
                    plt.ylabel('Probability')
                plt.ylim(0, 1)
                plt.xticks(ticks=np.arange(len(mean_probs)), labels=np.arange(len(mean_probs)))
                plt.grid()
            plt.suptitle(f'Mean Probability Vector per Class TaskID: {task_id}')
            plt.savefig(os.path.join(save_dir_clusters, f'mean_prob_vector_classes_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

        return None