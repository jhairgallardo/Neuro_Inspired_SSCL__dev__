import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast

from evaluate_cluster import evaluate as eval_pred
from utils import statistics

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
from tqdm import tqdm
import colorcet as cc

class Wake_Sleep_trainer:
    def __init__(self, 
                 view_encoder, 
                 semantic_memory, 
                 episode_batch_size,
                 num_episodes_per_sleep,
                 num_views,
                 dataset_mean,
                 dataset_std,
                 device,
                 save_dir,
                 ):
        self.view_encoder = view_encoder
        self.semantic_memory = semantic_memory
        self.episode_batch_size = episode_batch_size
        self.num_episodes_per_sleep = num_episodes_per_sleep
        self.num_views = num_views

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.device = device
        self.save_dir = save_dir

        self.episodic_memory_tensors = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)

        self.sleep_episode_counter = 0
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        self.view_encoder.eval()

        aux_memory_tensors = {}
        aux_memory_labels = {}
        for i, (batch_episodes, batch_labels , _ ) in enumerate(tqdm(incoming_dataloader)):
            batch_episodes_imgs = batch_episodes
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1))

            # forwards pass to get tensors
            batch_episodes_imgs = batch_episodes_imgs.to(self.device)
            batch_episodes_tensors = torch.empty(0).to(self.device)
            with torch.no_grad():
                for v in range(self.num_views):
                    batch_imgs = batch_episodes_imgs[:,v,:,:,:]
                    batch_tensors = self.view_encoder(batch_imgs) # shape (batch_size, 512, 7, 7)
                    batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_tensors.unsqueeze(1)], dim=1)
            # collect episodes
            aux_memory_tensors[i] = batch_episodes_tensors.cpu()
            aux_memory_labels[i] = batch_episodes_labels

        # Concatenate all
        aux_memory_tensors = torch.cat(list(aux_memory_tensors.values()), dim=0)
        aux_memory_labels = torch.cat(list(aux_memory_labels.values()), dim=0)
        self.episodic_memory_tensors = torch.cat([self.episodic_memory_tensors, aux_memory_tensors], dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        # Reset sleep iteration counter
        self.sleep_episode_counter = 0

        return None
    
    def NREM_sleep(self,
                    optimizers,
                    schedulers,
                    criterions,
                    task_id=None,
                    scaler=None,
                    writer=None):
        '''
        Train conditional generator and semantic memory
        '''

        self.view_encoder.eval()
        self.semantic_memory.train()

        # freeze view encoder
        for param in self.view_encoder.parameters():
            param.requires_grad = False
        # unfreeze semantic memory
        for param in self.semantic_memory.parameters():
            param.requires_grad = True

        # optimizers, schedulers, and criterions
        optimizer_sem = optimizers[0]
        scheduler_sem = schedulers[0]
        criterion_cluster = criterions[0]
        criterion_instance = criterions[1]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_tensors)) # Uniform sampling

        # Train model a bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_tensor = self.episodic_memory_tensors[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            batch_episodes_instfeats = torch.empty(0).to(self.device)  
            batch_episodes_probs = torch.empty(0).to(self.device)
            for v in range(self.num_views):
                batch_tensors = batch_episodes_tensor[:,v,:,:,:]
                ### Forward pass Semantic Memory
                with autocast():
                    batch_instfeats, batch_probs = self.semantic_memory(batch_tensors, return_inst=True)
                batch_episodes_instfeats = torch.cat([batch_episodes_instfeats, batch_instfeats.unsqueeze(1)], dim=1)
                batch_episodes_probs = torch.cat([batch_episodes_probs, batch_probs.unsqueeze(1)], dim=1)

            #### --- Calculate Semantic Memory Loss --- ####
            loss_cluster = 0
            loss_inst = 0
            for v in range(self.num_views-1):
                loss_cluster += criterion_cluster(batch_episodes_probs[:,0], batch_episodes_probs[:,v+1])
                loss_inst += criterion_instance(batch_episodes_instfeats[:,0], batch_episodes_instfeats[:,v+1])
            loss_cluster /= self.num_views
            loss_inst /= self.num_views
            ### -> Total Loss ###
            loss = loss_cluster + loss_inst

            #### --- Backward Pass --- ####
            optimizer_sem.zero_grad()
            scaler.scale(loss).backward()
            # Sanitycheck: check that gradients for view encoder are zeros or None (since it is frozen)
            for name, param in self.view_encoder.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)
            scaler.step(optimizer_sem)
            scaler.update()
            scheduler_sem.step()

            #### --- Update sleep counter --- #####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            sem_lr = scheduler_sem.get_last_lr()[0]
            loss_sem_value = loss_cluster.item()
            loss_inst_value = loss_inst.item()
            loss_value = loss.item()
            first_view_probs = batch_episodes_probs[:,0].detach().cpu()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                _, _, mi = statistics(first_view_probs)
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- sem lr: {sem_lr:.6f}' +
                      f' -- mi: {mi.item():.6f}' +
                      f' -- Loss Semantic Memory: {loss_sem_value:.6f}' +
                      f' -- Loss InstContrastive: {loss_inst_value:.6f}' +
                      f' -- Loss Total: {loss_value:.6f}'
                      )
                writer.add_scalar('MI', mi.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                
            #### --- Plot metrics --- ####
            writer.add_scalar('sem lr', sem_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Semantic Memory', loss_sem_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss InstContrastive', loss_inst_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)

        # Clean up gradients
        optimizer_sem.zero_grad()

        return None

def evaluate_semantic_memory(loader,
                             view_encoder,
                             semantic_memory,
                             num_pseudoclasses,
                             task_id,
                             device,
                             plot_clusters,
                             num_gt_classes,
                             save_dir):
        '''
        Evaluate semantic memory
        '''

        view_encoder.eval()
        semantic_memory.eval()
        
        pool_layer = torch.nn.AdaptiveAvgPool2d((1, 1))

        all_probs = []
        all_preds = []
        all_labels = []
        all_instproj = []
        all_embenc = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(loader):
                images = images.to(device)
                embenc = view_encoder(images)
                instproj, probs = semantic_memory(embenc, return_inst=True)
                preds = torch.argmax(probs, dim=1)
                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_labels.append(targets)
                all_instproj.append(instproj.detach().cpu())
                # Pool and flat encoder embeddings
                embenc = torch.flatten(pool_layer(embenc), 1)
                all_embenc.append(embenc.detach().cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_indices = np.arange(len(all_preds))
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_instproj = torch.cat(all_instproj).numpy()
        all_embenc = torch.cat(all_embenc).numpy()

        calc_cluster_acc = len(np.unique(all_labels)) == num_pseudoclasses
        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5, purity, hmg, cm, cfi = eval_pred(all_labels.astype(int), all_preds.astype(int), num_pseudoclasses, calc_acc=calc_cluster_acc, total_probs=all_probs)

        if plot_clusters:
            save_dir_clusters = os.path.join(save_dir, 'val_seen_semantic_clusters')
            os.makedirs(save_dir_clusters, exist_ok=True)

            dict_class_vs_clusters = {}
            labels_IDs = np.unique(all_labels)
            clusters_IDs = np.unique(all_preds)
            for i in labels_IDs:
                dict_class_vs_clusters[f'Class {i}'] = []
                for j in range(num_pseudoclasses):
                    indices = (all_labels==i) & (all_preds==j)
                    dict_class_vs_clusters[f'Class {i}'].append(np.sum(indices))
            palette_labelsID= cc.glasbey_category10[:num_gt_classes]

            label_id_to_color = {label_id: palette_labelsID[idx] for idx, label_id in enumerate(range(num_gt_classes))}

            ### Plot number of samples per cluster with color per class
            df = pd.DataFrame(dict_class_vs_clusters)
            color_list = [label_id_to_color[i] for i in labels_IDs]
            df.plot.bar(stacked=True, color=color_list, figsize=(21,8))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.title(f'Number of samples per cluster per class TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of samples')
            plt.xticks(rotation=0)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_number_samples_per_cluster_per_class.png'), bbox_inches='tight')
            plt.close()
            # save df, color_list, labels_IDs
            df.to_csv(os.path.join(save_dir_clusters, f'taskid_{task_id}_number_samples_per_cluster_per_class.csv'))
            np.save(os.path.join(save_dir_clusters, f'taskid_{task_id}_color_list.npy'), color_list)
            np.save(os.path.join(save_dir_clusters, f'taskid_{task_id}_labels_IDs.npy'), labels_IDs)

            ### Plot encoder outputs
            ## Plot all logits in a 2D space. (t-SNE).
            tsne = TSNE(n_components=2, random_state=0)
            all_embenc_2d = tsne.fit_transform(all_embenc)
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_embenc_2d[indices, 0], all_embenc_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20, color=label_id_to_color[i])
            plt.title(f'Encoder space (2D TSNE) TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_encoder_tsne_2d_space_labels.png'), bbox_inches='tight')
            plt.close()
            ## Plot all logits in a 2D space (PCA)
            pca = PCA(n_components=2, random_state=0)
            all_embenc_pca_2d = pca.fit_transform(all_embenc)
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_embenc_pca_2d[indices, 0], all_embenc_pca_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20, color=label_id_to_color[i])
            plt.title(f'Encoder space (2D PCA) TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_encoder_pca_2d_space_labels.png'), bbox_inches='tight')
            plt.close()

            ### Plot projector outputs
            ## Plot all logits in a 2D space. (t-SNE).
            tsne = TSNE(n_components=2, random_state=0)
            all_instproj_2d = tsne.fit_transform(all_instproj)
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_instproj_2d[indices, 0], all_instproj_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20, color=label_id_to_color[i])
            plt.title(f'\nContrastive Projector space (2D TSNE) TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_instproj_tsne_2d_space_labels.png'), bbox_inches='tight')
            plt.close()
            ## Plot all logits in a 2D space (PCA)
            pca = PCA(n_components=2, random_state=0)
            all_instproj_pca_2d = pca.fit_transform(all_instproj)
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_instproj_pca_2d[indices, 0], all_instproj_pca_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20, color=label_id_to_color[i])
            plt.title(f'Contrastive Projector space (2D PCA) TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_instproj_pca_2d_space_labels.png'), bbox_inches='tight')
            plt.close()

        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5,
                        'Purity': purity, 'Homogeneity': hmg, 'Completeness': cm, 'Class-Fragmentation': cfi}

        return task_metrics