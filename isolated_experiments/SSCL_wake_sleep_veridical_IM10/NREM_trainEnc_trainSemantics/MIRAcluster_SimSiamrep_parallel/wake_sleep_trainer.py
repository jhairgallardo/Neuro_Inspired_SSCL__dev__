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

import pandas as pd
from tqdm import tqdm
import colorcet as cc

@torch.no_grad()
def mira(k: torch.Tensor,
         tau: float,
         beta: float,
         iters: int):
    bs = k.size(0) #* dist.get_world_size()  # total batch-size

    # fixed point iteration
    k = F.softmax(k / tau / (1 - beta), dim=1)
    temp = k.sum(dim=0)
    # dist.all_reduce(temp)
    v = (temp / bs).pow(1 - beta)
    for _ in range(iters):
        temp = k / (v.pow(- beta / (1 - beta)) * k).sum(dim=1, keepdim=True)
        temp = temp.sum(dim=0)
        # dist.all_reduce(temp)
        v = (temp / bs).pow(1 - beta)
    temp = v.pow(- beta / (1 - beta)) * k
    target = temp / temp.sum(dim=1, keepdim=True)
    # if there is nan in the target, return k
    if torch.isnan(target).any():
        # error
        raise ValueError('Nan in target')
    return target

@torch.no_grad()
def mira_pseudolabeling(logits, num_views, tau, beta, iters):
    targets = torch.empty(0).to(logits.device)
    for t in range(num_views):
        targets_t = mira(logits[:,t], tau, beta, iters)
        targets = torch.cat([targets, targets_t.unsqueeze(1)], dim=1)
    return targets

class Wake_Sleep_trainer:
    def __init__(self, 
                 view_encoder,
                 projector_rep,
                 predictor_rep,
                 semantic_memory, 
                 episode_batch_size,
                 num_episodes_per_sleep,
                 num_views,
                 tau_t,
                 tau_s,
                 beta,
                 dataset_mean,
                 dataset_std,
                 device,
                 save_dir,
                 ):
        self.view_encoder = view_encoder
        self.projector_rep = projector_rep
        self.predictor_rep = predictor_rep
        self.semantic_memory = semantic_memory
        self.episode_batch_size = episode_batch_size
        self.num_episodes_per_sleep = num_episodes_per_sleep
        self.num_views = num_views

        self.tau_t = tau_t
        self.tau_s = tau_s
        self.beta = beta

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.device = device
        self.save_dir = save_dir

        self.episodic_memory_imgs = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)

        self.sleep_episode_counter = 0
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''

        aux_memory_imgs = {}
        aux_memory_labels = {}
        for i, (batch_episodes, batch_labels , _ ) in enumerate(tqdm(incoming_dataloader)):
            batch_episodes_imgs = batch_episodes
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1))
            # collect episodes
            aux_memory_imgs[i] = batch_episodes_imgs
            aux_memory_labels[i] = batch_episodes_labels

        # Concatenate all
        aux_memory_imgs = torch.cat(list(aux_memory_imgs.values()), dim=0)
        aux_memory_labels = torch.cat(list(aux_memory_labels.values()), dim=0)
        self.episodic_memory_imgs = torch.cat([self.episodic_memory_imgs, aux_memory_imgs], dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        # Reset sleep iteration counter
        self.sleep_episode_counter = 0

        return None
    
    def sleep(self,
            optimizers,
            schedulers,
            criterions,
            task_id=None,
            scaler=None,
            writer=None):

        self.view_encoder.train()
        self.projector_rep.train()
        self.predictor_rep.train()
        self.semantic_memory.train()

        # optimizers, schedulers, and criterions
        optimizer_rep = optimizers[0]
        scheduler_rep = schedulers[0]
        criterion_rep = criterions[0]
        optimizer_sem = optimizers[1]
        scheduler_sem = schedulers[1]
        criterion_sem = criterions[1]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_imgs)) # Uniform sampling

        # Train model a bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_imgs = self.episodic_memory_imgs[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            batch_episodes_logits = torch.empty(0).to(self.device)
            batch_episodes_predictions = torch.empty(0).to(self.device)
            batch_episodes_targets = torch.empty(0).to(self.device)          
            for v in range(self.num_views):
                batch_imgs = batch_episodes_imgs[:,v,:,:,:]
                ### Forward pass Semantic Memory
                with autocast():
                    batch_tensors = self.view_encoder(batch_imgs)
                    batch_targets = self.projector_rep(batch_tensors)
                    batch_predictions = self.predictor_rep(batch_targets)
                    batch_logits = self.semantic_memory(batch_tensors)
                batch_episodes_predictions = torch.cat([batch_episodes_predictions, batch_predictions.unsqueeze(1)], dim=1)
                batch_episodes_targets = torch.cat([batch_episodes_targets, batch_targets.unsqueeze(1)], dim=1)
                batch_episodes_logits = torch.cat([batch_episodes_logits, batch_logits.unsqueeze(1)], dim=1)

            #### --- Calculate Semantic Memory Loss --- ####
            batch_pseudolabels = mira_pseudolabeling(logits = batch_episodes_logits, num_views = self.num_views, tau=self.tau_t, beta=self.beta, iters=30)
            loss_sem = criterion_sem(batch_episodes_logits/self.tau_s, batch_pseudolabels)
            loss_rep = criterion_rep(batch_episodes_predictions, batch_episodes_targets)
            ### -> Total Loss ###
            loss = loss_rep + loss_sem

            #### --- Backward Pass --- ####
            optimizer_rep.zero_grad()
            optimizer_sem.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_rep)
            scaler.step(optimizer_sem)
            scaler.update()
            scheduler_rep.step()
            scheduler_sem.step()

            #### --- Update sleep counter --- #####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            rep_lr = scheduler_rep.get_last_lr()[0]
            sem_lr = scheduler_sem.get_last_lr()[0]
            loss_sem_value = loss_sem.item()
            loss_rep_value = loss_rep.item()
            loss_value = loss.item()
            first_view_logits = batch_episodes_logits[:,0].detach().cpu()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                ps = F.softmax(first_view_logits / self.tau_s, dim=1)
                pt = F.softmax(first_view_logits / self.tau_t, dim=1)
                _, _, mi_ps = statistics(ps)
                _, _, mi_pt = statistics(pt)
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- rep lr: {rep_lr:.6f}' +
                      f' -- sem lr: {sem_lr:.6f}' +
                      f' -- mi_ps: {mi_ps.item():.6f} -- mi_pt: {mi_pt.item():.6f}' +
                      f' -- Loss Representation: {loss_rep_value:.6f}' +
                      f' -- Loss Semantic Memory: {loss_sem_value:.6f}' +
                      f' -- Loss Total: {loss_value:.6f}'
                      )
                writer.add_scalar('MI_ps', mi_ps.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                writer.add_scalar('MI_pt', mi_pt.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                
            #### --- Plot metrics --- ####
            writer.add_scalar('rep lr', rep_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('sem lr', sem_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Representation', loss_rep_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Semantic Memory', loss_sem_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)

        # Clean up gradients
        optimizer_sem.zero_grad()

        return None

def evaluate_semantic_memory(loader,
                             view_encoder,
                             projector_rep,
                             predictor_rep,
                             semantic_memory,
                             tau_s,
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
        projector_rep.eval()
        predictor_rep.eval()
        semantic_memory.eval()
        
        all_logits = []
        all_probs = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(loader):
                images = images.to(device)
                enc_representation = view_encoder(images)
                logits = semantic_memory(enc_representation)
                probs = F.softmax(logits/tau_s, dim=1)
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

        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5,
                        'Purity': purity, 'Homogeneity': hmg, 'Completeness': cm, 'Class-Fragmentation': cfi}

        return task_metrics