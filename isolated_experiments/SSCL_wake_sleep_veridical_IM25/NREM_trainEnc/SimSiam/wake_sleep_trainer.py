import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast

from tqdm import tqdm

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
                 episode_batch_size,
                 num_episodes_per_sleep,
                 num_views,
                 dataset_mean,
                 dataset_std,
                 device,
                 save_dir,
                 ):
        self.episode_batch_size = episode_batch_size
        self.num_episodes_per_sleep = num_episodes_per_sleep
        self.num_views = num_views

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
            view_encoder,
            projector_rep,
            predictor_rep, 
            optimizers,
            schedulers,
            criterions,
            task_id=None,
            scaler=None,
            writer=None):
        '''
        Train conditional generator and semantic memory
        '''

        view_encoder.train()
        projector_rep.train()
        predictor_rep.train()

        # optimizers, schedulers, and criterions
        optimizer_rep = optimizers[0]
        scheduler_rep = schedulers[0]
        criterion_simsiam = criterions[0]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_imgs)) # Uniform sampling

        # Train model a bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_imgs = self.episodic_memory_imgs[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            predictions = torch.empty(0).to(self.device)
            targets = torch.empty(0).to(self.device)        
            for v in range(self.num_views):
                batch_imgs = batch_episodes_imgs[:,v,:,:,:]
                ### Forward pass
                with autocast():
                    batch_tensors = view_encoder(batch_imgs)
                    batch_targets = projector_rep(batch_tensors)
                    batch_predictions = predictor_rep(batch_targets)
                predictions = torch.cat([predictions, batch_predictions.unsqueeze(1)], dim=1)
                targets = torch.cat([targets, batch_targets.unsqueeze(1)], dim=1)

            #### --- Calculate Representation learning Loss --- ####
            loss_rep = criterion_simsiam(predictions, targets)
            ### Total loss
            loss = loss_rep

            #### --- Backward Pass --- ####
            optimizer_rep.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer_rep)
            scaler.update()

            scheduler_rep.step()

            #### --- Update sleep counter --- #####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            rep_lr = scheduler_rep.get_last_lr()[0]
            # pred_lr = optimizer_pred.param_groups[0]['lr']
            loss_rep_value = loss_rep.item()
            loss_value = loss.item()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- lr rep: {rep_lr:.6f}' +
                    #   f' -- lr pred: {pred_lr:.6}' +
                      f' -- Loss Representation: {loss_rep_value:.6f}' +
                      f' -- Loss Total: {loss_value:.6f}'
                    )
                
            #### --- Plot metrics --- ####
            writer.add_scalar('lr rep', rep_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Representation', loss_rep_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)

        # Clean up gradients
        optimizer_rep.zero_grad()
        # optimizer_pred.zero_grad()

        return None