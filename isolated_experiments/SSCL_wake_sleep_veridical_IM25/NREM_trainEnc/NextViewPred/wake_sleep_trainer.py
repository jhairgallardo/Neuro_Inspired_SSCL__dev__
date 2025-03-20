import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast

from tqdm import tqdm

class Wake_Sleep_trainer:
    def __init__(self, 
                 episode_batch_size,
                 num_episodes_per_sleep,
                 num_views,
                 dataset_mean,
                 dataset_std,
                 device,
                 save_dir,
                 koleo_gamma=0
                 ):
        self.episode_batch_size = episode_batch_size
        self.num_episodes_per_sleep = num_episodes_per_sleep
        self.num_views = num_views

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.device = device
        self.save_dir = save_dir

        self.episodic_memory_imgs = torch.empty(0)
        self.episodic_memory_actions = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)

        self.koleo_gamma = koleo_gamma

        self.sleep_episode_counter = 0
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''

        aux_memory_imgs = {}
        aux_memory_actions = {}
        aux_memory_labels = {}
        for i, (batch_episodes, batch_labels , _ ) in enumerate(tqdm(incoming_dataloader)):
            batch_episodes_imgs = batch_episodes[0]
            batch_episodes_actions = batch_episodes[1]
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1))
            # collect episodes
            aux_memory_imgs[i] = batch_episodes_imgs
            aux_memory_actions[i] = batch_episodes_actions
            aux_memory_labels[i] = batch_episodes_labels

        # Concatenate all
        aux_memory_imgs = torch.cat(list(aux_memory_imgs.values()), dim=0)
        aux_memory_actions = torch.cat(list(aux_memory_actions.values()), dim=0)
        aux_memory_labels = torch.cat(list(aux_memory_labels.values()), dim=0)
        self.episodic_memory_imgs = torch.cat([self.episodic_memory_imgs, aux_memory_imgs], dim=0)
        self.episodic_memory_actions = torch.cat([self.episodic_memory_actions, aux_memory_actions], dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        # Reset sleep iteration counter
        self.sleep_episode_counter = 0

        return None
    
    def sleep(self,
            view_encoder,
            predictor_net,
            optimizers,
            schedulers,
            criterions,
            task_id=None,
            scaler=None,
            writer=None,
            corruption_prob=0.75):
        '''
        Train conditional generator and semantic memory
        '''

        view_encoder.train()
        predictor_net.train()

        # optimizers, schedulers, and criterions
        optimizer = optimizers[0]
        scheduler = schedulers[0]
        criterion_mse = criterions[0]
        criterion_koleo = criterions[1]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_imgs)) # Uniform sampling

        # Train model a bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_imgs = self.episodic_memory_imgs[batch_episodes_idxs].to(self.device)
            batch_episodes_actions = self.episodic_memory_actions[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            # batch_episodes_tensors = torch.empty(0).to(self.device)
            # batch_episodes_outputs = torch.empty(0).to(self.device)
            # batch_first_view_imgs = batch_episodes_imgs[:,0]       
            # for v in range(self.num_views):
            #     batch_imgs = batch_episodes_imgs[:,v]
            #     batch_actions = batch_episodes_actions[:,v]
            #     ### Forward pass
            #     # with autocast():
            #     batch_tensors = view_encoder(batch_imgs)
            #     batch_first_view_tensors = view_encoder(batch_first_view_imgs)
            #     batch_outputs = predictor_net(batch_first_view_tensors, batch_actions)
            #     batch_episodes_outputs = torch.cat([batch_episodes_outputs, batch_outputs.unsqueeze(1)], dim=1)
            #     batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_tensors.unsqueeze(1)], dim=1)

            batch_episodes_tensors = torch.empty(0).to(self.device)
            batch_episodes_outputs = torch.empty(0).to(self.device)
            batch_episodes_outputs_noaction = torch.empty(0).to(self.device)
            batch_first_view_imgs = batch_episodes_imgs[:,0]
            batch_noaction = batch_episodes_actions[:,0] # no action is in the first view
            for v in range(1, self.num_views):
                batch_imgs = batch_episodes_imgs[:,v]
                batch_actions = batch_episodes_actions[:,v]
                ### Forward pass
                # with autocast():

                # View encoder forward pass
                batch_tensors = view_encoder(batch_imgs)
                batch_first_view_tensors = view_encoder(batch_first_view_imgs)

                # Predictor forward pass (with and without action)
                batch_outputs = predictor_net(batch_first_view_tensors, batch_actions)
                batch_tensors_corrupted = self.corrupt_batch_tensors(batch_tensors, p=corruption_prob)
                batch_outputs_noaction = predictor_net(batch_tensors_corrupted, batch_noaction) # corrupt batch_tensors for no action case

                # collect outputs
                batch_episodes_outputs = torch.cat([batch_episodes_outputs, batch_outputs.unsqueeze(1)], dim=1)
                batch_episodes_outputs_noaction = torch.cat([batch_episodes_outputs_noaction, batch_outputs_noaction.unsqueeze(1)], dim=1)
                batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_tensors.unsqueeze(1)], dim=1)
                           

            #### --- Calculate Representation learning Loss --- ####
            # get koleo
            if self.koleo_gamma != 0: 
                # Should I apply it to the first views only? (same branch as outputs) 
                # or to the targets? barch that has stop gradient on the MSE loss
                loss_koleo = criterion_koleo(batch_episodes_tensors.mean(dim=(3,4))) # pass the average pooled version (koleo works on vectors) 
            else:
                loss_koleo = torch.tensor(0).to(self.device)
            # get representation loss
            loss_rep = criterion_mse(batch_episodes_outputs, batch_episodes_tensors.detach())
            # get representation loss for no action
            loss_rep_noaction = criterion_mse(batch_episodes_outputs_noaction, batch_episodes_tensors.detach())
            ### Total loss
            # loss = loss_rep + self.koleo_gamma*loss_koleo
            loss = loss_rep + loss_rep_noaction + self.koleo_gamma*loss_koleo

            #### --- Backward Pass --- ####
            optimizer.zero_grad()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()

            scheduler.step()

            #### --- Update sleep counter --- #####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            lr = scheduler.get_last_lr()[0]
            loss_rep_value = loss_rep.item()
            loss_rep_noaction_value = loss_rep_noaction.item()
            loss_koleo_value = loss_koleo.item()
            loss_value = loss.item()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- lr: {lr:.6f}' +
                      f'{f" -- Loss Koleo: {loss_koleo_value:.6f}" if self.koleo_gamma > 0 else ""}' +
                      f' -- Loss Representation: {loss_rep_value:.6f}' +
                      f' -- Loss Representation NoAction: {loss_rep_noaction_value:.6f}' +
                      f' -- Loss Total: {loss_value:.6f}'
                    )
                
            #### --- Plot metrics --- ####
            writer.add_scalar('lr rep', lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if self.koleo_gamma > 0:
                writer.add_scalar('Loss Koleo', loss_koleo_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Representation', loss_rep_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)

        # Clean up gradients
        optimizer.zero_grad()

        return None
    
    def corrupt_batch_tensors(self, tensors, p=0.75):
        """
        Corrupt feature maps by randomly zeroing out complete channel vectors at spatial locations.

        Args:
            tensors (torch.Tensor): Input tensor of shape (B, C, H, W).
            p (float): Probability of zeroing out a spatial location (default is 0.75).

        Returns:
            torch.Tensor: Corrupted tensor.
        """
        # Generate a mask with shape (B, H, W): 1 means keep, 0 means zero out.
        # With probability 'p' we want to zero out, so we keep if random value >= p.
        mask = (torch.rand(tensors.shape[0], tensors.shape[2], tensors.shape[3], device=tensors.device) >= p).float()
        # Add a channel dimension to match tensors: shape becomes (B, 1, H, W).
        mask = mask.unsqueeze(1)
        # Multiply the input tensor by the mask. Broadcasting will zero out the entire channel vector where needed.
        return tensors * mask