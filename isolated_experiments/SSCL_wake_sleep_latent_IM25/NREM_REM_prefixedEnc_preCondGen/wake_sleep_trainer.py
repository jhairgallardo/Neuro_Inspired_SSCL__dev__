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

import seaborn as sns
import pandas as pd
from tqdm import tqdm
import einops
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

class SlopePlateauDetector:
    def __init__(self, window_size=50, patience=40, slope_threshold=1e-3, use_ema=False, ema_alpha=0.1):
        self.window_size = window_size
        self.patience = patience
        self.slope_threshold = slope_threshold
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.loss_window_history = []  # fixed-size list for storing recent loss (or smoothed loss) values
        self.consecutive_flat_windows = 0
        self.ema_loss = None  # stores the current EMA of the loss

    def step(self, loss):
        # Update the EMA if smoothing is enabled; otherwise, use the raw loss.
        if self.use_ema:
            if self.ema_loss is None:
                self.ema_loss = loss
            else:
                self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
            current_value = self.ema_loss
        else:
            current_value = loss
        # Maintain a fixed-size window of loss values.
        self.loss_window_history.append(current_value)
        if len(self.loss_window_history) > self.window_size:
            self.loss_window_history.pop(0)
        # Only proceed if the window is full.
        if len(self.loss_window_history) < self.window_size:
            return False, None, current_value
        # Compute the slope using the closed-form solution of linear regression.
        # Here, x is the array of indices (0 to window_size-1) and y is the loss window.
        x = np.arange(self.window_size)
        y = np.array(self.loss_window_history)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2) + 1e-8
        slope = numerator / denominator
        # Check if the slope is nearly zero.
        if abs(slope) < self.slope_threshold:
            self.consecutive_flat_windows += 1
        else:
            self.consecutive_flat_windows = 0
        # Return True if the loss has plateaued for 'patience' consecutive windows.
        return self.consecutive_flat_windows >= self.patience, abs(slope), current_value

class Wake_Sleep_trainer:
    def __init__(self, 
                 view_encoder, 
                 conditional_generator, 
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
        self.conditional_generator = conditional_generator
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

        self.episodic_memory_tensors = torch.empty(0)
        self.episodic_memory_actions = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)

        self.sleep_episode_counter = 0
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        self.view_encoder.eval()

        aux_memory_tensors = {}
        aux_memory_actions = {}
        aux_memory_labels = {}
        for i, (batch_episodes, batch_labels , _ ) in enumerate(tqdm(incoming_dataloader)):
            batch_episodes_imgs = batch_episodes[0]
            batch_episodes_actions = batch_episodes[1]
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
            aux_memory_actions[i] = batch_episodes_actions
            aux_memory_labels[i] = batch_episodes_labels

        # Concatenate all
        aux_memory_tensors = torch.cat(list(aux_memory_tensors.values()), dim=0)
        aux_memory_actions = torch.cat(list(aux_memory_actions.values()), dim=0)
        aux_memory_labels = torch.cat(list(aux_memory_labels.values()), dim=0)
        self.episodic_memory_tensors = torch.cat([self.episodic_memory_tensors, aux_memory_tensors], dim=0)
        self.episodic_memory_actions = torch.cat([self.episodic_memory_actions, aux_memory_actions], dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        # Reset sleep iteration counter
        self.sleep_episode_counter = 0

        return None
    
    def NREM_sleep(self,
                    optimizers,
                    schedulers,
                    criterions,
                    task_id=None,
                    patience=40,
                    threshold=1e-3,
                    window=50,
                    ema_alpha=0.3,
                    scaler=None,
                    writer=None,
                    nrem_indicator=0):
        '''
        Train conditional generator and semantic memory
        '''

        self.view_encoder.eval()
        self.conditional_generator.train()
        self.semantic_memory.train()

        # freeze view encoder
        for param in self.view_encoder.parameters():
            param.requires_grad = False
        # unfreeze conditional generator
        for param in self.conditional_generator.parameters():
            param.requires_grad = True
        # unfreeze semantic memory
        for param in self.semantic_memory.parameters():
            param.requires_grad = True

        # optimizers, schedulers, and criterions
        optimizer_sem = optimizers[0]
        scheduler_sem = schedulers[0]
        optimizer_condgen = optimizers[1]
        scheduler_condgen = schedulers[1]
        criterion_crossentropyswap = criterions[0]
        criterion_mse = criterions[1]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_tensors)) # Uniform sampling

        # Loss Plateau Detector
        loss_plateau_detector = SlopePlateauDetector(window_size=window, patience=patience, slope_threshold=threshold, use_ema=True, ema_alpha=ema_alpha)

        # Train model a bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_tensor = self.episodic_memory_tensors[batch_episodes_idxs].to(self.device)
            batch_episodes_actions = self.episodic_memory_actions[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            batch_episodes_gen_FTtensor = torch.empty(0).to(self.device)
            batch_episodes_gen_DecEnctensors = torch.empty(0).to(self.device)
            batch_episodes_gen_DecEnctensors_direct = torch.empty(0).to(self.device)
            batch_episodes_logits = torch.empty(0).to(self.device)
            episode_gen_imgs = torch.empty(0).to(self.device)
            first_view_tensor_batch = batch_episodes_tensor[:,0,:,:,:]            
            for v in range(self.num_views):
                batch_tensors = batch_episodes_tensor[:,v,:,:,:]
                batch_actions = batch_episodes_actions[:,v]
                ### Forward pass Conditional Generator (from first view to augmented view. Using corresponding action)
                with autocast():
                    batch_gen_imgs, batch_gen_FTtensor = self.conditional_generator(first_view_tensor_batch, batch_actions, return_transformed_feature_map=True)
                    batch_gen_DecEnctensors = self.view_encoder(batch_gen_imgs) # shape (batch_size, 512, 7, 7)
                batch_episodes_gen_FTtensor = torch.cat([batch_episodes_gen_FTtensor, batch_gen_FTtensor.unsqueeze(1)], dim=1)
                batch_episodes_gen_DecEnctensors = torch.cat([batch_episodes_gen_DecEnctensors, batch_gen_DecEnctensors.unsqueeze(1)], dim=1)
                ### Forward pass Conditional Generator direct (from augmented view to augmented view. Unconditional mode. Direct reconstruction without FT)
                with autocast():
                    batch_gen_imgs_direct = self.conditional_generator(batch_tensors, None, unconditional_mode=True)
                    batch_gen_DecEnctensors_direct = self.view_encoder(batch_gen_imgs_direct) # shape (batch_size, 512, 7, 7)
                batch_episodes_gen_DecEnctensors_direct = torch.cat([batch_episodes_gen_DecEnctensors_direct, batch_gen_DecEnctensors_direct.unsqueeze(1)], dim=1)
                ### Forward pass Semantic Memory
                with autocast():
                    batch_logits = self.semantic_memory(batch_tensors)
                batch_episodes_logits = torch.cat([batch_episodes_logits, batch_logits.unsqueeze(1)], dim=1)
                # collect generated images for plot purposes (only first episode)
                episode_gen_imgs = torch.cat([episode_gen_imgs, batch_gen_imgs[0].unsqueeze(0)], dim=0)

            #### --- Calculate losses --- ####
            ### -> Conditional Generator Loss ###
            ## Reconstruction loss from feature transformation tensor and saved tensor
            lossgen_1 = criterion_mse(batch_episodes_gen_FTtensor, batch_episodes_tensor)
            ## Reconstruction loss from generated tensor and saved tensor
            lossgen_2 = criterion_mse(batch_episodes_gen_DecEnctensors, batch_episodes_tensor)
            ## Reconstruction loss (direct use of tensor to create gentensor. No FT. Unconditional)
            lossgen_3 = criterion_mse(batch_episodes_gen_DecEnctensors_direct, batch_episodes_tensor)
            ## Generator loss
            loss_gen = lossgen_1 + lossgen_2 + lossgen_3
            ### -> Semantic Memory Loss ###
            batch_pseudolabels = mira_pseudolabeling(logits = batch_episodes_logits, num_views = self.num_views, tau=self.tau_t, beta=self.beta, iters=30)
            loss_sem = criterion_crossentropyswap(batch_episodes_logits/self.tau_s, batch_pseudolabels)
            ### -> Total Loss ###
            loss = loss_sem + loss_gen

            #### --- Backward Pass --- ####
            optimizer_sem.zero_grad()
            optimizer_condgen.zero_grad()
            scaler.scale(loss).backward()
            # Sanitycheck: check that gradients for view encoder are zeros or None (since it is frozen)
            for name, param in self.view_encoder.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)
            scaler.step(optimizer_sem)
            scaler.step(optimizer_condgen)
            scaler.update()
            scheduler_sem.step()
            scheduler_condgen.step()

            #### --- Update sleep counter --- #####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            condgen_lr = scheduler_condgen.get_last_lr()[0]
            sem_lr = scheduler_sem.get_last_lr()[0]
            loss_sem_value = loss_sem.item()
            loss_gen_value = loss_gen.item()
            lossgen_1_value = lossgen_1.item()
            lossgen_2_value = lossgen_2.item()
            lossgen_3_value = lossgen_3.item()
            loss_value = loss.item()
            first_view_logits = batch_episodes_logits[:,0].detach().cpu()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                ps = F.softmax(first_view_logits / self.tau_s, dim=1)
                pt = F.softmax(first_view_logits / self.tau_t, dim=1)
                _, _, mi_ps = statistics(ps)
                _, _, mi_pt = statistics(pt)
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- NREM' +
                      f' -- condgen lr: {condgen_lr:.6f}' +
                      f' -- sem lr: {sem_lr:.6f}' +
                      f' -- mi_ps: {mi_ps.item():.6f} -- mi_pt: {mi_pt.item():.6f}' +
                      f' -- Loss Semantic Memory: {loss_sem_value:.6f}' +
                      f' -- Loss Conditional Generator: {loss_gen_value:.6f}' +
                      f' -- Loss Gen_1 FTtensor: {lossgen_1_value:.6f}' +
                      f' -- Loss Gen_2 DecEnctensor: {lossgen_2_value:.6f}' +
                      f' -- Loss Gen_3 DecEnctensor direct: {lossgen_3_value:.6f}' +
                      f' -- Loss Total: {loss_value:.6f}'
                      )
                writer.add_scalar('MI_ps', mi_ps.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                writer.add_scalar('MI_pt', mi_pt.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                
            #### --- Plot metrics --- ####
            writer.add_scalar('NREM-REM Indicator', nrem_indicator, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('condgen lr', condgen_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('sem lr', sem_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Semantic Memory', loss_sem_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Conditional Generator', loss_gen_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Gen_1 FTtensor', lossgen_1_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Gen_2 DecEnctensor', lossgen_2_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Gen_3 DecEnctensor direct', lossgen_3_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)

            #### --- Plot reconstructions --- ####
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (40*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                save_dir_recon=os.path.join(self.save_dir, 'reconstructions_during_training')
                os.makedirs(save_dir_recon, exist_ok=True)
                episode_gen_imgs = episode_gen_imgs.detach().cpu()
                episode_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(self.dataset_mean, self.dataset_std)], [1/s for s in self.dataset_std]) for img in episode_gen_imgs]
                episode_gen_imgs = torch.stack(episode_gen_imgs, dim=0)
                grid = torchvision.utils.make_grid(episode_gen_imgs, nrow=self.num_views)
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
                image_name = f'taskid_{task_id}_img_{self.sleep_episode_counter}_NREM_reconstructed_images.png'
                grid.save(os.path.join(save_dir_recon, image_name))

            #### --- Check for plateau --- ####
            plateau_flag, abs_slope, smooth_loss = loss_plateau_detector.step(loss_value)
            writer.add_scalar(f'Total_Loss_Smooth_alpha_{ema_alpha}', smooth_loss, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if abs_slope is not None:
                writer.add_scalar('Abs_Slope', abs_slope, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if plateau_flag:
                print(f'**Loss Plateau detected. Switching to REM phase.')
                break

        # Clean up gradients
        optimizer_sem.zero_grad()
        optimizer_condgen.zero_grad()

        return None
    
    def REM_sleep(self,
                    optimizers, 
                    schedulers,
                    criterions,
                    task_id=None,
                    patience=40,
                    threshold=1e-3,
                    window=50,
                    ema_alpha=0.3,
                    scaler=None,
                    writer=None, 
                    rem_indicator=1):
        '''
        Train conditional generator and semantic memory
        '''

        self.view_encoder.eval()
        self.conditional_generator.eval()
        self.semantic_memory.train()

        # freeze view encoder
        for param in self.view_encoder.parameters():
            param.requires_grad = False
        # unfreeze conditional generator
        for param in self.conditional_generator.parameters():
            param.requires_grad = False
        # unfreeze semantic memory
        for param in self.semantic_memory.parameters():
            param.requires_grad = True

        # optimizers, schedulers, and criterions
        optimizer_sem = optimizers[0]
        scheduler_sem = schedulers[0]
        criterion_crossentropyswap = criterions[0]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_tensors)) # Uniform sampling

        # Loss Plateau Detector
        loss_plateau_detector = SlopePlateauDetector(window_size=window, patience=patience, slope_threshold=threshold, use_ema=True, ema_alpha=ema_alpha)

        # Train model a bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_tensor = self.episodic_memory_tensors[batch_episodes_idxs].to(self.device)
            batch_episodes_actions = self.episodic_memory_actions[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            batch_episodes_logits = torch.empty(0).to(self.device)
            episode_gen_imgs = torch.empty(0).to(self.device)
            first_view_tensor_batch = batch_episodes_tensor[:,0,:,:,:]            
            for v in range(self.num_views):
                batch_actions = batch_episodes_actions[:,v]
                batch_actions = batch_actions[torch.randperm(batch_actions.size(0))] # shuffle actions (creates not seen generated images)
                ### Forward pass Semantic Memory with novel generated images
                with autocast():
                    batch_gen_imgs = self.conditional_generator(first_view_tensor_batch, batch_actions)
                    batch_gen_DecEnctensors = self.view_encoder(batch_gen_imgs) # shape (batch_size, 512, 7, 7)
                    batch_logits = self.semantic_memory(batch_gen_DecEnctensors)
                batch_episodes_logits = torch.cat([batch_episodes_logits, batch_logits.unsqueeze(1)], dim=1)
                # collect generated images for plot purposes (only first episode)
                episode_gen_imgs = torch.cat([episode_gen_imgs, batch_gen_imgs[0].unsqueeze(0)], dim=0)

            #### --- Calculate losses --- ####
            ### -> Semantic Memory Loss ###
            batch_pseudolabels = mira_pseudolabeling(logits = batch_episodes_logits, num_views = self.num_views, tau=self.tau_t, beta=self.beta, iters=30)
            loss_sem = criterion_crossentropyswap(batch_episodes_logits/self.tau_s, batch_pseudolabels)
            ### -> Total Loss ###
            loss = loss_sem

            #### --- Backward Pass --- ####
            optimizer_sem.zero_grad()
            scaler.scale(loss).backward()
            # Sanitycheck: check that gradients for conditional generator are zeros or None (since it is frozen)
            for name, param in self.conditional_generator.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)
            scaler.step(optimizer_sem)
            scaler.update()
            scheduler_sem.step()

            #### --- Update sleep counter --- ####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            sem_lr = scheduler_sem.get_last_lr()[0]
            loss_sem_value = loss_sem.item()
            loss_value = loss.item()
            first_view_logits = batch_episodes_logits[:,0].detach().cpu()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                ps = F.softmax(first_view_logits / self.tau_s, dim=1)
                pt = F.softmax(first_view_logits / self.tau_t, dim=1)
                _, _, mi_ps = statistics(ps)
                _, _, mi_pt = statistics(pt)
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- REM' +
                      f' -- sem lr: {sem_lr:.6f}' +
                      f' -- mi_ps: {mi_ps.item():.6f} -- mi_pt: {mi_pt.item():.6f}' +
                      f' -- Loss Semantic Memory: {loss_sem_value:.6f}' +
                      f' -- Loss Total: {loss_value:.6f}'
                      )
                writer.add_scalar('MI_ps', mi_ps.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                writer.add_scalar('MI_pt', mi_pt.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                
            #### --- Plot metrics --- ####
            writer.add_scalar('NREM-REM Indicator', rem_indicator, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('sem lr', sem_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Semantic Memory', loss_sem_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)

            #### --- Plot reconstructions --- ####
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (40*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                save_dir_recon=os.path.join(self.save_dir, 'reconstructions_during_training')
                os.makedirs(save_dir_recon, exist_ok=True)
                episode_gen_imgs = episode_gen_imgs.detach().cpu()
                episode_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(self.dataset_mean, self.dataset_std)], [1/s for s in self.dataset_std]) for img in episode_gen_imgs]
                episode_gen_imgs = torch.stack(episode_gen_imgs, dim=0)
                grid = torchvision.utils.make_grid(episode_gen_imgs, nrow=self.num_views)
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
                image_name = f'taskid_{task_id}_img_{self.sleep_episode_counter}_REM_reconstructed_images.png'
                grid.save(os.path.join(save_dir_recon, image_name))

            #### --- Check for plateau --- ####
            plateau_flag, abs_slope, smooth_loss = loss_plateau_detector.step(loss_value)
            writer.add_scalar(f'Total_Loss_Smooth_alpha_{ema_alpha}', smooth_loss, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if abs_slope is not None:
                writer.add_scalar('Abs_Slope', abs_slope, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if plateau_flag:
                print(f'**Loss Plateau detected. Switching to REM phase.')
                break

        # Clean up gradients
        optimizer_sem.zero_grad()

        return None

def evaluate_semantic_memory(loader,
                             view_encoder,
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
        semantic_memory.eval()
        
        all_logits = []
        all_probs = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(loader):
                images = images.to(device)
                logits = semantic_memory(view_encoder(images))
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
        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(all_labels.astype(int), all_preds.astype(int), calc_acc=calc_cluster_acc, total_probs=all_probs)

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
            palette_clustersID= cc.glasbey_hv[:len(clusters_IDs)]

            label_id_to_color = {label_id: palette_labelsID[idx] for idx, label_id in enumerate(range(num_gt_classes))}
            cluster_id_to_color = {cluster_id: palette_clustersID[idx] for idx, cluster_id in enumerate(sorted(clusters_IDs))}

            ### Plot number of samples per cluster with color per class
            df = pd.DataFrame(dict_class_vs_clusters)
            color_list = [label_id_to_color[i] for i in labels_IDs]
            df.plot.bar(stacked=True, figsize=(10, 8), color=color_list)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.title(f'Number of samples per cluster per class TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of samples')
            plt.xticks(rotation=0)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_number_samples_per_cluster_per_class.png'), bbox_inches='tight')
            plt.close()

        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5}

        return task_metrics

def evaluate_generator_batch(batch_episodes,
                            view_encoder, 
                            conditional_generator, 
                            criterion,
                            task_id,
                            device,
                            mean,
                            std,
                            save_dir):
    '''
    Evaluate view generator model on a hold-out batch
    '''
    view_encoder.eval()
    conditional_generator.eval()

    ### -- Foward pass batch and calculate loss values -- ###
    batch_episodes_imgs = batch_episodes[0].to(device)
    batch_episodes_actions = batch_episodes[1].to(device)
    batch_labels = batch_episodes[2]
    num_views = batch_episodes_imgs.size(1)
    with torch.no_grad():
        # Get first view tensors
        batch_first_view_images = batch_episodes_imgs[:,0]
        batch_first_view_tensors = view_encoder(batch_first_view_images)
        # Get views tensors, generated tensors, and generated images
        batch_episode_tensors = torch.empty(0).to(device)
        batch_episode_gen_FTtensors = torch.empty(0).to(device)
        batch_episode_gen_DecEnctensors = torch.empty(0).to(device)
        batch_episode_gen_DecEnctensors_direct = torch.empty(0).to(device)
        batch_episode_gen_imgs = torch.empty(0).to(device)
        for v in range(num_views):
            batch_imgs = batch_episodes_imgs[:,v]
            batch_actions = batch_episodes_actions[:,v]
            batch_tensors = view_encoder(batch_imgs)
            # Conditional forward pass
            batch_gen_images, batch_gen_FTtensors = conditional_generator(batch_first_view_tensors, batch_actions, return_transformed_feature_map=True)
            batch_gen_DecEnctensors = view_encoder(batch_gen_images)
            # Uncoditional forward pass
            batch_gen_images_direct = conditional_generator(batch_tensors, None, unconditional_mode=True)
            batch_gen_DecEnctensors_direct = view_encoder(batch_gen_images_direct)
            # Concatenate tensors
            batch_episode_tensors = torch.cat([batch_episode_tensors, batch_tensors.unsqueeze(1)], dim=1)
            batch_episode_gen_FTtensors = torch.cat([batch_episode_gen_FTtensors, batch_gen_FTtensors.unsqueeze(1)], dim=1)
            batch_episode_gen_DecEnctensors = torch.cat([batch_episode_gen_DecEnctensors, batch_gen_DecEnctensors.unsqueeze(1)], dim=1)
            batch_episode_gen_DecEnctensors_direct = torch.cat([batch_episode_gen_DecEnctensors_direct, batch_gen_DecEnctensors_direct.unsqueeze(1)], dim=1)
            # Concatenate generated images
            batch_episode_gen_imgs = torch.cat([batch_episode_gen_imgs, batch_gen_images.unsqueeze(1)], dim=1)
        lossgen_1 = criterion(batch_episode_gen_FTtensors, batch_episode_tensors)
        lossgen_2 = criterion(batch_episode_gen_DecEnctensors, batch_episode_tensors)
        lossgen_3 = criterion(batch_episode_gen_DecEnctensors_direct, batch_episode_tensors)
        total_genloss = lossgen_1 + lossgen_2 + lossgen_3

    ### -- Plot episodes images and gen images -- ###
    save_dir_recon=os.path.join(save_dir, 'trainbatch_reconstructions')
    os.makedirs(save_dir_recon, exist_ok=True)
    for i in range(batch_episodes_imgs.shape[0]):
        episode_views = batch_episodes_imgs[i]
        episode_gen_views = batch_episode_gen_imgs[i]
        episode_views = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_views]
        episode_gen_views = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_gen_views]
        episode_views = torch.stack(episode_views, dim=0)
        episode_gen_views = torch.stack(episode_gen_views, dim=0)
        grid = torchvision.utils.make_grid(torch.cat([episode_views, episode_gen_views], dim=0), nrow=num_views)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
        image_name = f'taskid_{task_id}_class_{batch_labels[i]}_recon_example.png'
        grid.save(os.path.join(save_dir_recon, image_name))

    return [lossgen_1, lossgen_2, lossgen_3, total_genloss]






    
    # def evaluate_generator(self, loader, device, save_dir=None, task_id=None):
    #     '''
    #     Evaluate view generator model
    #     '''
    #     self.view_encoder.eval()
    #     self.conditional_generator.eval()

    #     os.makedirs(save_dir, exist_ok=True)

    #     mean = self.dataset_mean
    #     std = self.dataset_std

    #     all_episodes = []
    #     all_actions = []
    #     all_labels = []
    #     classes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #     with torch.no_grad():
    #         for i, (batch, targets, _ ) in enumerate(loader):
    #             if targets[0] in classes_to_plot:
    #                 all_episodes.append(batch[0][0].unsqueeze(0))
    #                 all_actions.append(batch[1][0].unsqueeze(0))
    #                 all_labels.append(targets[0])
    #                 classes_to_plot.remove(targets[0])

    #     all_episodes = torch.cat(all_episodes, dim=0)
    #     all_actions = torch.cat(all_actions, dim=0)
    #     all_labels = np.array(all_labels)

    #     num_views = all_episodes.size(1)
    #     for i in range(all_episodes.shape[0]):
    #         episode_imgs = all_episodes[i]
    #         episode_actions = all_actions[i]
    #         episode_gtclass = all_labels[i]

    #         episode_first_view_imgs = episode_imgs[0,:,:,:]
    #         episode_first_view_imgs = episode_first_view_imgs.unsqueeze(0) # add view dimension
    #         episode_first_view_imgs = episode_first_view_imgs.repeat(num_views, 1, 1, 1) # repeat first view tensor for all views
    #         with torch.no_grad():
    #             episode_imgs_recon = self.conditional_generator(self.view_encoder(episode_first_view_imgs.to(device)), episode_actions.to(device)).cpu()

    #         episode_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_imgs]
    #         episode_imgs_recon = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_imgs_recon]
    #         episode_imgs = torch.stack(episode_imgs, dim=0)
    #         episode_imgs_recon = torch.stack(episode_imgs_recon, dim=0)

    #         grid = torchvision.utils.make_grid(torch.cat([episode_imgs, episode_imgs_recon], dim=0), nrow=num_views)
    #         grid = grid.permute(1, 2, 0).cpu().numpy()
    #         grid = (grid * 255).astype(np.uint8)
    #         grid = Image.fromarray(grid)
    #         image_name = f'taskid_{task_id}_original_reconstructed_images_class_{episode_gtclass}.png'
    #         grid.save(os.path.join(save_dir, image_name))

    #     return None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)