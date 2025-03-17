import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast

import numpy as np
from PIL import Image

from tqdm import tqdm

class SlopePlateauDetector:
    def __init__(self, window_size=50, patience=40, slope_threshold=1e-3, use_smooth_loss=False, smooth_loss_alpha=0.1):
        self.window_size = window_size
        self.patience = patience
        self.slope_threshold = slope_threshold
        self.use_smooth_loss = use_smooth_loss
        self.smooth_loss_alpha = smooth_loss_alpha
        self.loss_window_history = []  # fixed-size list for storing recent loss (or smoothed loss) values
        self.consecutive_flat_windows = 0
        self.smooth_loss = None  # stores the current EMA of the loss

    def step(self, loss):
        # Update the EMA if smoothing is enabled; otherwise, use the raw loss.
        if self.use_smooth_loss:
            if self.smooth_loss is None:
                self.smooth_loss = loss
            else:
                self.smooth_loss = self.smooth_loss_alpha * loss + (1 - self.smooth_loss_alpha) * self.smooth_loss
            current_value = self.smooth_loss
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
                 episode_batch_size,
                 num_episodes_per_sleep,
                 num_views,
                 dataset_mean,
                 dataset_std,
                 device,
                 save_dir,
                 koleo_gamma
                 ):
        self.episode_batch_size = episode_batch_size
        self.num_episodes_per_sleep = num_episodes_per_sleep
        self.num_views = num_views

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.device = device
        self.save_dir = save_dir

        self.koleo_gamma = koleo_gamma

        self.episodic_memory_images = torch.empty(0)
        self.episodic_memory_tensors = torch.empty(0)
        self.episodic_memory_actions = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)

        self.sleep_episode_counter = 0
    
    def wake_phase(self, view_encoder, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        view_encoder.eval()

        aux_memory_imgs = {}
        aux_memory_tensors = {}
        aux_memory_actions = {}
        aux_memory_labels = {}
        for i, (batch_episodes, batch_labels , _ ) in enumerate(tqdm(incoming_dataloader)):
            batch_episodes_imgs = batch_episodes[0].to(self.device)
            batch_episodes_actions = batch_episodes[1]
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1))

            # forwards pass to get tensors
            batch_episodes_tensors = torch.empty(0).to(self.device)
            with torch.no_grad():
                for v in range(self.num_views):
                    batch_imgs = batch_episodes_imgs[:,v,:,:,:]
                    batch_tensors = view_encoder(batch_imgs) # shape (batch_size, 512, 7, 7)
                    batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_tensors.unsqueeze(1)], dim=1)
            # collect episodes
            aux_memory_imgs[i] = batch_episodes_imgs.cpu()
            aux_memory_tensors[i] = batch_episodes_tensors.cpu()
            aux_memory_actions[i] = batch_episodes_actions
            aux_memory_labels[i] = batch_episodes_labels

        # Concatenate all
        aux_memory_imgs = torch.cat(list(aux_memory_imgs.values()), dim=0)
        aux_memory_tensors = torch.cat(list(aux_memory_tensors.values()), dim=0)
        aux_memory_actions = torch.cat(list(aux_memory_actions.values()), dim=0)
        aux_memory_labels = torch.cat(list(aux_memory_labels.values()), dim=0)
        self.episodic_memory_images = torch.cat([self.episodic_memory_images, aux_memory_imgs], dim=0)
        self.episodic_memory_tensors = torch.cat([self.episodic_memory_tensors, aux_memory_tensors], dim=0)
        self.episodic_memory_actions = torch.cat([self.episodic_memory_actions, aux_memory_actions], dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        # Reset sleep iteration counter
        self.sleep_episode_counter = 0

        return None
    
    def NREM_sleep(self,
                   view_encoder,
                   conditional_generator,
                   optimizers,
                   schedulers,
                   criterions,
                   task_id=None,
                   patience=40,
                   threshold=1e-3,
                   window=50,
                   smooth_loss_alpha=0.3,
                   scaler=None,
                   writer=None,
                   nrem_indicator=0):
        '''
        Train conditional generator and semantic memory
        '''

        # freeze view encoder
        view_encoder.eval()
        for param in view_encoder.parameters():
            param.requires_grad = False
        
        # unfreeze conditional generator
        conditional_generator.train() 
        for param in conditional_generator.parameters():
            param.requires_grad = True

        # optimizers, schedulers, and criterions
        optimizer_condgen = optimizers[0]
        scheduler_condgen = schedulers[0]
        criterion_mse = criterions[0]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_tensors)) # Uniform sampling

        # Loss Plateau Detector
        loss_plateau_detector = SlopePlateauDetector(window_size=window, patience=patience, 
                                                     slope_threshold=threshold, use_smooth_loss=True, 
                                                     smooth_loss_alpha=smooth_loss_alpha)

        # Train model a bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_tensor = self.episodic_memory_tensors[batch_episodes_idxs].to(self.device)
            batch_episodes_actions = self.episodic_memory_actions[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            episode_gen_imgs = torch.empty(0).to(self.device) # for plot purposes
            batch_episodes_gen_FTtensor = torch.empty(0).to(self.device)
            batch_episodes_gen_DecEnctensors = torch.empty(0).to(self.device)
            batch_episodes_gen_DecEnctensors_direct = torch.empty(0).to(self.device)
            batch_first_view_tensors = batch_episodes_tensor[:,0,:,:,:]            
            for v in range(self.num_views):
                batch_tensors = batch_episodes_tensor[:,v,:,:,:]
                batch_actions = batch_episodes_actions[:,v]
                
                with autocast():
                    ### Forward pass Conditional Generator (from first view to augmented view. Using corresponding action)
                    batch_gen_imgs, batch_gen_FTtensor = conditional_generator(batch_first_view_tensors, batch_actions)
                    batch_gen_DecEnctensors = view_encoder(batch_gen_imgs)

                    ### Forward pass Conditional Generator direct (from augmented view to augmented view. Unconditional mode. Direct reconstruction without FTN)
                    batch_gen_imgs_direct = conditional_generator(batch_tensors, None, skip_FTN=True)
                    batch_gen_DecEnctensors_direct = view_encoder(batch_gen_imgs_direct)
                
                # concatenate tensors
                batch_episodes_gen_FTtensor = torch.cat([batch_episodes_gen_FTtensor, batch_gen_FTtensor.unsqueeze(1)], dim=1)
                batch_episodes_gen_DecEnctensors = torch.cat([batch_episodes_gen_DecEnctensors, batch_gen_DecEnctensors.unsqueeze(1)], dim=1)
                batch_episodes_gen_DecEnctensors_direct = torch.cat([batch_episodes_gen_DecEnctensors_direct, batch_gen_DecEnctensors_direct.unsqueeze(1)], dim=1)
                # collect generated images for plot purposes (only first episode)
                episode_gen_imgs = torch.cat([episode_gen_imgs, batch_gen_imgs[0].unsqueeze(0)], dim=0)

            #### --- Calculate losses --- ####
            ## Reconstruction loss from feature transformation tensor and saved tensor
            lossgen_1 = criterion_mse(batch_episodes_gen_FTtensor, batch_episodes_tensor)
            ## Reconstruction loss from generated tensor and saved tensor
            lossgen_2 = criterion_mse(batch_episodes_gen_DecEnctensors, batch_episodes_tensor)
            ## Reconstruction loss (direct use of tensor to create gentensor. No FT. Unconditional)
            lossgen_3 = criterion_mse(batch_episodes_gen_DecEnctensors_direct, batch_episodes_tensor)
            ## Generator loss
            loss_condgen = lossgen_1 + lossgen_2 + lossgen_3
            ### -> Total Loss ###
            loss = loss_condgen

            #### --- Backward Pass --- ####
            optimizer_condgen.zero_grad()
            scaler.scale(loss).backward()
            # Sanitycheck: check that gradients for view encoder are zeros or None (since it is frozen)
            for name, param in view_encoder.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)
            scaler.step(optimizer_condgen)
            scaler.update()
            scheduler_condgen.step()

            #### --- Update sleep counter --- #####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            condgen_lr = scheduler_condgen.get_last_lr()[0]
            loss_condgen_value = loss_condgen.item()
            lossgen_1_value = lossgen_1.item()
            lossgen_2_value = lossgen_2.item()
            lossgen_3_value = lossgen_3.item()
            loss_value = loss.item()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- NREM' +
                      f' -- condgen lr: {condgen_lr:.6f}' +
                      f' -- Loss Gen_1 FTtensor: {lossgen_1_value:.6f}' +
                      f' -- Loss Gen_2 DecEnctensor: {lossgen_2_value:.6f}' +
                      f' -- Loss Gen_3 DecEnctensor direct: {lossgen_3_value:.6f}' +
                      f' -- Loss Conditional Generator: {loss_condgen_value:.6f}' +
                      f' -- Loss Total: {loss_value:.6f}'
                      )
                
            #### --- Plot metrics --- ####
            writer.add_scalar('NREM-REM Indicator', nrem_indicator, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('condgen lr', condgen_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Gen_1 FTtensor', lossgen_1_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Gen_2 DecEnctensor', lossgen_2_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Gen_3 DecEnctensor direct', lossgen_3_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Conditional Generator', loss_condgen_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)

            #### --- Plot reconstructions --- ####
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (40*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                save_dir_recon=os.path.join(self.save_dir, 'Generated_images_during_training')
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
            writer.add_scalar(f'Total Loss Smooth', smooth_loss, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if abs_slope is not None:
                writer.add_scalar('Abs Slope', abs_slope, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if plateau_flag:
                print(f'**Loss Plateau detected. Switching to REM phase.')
                break

        # Clean up gradients
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
    
