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

        # Train model a mini-bacth at a time
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
                # print(f'Abs Slope: {abs_slope}')
                writer.add_scalar('Abs Slope', abs_slope, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if plateau_flag:
                print(f'**Loss Plateau detected. Switching to REM phase.')
                break

        # Clean up gradients
        optimizer_condgen.zero_grad()

        return None
    
    def REM_sleep(self,
                  view_encoder,
                  projector_rep,
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
                  rem_indicator=1):
        '''
        Train conditional generator and semantic memory
        '''

        # unfreeze view encoder
        view_encoder.train()
        for param in view_encoder.parameters():
            param.requires_grad = True
        # freeze conditional generator
        conditional_generator.eval()
        for param in conditional_generator.parameters():
            param.requires_grad = False
        # unfreeze projector representation
        projector_rep.train()
        for param in projector_rep.parameters():
            param.requires_grad = True

        # optimizers, schedulers, and criterions
        optimizer_rep = optimizers[0]
        scheduler_rep = schedulers[0]
        criterion_rep = criterions[0]
        criterion_koleo = criterions[1]

        # Sampling weights for weighted random sampler (It defines the probability of each sample to be selected)
        sampling_weights = torch.ones(len(self.episodic_memory_tensors)) # Uniform sampling

        # Loss Plateau Detector
        loss_plateau_detector = SlopePlateauDetector(window_size=window, patience=patience, 
                                                     slope_threshold=threshold, use_smooth_loss=True, 
                                                     smooth_loss_alpha=smooth_loss_alpha)
        
        # Train model a mini-bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Sample episodes and actions --- ####
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))
            batch_episodes_tensor = self.episodic_memory_tensors[batch_episodes_idxs].to(self.device)
            batch_episodes_actions = self.episodic_memory_actions[batch_episodes_idxs].to(self.device)

            #### --- Forward pass --- ####
            episode_gen_imgs = torch.empty(0).to(self.device) # for plot purposes
            batch_episodes_outputs = torch.empty(0).to(self.device)
            if self.koleo_gamma !=0: batch_episodes_tensors = torch.empty(0).to(self.device)
            batch_first_view_tensors = batch_episodes_tensor[:,0,:,:,:]            
            for v in range(self.num_views):
                batch_actions = batch_episodes_actions[:,v]
                # shuffle actions in the mini-batch (creates novel episodes by using actions from other episodes)
                batch_actions = batch_actions[torch.randperm(batch_actions.size(0))] 
                ### Forward pass Semantic Memory with novel generated images
                with autocast():
                    batch_gen_imgs, _ = conditional_generator(batch_first_view_tensors, batch_actions)
                    batch_gen_DecEnctensors = view_encoder(batch_gen_imgs) # shape (batch_size, 512, 7, 7)
                    batch_outputs = projector_rep(batch_gen_DecEnctensors)
                batch_episodes_outputs = torch.cat([batch_episodes_outputs, batch_outputs.unsqueeze(1)], dim=1)
                if self.koleo_gamma !=0: # Collect tensors for koleo loss
                    batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_gen_DecEnctensors.unsqueeze(1)], dim=1)
                # collect generated images for plot purposes (only first episode)
                episode_gen_imgs = torch.cat([episode_gen_imgs, batch_gen_imgs[0].unsqueeze(0)], dim=0)

            #### --- Calculate losses --- ####
            ### -> Representation Learning loss ###
            loss_rep = criterion_rep(batch_episodes_outputs)
            ### -> Koleo loss ###
            if self.koleo_gamma != 0: loss_koleo = criterion_koleo(batch_episodes_tensors.mean(dim=(3,4))) # pass the average pooled version (koleo works on vectors) 
            else: loss_koleo = torch.tensor(0).to(self.device)

            ### -> Total Loss ###
            loss = loss_rep + self.koleo_gamma*loss_koleo

            #### --- Backward Pass --- ####
            optimizer_rep.zero_grad()
            scaler.scale(loss).backward()
            # Sanitycheck: check that gradients for conditional generator are zeros or None (since it is frozen)
            for name, param in conditional_generator.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)
            scaler.step(optimizer_rep)
            scaler.update()
            scheduler_rep.step()

            #### --- Update sleep counter --- ####
            self.sleep_episode_counter += self.episode_batch_size

            #### --- Print metrics --- ####
            rep_lr = scheduler_rep.get_last_lr()[0]
            loss_rep_value = loss_rep.item()
            loss_koleo_value = loss_koleo.item()
            loss_value = loss.item()
            if self.sleep_episode_counter == self.episode_batch_size or self.sleep_episode_counter % (5*self.episode_batch_size) == 0 or self.sleep_episode_counter == self.num_episodes_per_sleep:
                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- REM' +
                      f' -- rep lr: {rep_lr:.6f}' +
                      f' -- Loss Representation: {loss_rep_value:.6f}' +
                      f'{f" -- Loss Koleo: {loss_koleo_value:.6f}" if self.koleo_gamma > 0 else ""}' +
                      f' -- Loss Total: {loss_value:.6f}'
                      )
                
            #### --- Plot metrics --- ####
            writer.add_scalar('NREM-REM Indicator', rem_indicator, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('rep lr', rep_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Loss Representation', loss_rep_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if self.koleo_gamma > 0:
                writer.add_scalar('Loss Koleo', loss_koleo_value, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
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
                image_name = f'taskid_{task_id}_img_{self.sleep_episode_counter}_REM_reconstructed_images.png'
                grid.save(os.path.join(save_dir_recon, image_name))

            #### --- Check for plateau --- ####
            plateau_flag, abs_slope, smooth_loss = loss_plateau_detector.step(loss_value)
            writer.add_scalar(f'Total Loss Smooth', smooth_loss, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if abs_slope is not None:
                # print(f'Abs Slope: {abs_slope}')
                writer.add_scalar('Abs Slope', abs_slope, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if plateau_flag:
                print(f'**Loss Plateau detected. Switching to REM phase.')
                break

        # Clean up gradients
        optimizer_rep.zero_grad()

        return None
    
    def update_episodic_memory(self, view_encoder, device):
        '''
        Recalculate episodic_memory_tensors. Use view encoder and episodic_memory_images.
        Do a forward pass for each episode image and store the tensor in episodic_memory_tensors (replacing the old one)
        '''
        view_encoder.eval()
        view_encoder.to(device)

        with torch.no_grad():
            for i in range(0, len(self.episodic_memory_images), self.episode_batch_size):
                batch_episodes_imgs = self.episodic_memory_images[i:i+self.episode_batch_size].to(device)
                batch_episodes_tensors = torch.empty(0).to(device)
                for v in range(self.num_views):
                    batch_imgs = batch_episodes_imgs[:,v,:,:,:]
                    batch_tensors = view_encoder(batch_imgs)
                    batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_tensors.unsqueeze(1)], dim=1)
                self.episodic_memory_tensors[i:i+self.episode_batch_size] = batch_episodes_tensors.cpu()     
        return None
