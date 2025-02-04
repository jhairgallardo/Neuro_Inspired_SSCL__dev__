import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast

from evaluate_cluster import evaluate as eval_pred
from utils import statistics, encode_label

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

def find_slope(loss_history, window=100):
    if len(loss_history) < window: # if not enough iterations
        return np.inf
    
    loss_values = loss_history[-window:]  # Get last window iterations

    x = np.arange(len(loss_values), dtype=np.float32)  # Indices as x values
    y = np.array(loss_values, dtype=np.float32)  # Convert loss values to NumPy array

    # Compute means of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute the numerator and denominator for the slope formula
    numerator = np.sum((x - x_mean) * (y - y_mean))  # Covariance of x and y
    denominator = np.sum((x - x_mean) ** 2) + 1e-8  # Variance of x (avoid division by zero)

    # Compute slope
    slope = numerator / denominator

    return slope
    
    

class Wake_Sleep_trainer:
    def __init__(self, 
                 view_encoder, 
                 conditional_generator, 
                 semantic_memory, 
                 episode_batch_size, 
                 tau_t,
                 tau_s,
                 beta,
                 num_episodes_per_sleep,
                 device,
                 num_views,
                 save_dir,
                 dataset_mean,
                 dataset_std,
                 num_pseudoclasses,
                 ):
        self.view_encoder = view_encoder
        self.conditional_generator = conditional_generator
        self.semantic_memory = semantic_memory

        self.episodic_memory_tensors = torch.empty(0)
        self.episodic_memory_actions = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)

        self.episode_batch_size = episode_batch_size

        self.tau_t = tau_t
        self.tau_s = tau_s
        self.beta = beta

        self.firstwake=True
        self.sleep_episode_counter = 0
        self.num_episodes_per_sleep = num_episodes_per_sleep

        self.device = device
        self.num_views = num_views
        self.save_dir = save_dir
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.num_pseudoclasses = num_pseudoclasses
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        self.view_encoder.eval()

        aux_memory_tensors = {}
        aux_memory_actions = {}
        aux_memory_labels = {}
        for i, (episode_batch, labels , _ ) in enumerate(tqdm(incoming_dataloader)):
            episode_batch_imgs = episode_batch[0]
            episode_batch_actions = episode_batch[1]
            episode_labels = labels.unsqueeze(1).repeat(1, episode_batch_imgs.size(1))

            # forwards pass to get tensors
            episode_batch_imgs = episode_batch_imgs.to(self.device)
            batch_tensors = torch.empty(0).to(self.device)
            with torch.no_grad():
                for v in range(self.num_views):
                    img_batch = episode_batch_imgs[:,v,:,:,:]
                    tensors_out = self.view_encoder(img_batch) # shape (batch_size, 512, 7, 7)
                    batch_tensors = torch.cat([batch_tensors, tensors_out.unsqueeze(1)], dim=1)
            # collect episodes
            aux_memory_tensors[i] = batch_tensors.cpu()
            aux_memory_actions[i] = episode_batch_actions
            aux_memory_labels[i] = episode_labels

        # Concatenate all
        aux_memory_tensors = list(aux_memory_tensors.values())
        aux_memory_tensors = torch.cat(aux_memory_tensors, dim=0)
        self.episodic_memory_tensors = torch.cat([self.episodic_memory_tensors, aux_memory_tensors], dim=0)

        aux_memory_actions = list(aux_memory_actions.values())
        aux_memory_actions = torch.cat(aux_memory_actions, dim=0)
        self.episodic_memory_actions = torch.cat([self.episodic_memory_actions, aux_memory_actions], dim=0)

        aux_memory_labels = list(aux_memory_labels.values())
        aux_memory_labels = torch.cat(aux_memory_labels, dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        # Reset sleep iteration counter
        self.sleep_episode_counter = 0

        return None
    
    def NREM_sleep(self,
                    optimizers,
                    schedulers,
                    criterions,
                    writer=None, 
                    task_id=None,
                    scaler=None,
                    patience=40,
                    threshold=1e-3,
                    window=50):
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

        # optimizers and schedulers
        optimizer_sm = optimizers[0]
        scheduler_sm = schedulers[0]
        optimizer_gen = optimizers[1]
        scheduler_gen = schedulers[1]

        criterion_crossentropyswap = criterions[0]
        criterion_mse = criterions[1]

        train_logits = []
        train_probs = []
        train_preds = []
        train_gtlabels = []

        loss_gen1_fttensor_tracker = AverageMeter('FTtensor_loss', ':.6f')
        loss_gen2_gentensor_tracker = AverageMeter('GENtensor_loss', ':.6f')
        loss_gen3_uncondgentensor_tracker = AverageMeter('UncondGENtensor_loss', ':.6f')
        loss_history = []

        sampling_weights = torch.ones(len(self.episodic_memory_tensors))

        # Train model on sleep episodes a bacth at a time
        min_loss = float('inf')
        patience_counter = 0
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            
            #### -- Sample batch idxs of episodes -- ####
            # Uniform sampling
            batch_episodes_idxs = list(WeightedRandomSampler(sampling_weights, self.episode_batch_size, replacement=False))

            #### -- Get Data -- ####
            batch_episodes_tensor = self.episodic_memory_tensors[batch_episodes_idxs]
            batch_episodes_actions = self.episodic_memory_actions[batch_episodes_idxs]
            batch_episodes_tensor = batch_episodes_tensor.to(self.device)
            batch_episodes_actions = batch_episodes_actions.to(self.device)

            #### -- Forward pass -- ####
            batch_episodes_FTtensor = torch.empty(0).to(self.device)
            batch_episodes_GENtensor = torch.empty(0).to(self.device)
            batch_episodes_GENtensor_direct = torch.empty(0).to(self.device)
            batch_episodes_GENimgs = torch.empty(0)
            batch_episodes_logits = torch.empty(0).to(self.device)
            
            first_view_tensor_batch = batch_episodes_tensor[:,0,:,:,:]            
            for v in range(self.num_views):
                tensor_batch = batch_episodes_tensor[:,v,:,:,:]
                action_batch = batch_episodes_actions[:,v]

                ### Forward pass Conditional Generator (from first view to augmented view. Using corresponding action)
                with autocast():
                    GENimg_batch, FTtensor_batch = self.conditional_generator(first_view_tensor_batch, action_batch, return_transformed_feature_map=True)
                    GENtensor_batch = self.view_encoder(GENimg_batch) # shape (batch_size, 512, 7, 7)
                batch_episodes_FTtensor = torch.cat([batch_episodes_FTtensor, FTtensor_batch.unsqueeze(1)], dim=1)
                batch_episodes_GENtensor = torch.cat([batch_episodes_GENtensor, GENtensor_batch.unsqueeze(1)], dim=1)

                ### Forward pass Conditional Generator (from augmented view to augmented view. Unconditional mode. Direct reconstruction without FT)
                with autocast():
                    GENimg_direct_batch_ = self.conditional_generator(tensor_batch, None, unconditional_mode=True)
                    GENtensor_direct_batch_ = self.view_encoder(GENimg_direct_batch_) # shape (batch_size, 512, 7, 7)
                batch_episodes_GENtensor_direct = torch.cat([batch_episodes_GENtensor_direct, GENtensor_direct_batch_.unsqueeze(1)], dim=1)

                batch_episodes_GENimgs = torch.cat([batch_episodes_GENimgs, GENimg_batch.unsqueeze(1).detach().cpu()], dim=1)

                ### Forward pass Semantic Memory
                with autocast():
                    logit_batch = self.semantic_memory(tensor_batch)
                batch_episodes_logits = torch.cat([batch_episodes_logits, logit_batch.unsqueeze(1)], dim=1)

            #### -- Calculate losses -- ####

            ### Conditional Generator losses
            ## Reconstruction loss from feature transformation tensor and saved tensor
            lossgen_1 = criterion_mse(batch_episodes_FTtensor, batch_episodes_tensor)
            ## Reconstruction loss from generated tensor and saved tensor
            lossgen_2 = criterion_mse(batch_episodes_GENtensor, batch_episodes_tensor)
            ## Reconstruction loss (direct use of tensor to create gentensor. No FT. Unconditional)
            lossgen_3 = criterion_mse(batch_episodes_GENtensor_direct, batch_episodes_tensor)
            ## Generator loss
            loss_gen = lossgen_1 + lossgen_2 + lossgen_3

            ### Track generator losses
            loss_gen1_fttensor_tracker.update(lossgen_1.item(), self.episode_batch_size)
            loss_gen2_gentensor_tracker.update(lossgen_2.item(), self.episode_batch_size)
            loss_gen3_uncondgentensor_tracker.update(lossgen_3.item(), self.episode_batch_size)

            ### Semantic Memory losses
            ## Get MIRA pseudolabels
            batch_labels = mira_pseudolabeling(logits = batch_episodes_logits, 
                                            num_views = self.num_views,
                                            tau=self.tau_t, 
                                            beta=self.beta, 
                                            iters=30)
            ## Cross Entropy Swap Loss
            crossentropyswap_loss = criterion_crossentropyswap(batch_episodes_logits/self.tau_s, batch_labels)
            
            ### Total Loss
            loss = crossentropyswap_loss + loss_gen

            #### -- Backward Pass -- ####
            optimizer_sm.zero_grad()
            optimizer_gen.zero_grad()

            scaler.scale(loss).backward()

            # Sanitycheck: check that gradients for view encoder are zeros or None (since it is frozen)
            for name, param in self.view_encoder.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)

            scaler.step(optimizer_sm)
            scaler.step(optimizer_gen)
            scaler.update()
            scheduler_sm.step()
            scheduler_gen.step()

            ### -- Update sleep counter -- ####
            self.sleep_episode_counter += self.episode_batch_size

            #### -- Accumulate for metrics (only first view) -- #### 
            first_view_logits = batch_episodes_logits[:,0].detach().cpu()
            first_view_probs = F.softmax(first_view_logits/self.tau_s, dim=1)
            first_view_preds = torch.argmax(first_view_probs, dim=1)
            first_view_gtlabels = self.episodic_memory_labels[batch_episodes_idxs][:,0]
            train_logits.append(first_view_logits)
            train_probs.append(first_view_probs)
            train_preds.append(first_view_preds)
            train_gtlabels.append(first_view_gtlabels)

            gen_lr = scheduler_gen.get_last_lr()[0]
            sm_lr = scheduler_sm.get_last_lr()[0]
            
            #### -- Print metrics -- ####
            if self.sleep_episode_counter == self.episode_batch_size or \
               (self.sleep_episode_counter // self.episode_batch_size) % 5 == 0 or \
               self.sleep_episode_counter == self.num_episodes_per_sleep:
                
                ps = F.softmax(first_view_logits / self.tau_s, dim=1)
                pt = F.softmax(first_view_logits / self.tau_t, dim=1)
                _, _, mi_ps = statistics(ps)
                _, _, mi_pt = statistics(pt)

                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- NREM (Indicator={0})' + # NREM: 0 , REM: 1
                      f' -- gen lr: {gen_lr:.6f}' +
                      f' -- sm lr: {sm_lr:.6f}' +
                      f' -- mi_ps: {mi_ps.item():.6f} -- mi_pt: {mi_pt.item():.6f}' +
                      f' -- CrossEntropySwap Loss: {crossentropyswap_loss.item():.6f}' +
                      f' -- FTtensor Loss: {lossgen_1.item():.6f}' +
                      f' -- CondGENtensor Loss: {lossgen_2.item():.6f}' +
                      f' -- UncondGENtensor Loss: {lossgen_3.item():.6f}' +
                      f' -- Total: {loss.item():.6f}'
                      )
                writer.add_scalar('MI_ps', mi_ps.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                writer.add_scalar('MI_pt', mi_pt.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                
            #### -- Plot metrics -- ####
            writer.add_scalar('CrossEntropySwap Loss', crossentropyswap_loss.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('FTtensor Loss', lossgen_1.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('CondGENtensor Loss', lossgen_2.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('UncondGENtensor Loss', lossgen_3.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('gen lr', gen_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('sm lr', sm_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('NREM-REM Indicator', 0, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            
            #### -- Plot reconstructions -- ####
            if self.sleep_episode_counter == self.episode_batch_size or \
               (self.sleep_episode_counter // self.episode_batch_size) % 40 == 0 or \
               self.sleep_episode_counter == self.num_episodes_per_sleep:
                
                save_dir_recon=os.path.join(self.save_dir, 'reconstructions_during_training')
                os.makedirs(save_dir_recon, exist_ok=True)
                episode_imgs_recon = batch_episodes_GENimgs[0,:self.num_views,:,:,:]#.detach().cpu()
                episode_imgs_recon = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(self.dataset_mean, self.dataset_std)], [1/s for s in self.dataset_std]) for img in episode_imgs_recon]
                episode_imgs_recon = torch.stack(episode_imgs_recon, dim=0)

                grid = torchvision.utils.make_grid(episode_imgs_recon, nrow=self.num_views)
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
                image_name = f'taskid_{task_id}_img_{self.sleep_episode_counter}_NREM_reconstructed_images.png'
                grid.save(os.path.join(save_dir_recon, image_name))

            
            
            # Finding plateau by checking if loss has not decreased in the last patience iterations
            # if loss.item() < min_loss:
            #     min_loss = loss.item()
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print(f'**Loss Plateau detected. (min val patience) Switching to REM phase.')
            #         break
            
            # Finding plateau by checking the slope of the loss in a window of iterations. If slope has been low for the last patience iterations, switch to REM phase
            loss_history.append(loss.item())
            slope = find_slope(loss_history, window=window)
            if slope != np.inf:
                writer.add_scalar('Slope', slope, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if abs(slope) < threshold:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'**Loss Plateau detected. (slope patience) Switching to REM phase.')
                    break
            else:
                patience_counter = 0
                
            



        #### Track train metrics ####
        train_logits = torch.cat(train_logits).numpy()
        train_probs = torch.cat(train_probs).numpy()
        train_preds = torch.cat(train_preds).numpy()
        train_gtlabels = torch.cat(train_gtlabels).numpy()

        calc_cluster_acc = len(np.unique(train_gtlabels)) == self.num_pseudoclasses
        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(train_gtlabels.astype(int), train_preds.astype(int), calc_acc=calc_cluster_acc, total_probs=train_probs)
        
        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5,
                        'FTtensor_loss': loss_gen1_fttensor_tracker.avg,
                        'GENtensor_loss': loss_gen2_gentensor_tracker.avg,
                        'UncondGENtensor_loss': loss_gen3_uncondgentensor_tracker.avg}

        # Clean up gradients
        optimizer_sm.zero_grad()
        optimizer_gen.zero_grad()

        return task_metrics
    
    def REM_sleep(self,
                    optimizers, 
                    schedulers,
                    criterions, 
                    writer=None, 
                    task_id=None,
                    scaler=None,
                    patience=40,
                    threshold=1e-3,
                    window=50):
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

        criterion_crossentropyswap = criterions[0]

        # optimizers and schedulers
        optimizer_sm = optimizers[0]
        scheduler_sm = schedulers[0]

        train_logits = []
        train_probs = []
        train_preds = []
        train_gtlabels = []

        loss_history = []

        weights = torch.ones(len(self.episodic_memory_tensors))

        # Train model on sleep episodes a bacth at a time
        min_loss = float('inf')
        patience_counter = 0
        while self.sleep_episode_counter < self.num_episodes_per_sleep:

            #### -- Sample batch idxs of episodes -- ####
            # Uniform sampling
            batch_episodes_idxs = list(WeightedRandomSampler(weights, self.episode_batch_size, replacement=False))

            #### -- Data -- ####
            batch_episodes_tensor = self.episodic_memory_tensors[batch_episodes_idxs]
            batch_episodes_actions = self.episodic_memory_actions[batch_episodes_idxs]
            batch_episodes_tensor = batch_episodes_tensor.to(self.device)
            batch_episodes_actions = batch_episodes_actions.to(self.device)

            #### -- Forward pass -- ####
            batch_episodes_GENimgs = torch.empty(0)
            batch_episodes_logits = torch.empty(0).to(self.device)
            
            first_view_tensor_batch = batch_episodes_tensor[:,0,:,:,:]            
            for v in range(self.num_views):
                action_batch = batch_episodes_actions[:,v]

                # shuffle actions (creates not seen generated images)
                action_batch = action_batch[torch.randperm(action_batch.size(0))]

                ### Forward pass
                with autocast():
                    GENimg_batch = self.conditional_generator(first_view_tensor_batch, action_batch)
                    GENtensor_batch = self.view_encoder(GENimg_batch) # shape (batch_size, 512, 7, 7)
                    logit_batch = self.semantic_memory(GENtensor_batch)
                batch_episodes_GENimgs = torch.cat([batch_episodes_GENimgs, GENimg_batch.unsqueeze(1).detach().cpu()], dim=1)
                batch_episodes_logits = torch.cat([batch_episodes_logits, logit_batch.unsqueeze(1)], dim=1)

            #### -- Calculate losses -- ####
            ### Semantic Memory losses
            ## Get MIRA pseudolabels
            batch_labels = mira_pseudolabeling(logits = batch_episodes_logits, 
                                            num_views = self.num_views,
                                            tau=self.tau_t, 
                                            beta=self.beta, 
                                            iters=30)
            ## Cross Entropy Swap Loss
            crossentropyswap_loss = criterion_crossentropyswap(batch_episodes_logits/self.tau_s, batch_labels)
            
            ### Total Loss
            loss = crossentropyswap_loss

            #### -- Backward Pass -- ####
            optimizer_sm.zero_grad()
            scaler.scale(loss).backward()

            # Sanitycheck: check that gradients for conditional generator are zeros or None (since it is frozen)
            for name, param in self.conditional_generator.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)

            scaler.step(optimizer_sm)
            scaler.update()
            scheduler_sm.step()

            ### -- Update sleep counter -- ####
            self.sleep_episode_counter += self.episode_batch_size

            #### -- Accumulate for metrics (only first view) -- #### 
            first_view_logits = batch_episodes_logits[:,0].detach().cpu()
            first_view_probs = F.softmax(first_view_logits/self.tau_s, dim=1)
            first_view_preds = torch.argmax(first_view_probs, dim=1)
            first_view_gtlabels = self.episodic_memory_labels[batch_episodes_idxs][:,0]
            train_logits.append(first_view_logits)
            train_probs.append(first_view_probs)
            train_preds.append(first_view_preds)
            train_gtlabels.append(first_view_gtlabels)

            sm_lr = scheduler_sm.get_last_lr()[0]


            #### -- Print metrics -- ####
            if self.sleep_episode_counter == self.episode_batch_size or \
               (self.sleep_episode_counter // self.episode_batch_size) % 5 == 0 or \
               self.sleep_episode_counter == self.num_episodes_per_sleep:
                
                ps = F.softmax(first_view_logits / self.tau_s, dim=1)
                pt = F.softmax(first_view_logits / self.tau_t, dim=1)
                _, _, mi_ps = statistics(ps)
                _, _, mi_pt = statistics(pt)

                print(f'Episode [{self.sleep_episode_counter}/{self.num_episodes_per_sleep}]' +
                      f' -- REM (Indicator={1})' + # NREM: 0 , REM: 1
                      f' -- sm lr: {sm_lr:.6f}' +
                      f' -- mi_ps: {mi_ps.item():.6f} -- mi_pt: {mi_pt.item():.6f}' +
                      f' -- CrossEntropySwap: {crossentropyswap_loss.item():.6f}' +
                      f' -- Total: {loss.item():.6f}'
                      )
                writer.add_scalar('MI_ps', mi_ps.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
                writer.add_scalar('MI_pt', mi_pt.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            
            #### -- Plot metrics -- ####
            writer.add_scalar('CrossEntropySwap Loss', crossentropyswap_loss.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('Total Loss', loss.item(), task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('sm lr', sm_lr, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            writer.add_scalar('NREM-REM Indicator', 1, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            
            #### -- Plot reconstructions -- ####
            if self.sleep_episode_counter == self.episode_batch_size or \
               (self.sleep_episode_counter // self.episode_batch_size) % 40 == 0 or \
               self.sleep_episode_counter == self.num_episodes_per_sleep:
                
                #### Plot reconstructions
                save_dir_recon=os.path.join(self.save_dir, 'reconstructions_during_training')
                os.makedirs(save_dir_recon, exist_ok=True)
                episode_imgs_recon = batch_episodes_GENimgs[0,:self.num_views,:,:,:]#.detach().cpu()
                episode_imgs_recon = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(self.dataset_mean, self.dataset_std)], [1/s for s in self.dataset_std]) for img in episode_imgs_recon]
                episode_imgs_recon = torch.stack(episode_imgs_recon, dim=0)

                grid = torchvision.utils.make_grid(episode_imgs_recon, nrow=self.num_views)
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
                image_name = f'taskid_{task_id}_img_{self.sleep_episode_counter}_REM_reconstructed_images.png'
                grid.save(os.path.join(save_dir_recon, image_name))

            # Finding plateau by checking if loss has not decreased in the last patience iterations
            # if loss.item() < min_loss:
            #     min_loss = loss.item()
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print(f'**Loss Plateau detected. (min val patience) Switching to REM phase.')
            #         break
            
            # Finding plateau by checking the slope of the loss in a window of iterations. If slope has been low for the last patience iterations, switch to REM phase
            loss_history.append(loss.item())
            slope = find_slope(loss_history, window=window)
            if slope != np.inf:
                writer.add_scalar('Slope', slope, task_id*self.num_episodes_per_sleep + self.sleep_episode_counter)
            if abs(slope) < threshold:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'**Loss Plateau detected. (slope patience) Switching to REM phase.')
                    break
            else:
                patience_counter = 0


        #### Track train metrics ####
        train_logits = torch.cat(train_logits).numpy()
        train_probs = torch.cat(train_probs).numpy()
        train_preds = torch.cat(train_preds).numpy()
        train_gtlabels = torch.cat(train_gtlabels).numpy()

        calc_cluster_acc = len(np.unique(train_gtlabels)) == self.num_pseudoclasses
        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(train_gtlabels.astype(int), train_preds.astype(int), calc_acc=calc_cluster_acc, total_probs=train_probs)
        
        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5}

        # Clean up gradients
        optimizer_sm.zero_grad()

        return task_metrics
    
    def evaluate_semantic_memory(self, 
                                val_loader,
                                num_gt_classes=None,
                                plot_clusters=False, 
                                save_dir_clusters=None, 
                                task_id=None, 
                                mean=None, 
                                std=None):
        '''
        Evaluate model on validation set
        '''
        self.semantic_memory.eval()
        self.view_encoder.eval()

        all_logits = []
        all_probs = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(val_loader):
                images = images.to(self.device)
                logits = self.semantic_memory(self.view_encoder(images))
                probs = F.softmax(logits/self.tau_s, dim=1)
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

        calc_cluster_acc = len(np.unique(all_labels)) == self.num_pseudoclasses
        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(all_labels.astype(int), all_preds.astype(int), calc_acc=calc_cluster_acc, total_probs=all_probs)

        if plot_clusters:
            assert num_gt_classes is not None
            assert save_dir_clusters is not None
            assert task_id is not None
            assert mean is not None
            assert std is not None

            dict_class_vs_clusters = {}
            labels_IDs = np.unique(all_labels)
            clusters_IDs = np.unique(all_preds)
            for i in labels_IDs:
                dict_class_vs_clusters[f'Class {i}'] = []
                for j in range(self.num_pseudoclasses):
                    indices = (all_labels==i) & (all_preds==j)
                    dict_class_vs_clusters[f'Class {i}'].append(np.sum(indices))
            palette_labelsID= cc.glasbey_category10[:num_gt_classes]
            palette_clustersID= cc.glasbey_hv[:len(clusters_IDs)]

            label_id_to_color = {label_id: palette_labelsID[idx] for idx, label_id in enumerate(range(num_gt_classes))}
            cluster_id_to_color = {cluster_id: palette_clustersID[idx] for idx, cluster_id in enumerate(sorted(clusters_IDs))}

            ### plot 25 images per cluster
            os.makedirs(save_dir_clusters, exist_ok=True)
            for i in range(self.num_pseudoclasses): 
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
                image_name = f'taskid_{task_id}_pseudoclass_{i}.png'
                grid.save(os.path.join(save_dir_clusters, image_name))
                if i == 49: # Only plot 50 clusters (from 0 to 49)
                    break

            name = save_dir_clusters.split('/')[-1]

            ### Plot all logits in a 2D space. (t-SNE).
            tsne = TSNE(n_components=2, random_state=0)
            all_logits_2d = tsne.fit_transform(all_logits)
            # Legend is cluster ID
            plt.figure(figsize=(8, 8))
            for i in clusters_IDs:
                indices = all_preds==i
                plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Cluster {i}', alpha=0.75, s=20, color=cluster_id_to_color[i])
            plt.title(f'{name}\nLogits 2D space TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(clusters_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_logits_2d_space_clusters.png'), bbox_inches='tight')
            plt.close()
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20, color=label_id_to_color[i])
            plt.title(f'{name}\nLogits 2D space TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_logits_2d_space_labels.png'), bbox_inches='tight')
            plt.close()

            ### Plot all logits in a 2D space (PCA)
            pca = PCA(n_components=2, random_state=0)
            all_logits_pca_2d = pca.fit_transform(all_logits)
            # Legend is cluster ID
            plt.figure(figsize=(8, 8))
            for i in clusters_IDs:
                indices = all_preds==i
                plt.scatter(all_logits_pca_2d[indices, 0], all_logits_pca_2d[indices, 1], label=f'Cluster {i}', alpha=0.75, s=20, color=cluster_id_to_color[i])
            plt.title(f'{name}\nLogits PCA 2D space TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(clusters_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_logits_pca_2d_space_clusters.png'), bbox_inches='tight')
            plt.close()
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_logits_pca_2d[indices, 0], all_logits_pca_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20, color=label_id_to_color[i])
            plt.title(f'{name}\nLogits PCA 2D space TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_logits_pca_2d_space_labels.png'), bbox_inches='tight')
            plt.close()

            ### Plot headmap showing cosine similarity matrix of each cluster weights
            clusters_weights = self.semantic_memory.module.linear_head.weight.data.cpu()
            clusters_weights = F.normalize(clusters_weights, p=2, dim=1)
            clusters_cosine_sim = torch.mm(clusters_weights, clusters_weights.T)
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(clusters_cosine_sim, cmap='viridis', ax=ax, annot= self.num_pseudoclasses==10 , vmax=1, vmin=-1)
            plt.title(f'{name}\nCosine Similarity Matrix TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Cluster ID')
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_cosine_similarity_matrix.png'), bbox_inches='tight')
            plt.close()

            ### Plot number of samples per cluster with color per class
            df = pd.DataFrame(dict_class_vs_clusters)
            color_list = [label_id_to_color[i] for i in labels_IDs]
            df.plot.bar(stacked=True, figsize=(10, 8), color=color_list)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.title(f'{name}\nNumber of samples per cluster per class TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of samples')
            plt.xticks(rotation=0)
            plt.savefig(os.path.join(save_dir_clusters, f'taskid_{task_id}_number_samples_per_cluster_per_class.png'), bbox_inches='tight')
            plt.close()

        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5}

        return task_metrics
    
    def evaluate_generator(self, loader, device, save_dir=None, task_id=None):
        '''
        Evaluate view generator model
        '''
        self.view_encoder.eval()
        self.conditional_generator.eval()

        os.makedirs(save_dir, exist_ok=True)

        mean = self.dataset_mean
        std = self.dataset_std

        all_episodes = []
        all_actions = []
        all_labels = []
        classes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        with torch.no_grad():
            for i, (batch, targets, _ ) in enumerate(loader):
                if targets[0] in classes_to_plot:
                    all_episodes.append(batch[0][0].unsqueeze(0))
                    all_actions.append(batch[1][0].unsqueeze(0))
                    all_labels.append(targets[0])
                    classes_to_plot.remove(targets[0])

        all_episodes = torch.cat(all_episodes, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        all_labels = np.array(all_labels)

        num_views = all_episodes.size(1)
        for i in range(all_episodes.shape[0]):
            episode_imgs = all_episodes[i]
            episode_actions = all_actions[i]
            episode_gtclass = all_labels[i]

            episode_first_view_imgs = episode_imgs[0,:,:,:]
            episode_first_view_imgs = episode_first_view_imgs.unsqueeze(0) # add view dimension
            episode_first_view_imgs = episode_first_view_imgs.repeat(num_views, 1, 1, 1) # repeat first view tensor for all views
            with torch.no_grad():
                episode_imgs_recon = self.conditional_generator(self.view_encoder(episode_first_view_imgs.to(device)), episode_actions.to(device)).cpu()

            episode_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_imgs]
            episode_imgs_recon = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_imgs_recon]
            episode_imgs = torch.stack(episode_imgs, dim=0)
            episode_imgs_recon = torch.stack(episode_imgs_recon, dim=0)

            grid = torchvision.utils.make_grid(torch.cat([episode_imgs, episode_imgs_recon], dim=0), nrow=num_views)
            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            grid = Image.fromarray(grid)
            image_name = f'taskid_{task_id}_original_reconstructed_images_class_{episode_gtclass}.png'
            grid.save(os.path.join(save_dir, image_name))

        return None

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