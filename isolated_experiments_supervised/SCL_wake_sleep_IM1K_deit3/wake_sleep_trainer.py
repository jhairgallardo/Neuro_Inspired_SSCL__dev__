import os
import random

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast

import numpy as np
from PIL import Image

from tqdm import tqdm
from utils import MetricLogger, accuracy, reduce_tensor

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
                 patience,
                 threshold_nrem,
                 threshold_rem,
                 window,
                 smooth_loss_alpha,
                 device,
                 save_dir,
                 print_freq=10
                 ):
        self.episode_batch_size = episode_batch_size
        self.num_episodes_per_sleep = num_episodes_per_sleep
        self.num_views = num_views

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.patience = patience
        self.threshold_nrem = threshold_nrem
        self.threshold_rem = threshold_rem
        self.window = window
        self.smooth_loss_alpha = smooth_loss_alpha

        self.device = device
        self.save_dir = save_dir

        self.episodic_memory_tensors = []
        self.episodic_memory_labels = []
        self.episodic_memory_actions = []

        # create on‐disk directories once:
        for sub in ("episodic_memory_tensors",
                    "episodic_memory_labels",
                    "episodic_memory_actions"):
            path = os.path.join(self.save_dir, sub)
            os.makedirs(path, exist_ok=True)

        self.nrem_indicator = 0
        self.rem_indicator = 1

        self.sleep_episode_counter = 0

        self.total_num_seen_episodes = 0  # Total number of episodes seen so far

        self.print_freq = print_freq
    
    def append_memory(self,
                      tensor_paths:  list,
                      label_paths:  list,
                      action_paths: list):
        """
        Once every rank has received the new file‐lists, extend our master lists.
        """
        self.episodic_memory_tensors += tensor_paths
        self.episodic_memory_labels  += label_paths
        self.episodic_memory_actions += action_paths

        return None

    def reset_sleep_counter(self):
        """
        Reset the sleep episode counter to zero.
        This is called at the beginning of each sleep phase.
        """
        self.sleep_episode_counter = 0

        return None

    def replay_sampling_uniform(self, num_samples):
        """
        Sample a number of episodes uniformly from the episodic memory.
        """
        if len(self.episodic_memory_tensors) == 0:
            raise ValueError("Episodic memory is empty. Please collect data in the wake phase before sampling.")
        
        sampling_weights = torch.ones(len(self.episodic_memory_tensors)) # Uniform sampling across the whole episodic_memory
        sampled_indices = list(WeightedRandomSampler(sampling_weights, num_samples, replacement=True)) # We can repeat samples since num_samples>> len(self.episodic_memory_tensors)

        # Shuffle sampled indices to ensure randomness
        random.shuffle(sampled_indices)

        return sampled_indices
    
    def wake_phase(self, view_encoder, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        view_encoder.eval()

        next_idx = len(self.episodic_memory_tensors)  # next filename index

        new_tensors_path = []
        new_labels_path = []
        new_actions_path = []

        for batch_episodes, batch_labels , _  in tqdm(incoming_dataloader, desc="Wake Phase"):
            batch_episodes_imgs = batch_episodes[0].to(self.device)
            batch_episodes_actions = batch_episodes[1]
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1))
            B, V, C, H, W = batch_episodes_imgs.shape
            # forwards pass to get tensors
            with torch.no_grad():
                flat_episodes_imgs = batch_episodes_imgs.view(B*V, C, H, W) # (B*V, C, H, W)
                flat_episodes_tensors = view_encoder(flat_episodes_imgs) # (B*V, T, D)
                T = flat_episodes_tensors.size(1) # number of tokens (includes the cls token and feature tokens)
                D = flat_episodes_tensors.size(2) # feature dimension
                batch_episodes_tensors = flat_episodes_tensors.view(B, V, T, D).cpu().detach() # (B, V, T, D)

            # Save each episode individualy
            for i in range(B):
                idx = next_idx
                basename = f"{idx:08d}"
                # Tensors
                tensor_path = os.path.join(self.save_dir, "episodic_memory_tensors",  basename + ".npy")
                np.save(tensor_path, batch_episodes_tensors[i].numpy())
                new_tensors_path.append(tensor_path)
                # Labels
                label_path = os.path.join(self.save_dir, "episodic_memory_labels", basename + ".pt")
                torch.save(batch_episodes_labels[i], label_path)
                new_labels_path.append(label_path)
                # Actions
                action_path = os.path.join(self.save_dir, "episodic_memory_actions", basename + ".pkl")
                with open(action_path, 'wb') as f:
                    torch.save(batch_episodes_actions[i], f)
                new_actions_path.append(action_path)

                next_idx += 1

        return new_tensors_path, new_labels_path, new_actions_path
    
    def NREM_sleep(self,
                   view_encoder,
                   classifier,
                   cond_generator,
                   optimizers,
                   schedulers,
                   criterions,
                   task_id,
                   scaler,
                   writer,
                   is_main,
                   ddp):
        '''
        Train conditional generator and classifier using stored data in episodic memory
        '''

        # freeze view encoder
        view_encoder.eval()
        for param in view_encoder.parameters():
            param.requires_grad = False
        # unfreeze classifier
        classifier.train()
        for param in classifier.parameters():
            param.requires_grad = True
        # unfreeze conditional generator
        cond_generator.train() 
        for param in cond_generator.parameters():
            param.requires_grad = True

        # optimizers, schedulers, and criterions
        _, optimizer_classifier, optimizer_condgen = optimizers
        _, scheduler_classifier, scheduler_condgen = schedulers
        criterion_sup, criterion_condgen = criterions

        # Loss Plateau Detector
        loss_plateau_detector = SlopePlateauDetector(window_size=self.window, patience=self.patience, 
                                                     slope_threshold=self.threshold_nrem, use_smooth_loss=True, 
                                                     smooth_loss_alpha=self.smooth_loss_alpha)
        
        loss_condgen_log = MetricLogger("Loss CondGen")
        lossgen_1_log = MetricLogger("Loss Gen_1 FT tensor")
        lossgen_2_log = MetricLogger("Loss Gen_2 DecEnc tensor")
        lossgen_3_log = MetricLogger("Loss Gen_3 DecEnc direct tensor")
        loss_sup_log = MetricLogger("Loss Sup")
        acc1_log = MetricLogger("Acc@1")
        acc5_log = MetricLogger("Acc@5")

        # Sample episodes idxes to use for training
        sampled_episodes_idxs = self.replay_sampling_uniform(self.num_episodes_per_sleep)

        # Train model a mini-bacth at a time
        batch_idx=0
        while self.sleep_episode_counter < self.num_episodes_per_sleep/2: # first half in NREM
            #### --- Load episodes --- ####
            batch_episodes_idxs = sampled_episodes_idxs[batch_idx*self.episode_batch_size:(batch_idx+1)*self.episode_batch_size]
            tensors_loaded = []
            labels_loaded = []
            actions_loaded = []
            for i in batch_episodes_idxs:
                tensor_ = torch.tensor(np.load(self.episodic_memory_tensors[i]))  # Load tensors
                label_ = torch.load(self.episodic_memory_labels[i])  # Load labels
                with open(self.episodic_memory_actions[i], 'rb') as f:
                    action_ = torch.load(f)  # Load actions
                tensors_loaded.append(tensor_)
                labels_loaded.append(label_)
                actions_loaded.append(action_)
            batch_episodes_tensor = torch.stack(tensors_loaded, dim=0).to(self.device)  # (B, V, T, D)
            batch_episodes_labels = torch.stack(labels_loaded, dim=0).to(self.device)  # (B, V)
            batch_episodes_actions = actions_loaded  # list of length B, each element is a list of actions for each view

            #### --- Forward pass --- ####
            B, V, T, D = batch_episodes_tensor.shape
            with (autocast()):
                # Get flat versions of tensors, labels, and actions
                flat_tensors = batch_episodes_tensor.view(B*V, T, D) # (B*V, T, D)
                flat_labels = batch_episodes_labels.view(B*V) # (B*V)
                flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V
                
                # Get feats and cls
                flat_feats = flat_tensors[:, 1:, :]  # (B*V, T-1, D) Exclude cls token
                flat_cls = flat_tensors[:, 0, :]  # (B*V, D) Get cls token from all views

                #--- Conditional Generator ---#
                ## Conditional Generator forward pass
                first_view_feats = batch_episodes_tensor[:, 0, 1:, :]  # (B, T-1, D) Get first views and exclude cls token
                flat_first_feats = first_view_feats.unsqueeze(1)  # (B, 1,  T-1, D)
                flat_first_feats = flat_first_feats.expand(-1, V, -1, -1) # (B, V, T-1, D)
                flat_first_feats = flat_first_feats.reshape(B * V, T-1, D)   # (B*V, T-1, D) Expand first views
                flat_gen_imgs, flat_ftn_gen_feats = cond_generator(flat_first_feats, flat_actions) # (B*V, C, H, W), (B*V, T-1, D)
                flat_dec_gen_feats = view_encoder(flat_gen_imgs)[:, 1:, :]  # (B*V, T-1, D)
                ## Conditional Generator direct forward pass
                flat_dir_gen_imgs = cond_generator(flat_feats, None, skip_conditioning=True)  # (B*V, C, H, W)
                flat_gen_dir_feats = view_encoder(flat_dir_gen_imgs)[:, 1:, :]                  # (B*V, T-1, D)
                ## CondGen losses
                lossgen_1 = criterion_condgen(flat_ftn_gen_feats, flat_feats)  # FT tensor loss
                lossgen_2 = criterion_condgen(flat_dec_gen_feats, flat_feats)  # DecEnc tensor loss
                lossgen_3 = criterion_condgen(flat_gen_dir_feats, flat_feats)  # DecEnc direct tensor loss
                loss_condgen = lossgen_1 + lossgen_2 + lossgen_3  # Total CondGen loss

                #--- Classifier ---#
                ## Classifier forward pass (ignore v=0) ##
                mask = torch.ones(B, V, dtype=torch.bool, device=self.device)
                mask[:, 0] = False                                                                     # zero out first view
                flat_mask = mask.reshape(-1)                                                           # (B*V,)
                sup_logits = classifier(flat_cls[flat_mask])                                           # (B*(V-1), num_classes)
                sup_labels = flat_labels[flat_mask]                                                    # (B*(V-1),)
                ## Classifier losses and accuracy
                loss_sup  = criterion_sup(sup_logits, sup_labels)
                acc1, acc5 = accuracy(sup_logits, sup_labels, topk=(1, 5))

                #--- Total Loss ---#
                loss = loss_condgen + loss_sup  # Total loss

            #### --- Backward Pass --- ####
            optimizer_condgen.zero_grad()
            optimizer_classifier.zero_grad()
    
            scaler.scale(loss).backward()

            # Sanitycheck: check that gradients for view encoder are zeros or None (since it is frozen)
            for name, param in view_encoder.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)

            # Clip gradients and perform step for conditional generator
            scaler.unscale_(optimizer_condgen)
            torch.nn.utils.clip_grad_norm_(cond_generator.parameters(), 1.0)  # Clip gradients for conditional generator
            scaler.step(optimizer_condgen)

            # Clip gradients and perform step for classifier
            scaler.unscale_(optimizer_classifier)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)  # Clip gradients for classifier
            scaler.step(optimizer_classifier)

            # Update scaler and schedulers
            scaler.update()
            scheduler_condgen.step()
            scheduler_classifier.step()
            
            #### --- Update sleep counter --- ####
            self.sleep_episode_counter += self.episode_batch_size

            ### --- Update total number of seen episodes --- ###
            self.total_num_seen_episodes += self.episode_batch_size

            #### --- Print metrics --- ####
            if ddp:
                lossgen_1_reduced = reduce_tensor(torch.tensor(lossgen_1.item(), device=self.device), mean=True).item()
                lossgen_2_reduced = reduce_tensor(torch.tensor(lossgen_2.item(), device=self.device), mean=True).item()
                lossgen_3_reduced = reduce_tensor(torch.tensor(lossgen_3.item(), device=self.device), mean=True).item()
                loss_condgen_reduced = reduce_tensor(torch.tensor(loss_condgen.item(), device=self.device), mean=True).item()
                loss_sup_reduced = reduce_tensor(torch.tensor(loss_sup.item(), device=self.device), mean=True).item()
                acc1_reduced = reduce_tensor(torch.tensor(acc1.item(), device=self.device), mean=True).item()
                acc5_reduced = reduce_tensor(torch.tensor(acc5.item(), device=self.device), mean=True).item()
                sleep_episode_counter_reduced = reduce_tensor(torch.tensor(self.sleep_episode_counter, device=self.device), mean=False).item()
                num_episodes_per_sleep_reduced = self.num_episodes_per_sleep*torch.distributed.get_world_size()
                total_batch_size = self.episode_batch_size * torch.distributed.get_world_size()
                total_num_seen_episodes_reduced = self.total_num_seen_episodes * torch.distributed.get_world_size()
            else:
                lossgen_1_reduced = lossgen_1.item()
                lossgen_2_reduced = lossgen_2.item()
                lossgen_3_reduced = lossgen_3.item()
                loss_condgen_reduced = loss_condgen.item()
                loss_sup_reduced = loss_sup.item()
                acc1_reduced = acc1.item()
                acc5_reduced = acc5.item()
                sleep_episode_counter_reduced = self.sleep_episode_counter
                num_episodes_per_sleep_reduced = self.num_episodes_per_sleep
                total_batch_size = self.episode_batch_size
                total_num_seen_episodes_reduced = self.total_num_seen_episodes

            if is_main:
                if batch_idx % self.print_freq == 0 :
                    print(f'Episode [{sleep_episode_counter_reduced}/{num_episodes_per_sleep_reduced}]' +
                        f' -- NREM' +
                        f' -- Loss Gen_1: {lossgen_1_reduced:.6f}' +
                        f' -- Loss Gen_2: {lossgen_2_reduced:.6f}' +
                        f' -- Loss Gen_3: {lossgen_3_reduced:.6f}' +
                        f' -- Loss CondGen: {loss_condgen_reduced:.6f}' +
                        f' -- Loss Sup: {loss_sup_reduced:.6f}' +
                        f' -- Acc@1: {acc1_reduced:.2f}' +
                        f' -- Acc@5: {acc5_reduced:.2f}'
                        )
            batch_idx += 1

            ### --- Log metrics --- ###
            loss_condgen_log.update(loss_condgen_reduced, total_batch_size)
            lossgen_1_log.update(lossgen_1_reduced, total_batch_size)
            lossgen_2_log.update(lossgen_2_reduced, total_batch_size)
            lossgen_3_log.update(lossgen_3_reduced, total_batch_size)
            loss_sup_log.update(loss_sup_reduced, total_batch_size)
            acc1_log.update(acc1_reduced, total_batch_size)
            acc5_log.update(acc5_reduced, total_batch_size)

            ### --- Write metrics per batch for NREM --- ###
            if is_main:
                writer.add_scalar('NREM_REM_indicator_per batch', self.nrem_indicator,total_num_seen_episodes_reduced)
                writer.add_scalar('Loss_CondGen_per_batch', loss_condgen_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Loss_Gen_1_per_batch', lossgen_1_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Loss_Gen_2_per_batch', lossgen_2_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Loss_Gen_3_per_batch', lossgen_3_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Loss_Sup_per_batch', loss_sup_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Acc@1_per_batch', acc1_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Acc@5_per_batch', acc5_reduced, total_num_seen_episodes_reduced)
                # write learning rate per batch
                writer.add_scalar('CondGen_LR_per_batch', scheduler_condgen.get_last_lr()[0], total_num_seen_episodes_reduced)
                writer.add_scalar('Classifier_LR_per_batch', scheduler_classifier.get_last_lr()[0], total_num_seen_episodes_reduced)
        
        ### --- Write metrics accumulated after NREM --- ###
        if is_main:
            writer.add_scalar('NREM_REM_indicator', self.nrem_indicator, total_num_seen_episodes_reduced)
            writer.add_scalar('Loss_CondGen', loss_condgen_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Loss_Gen_1', lossgen_1_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Loss_Gen_2', lossgen_2_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Loss_Gen_3', lossgen_3_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Loss_Sup', loss_sup_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Acc@1', acc1_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Acc@5', acc5_log.avg, total_num_seen_episodes_reduced)

        # Print accumulated metrics at the end of NREM
        if is_main:
            print(f'Training metrics -- ' +
                  f'Loss Gen_1: {lossgen_1_log.avg:.6f}, ' +
                  f'Loss Gen_2: {lossgen_2_log.avg:.6f}, ' +
                  f'Loss Gen_3: {lossgen_3_log.avg:.6f}, ' +
                  f'Loss CondGen: {loss_condgen_log.avg:.6f}, ' +
                  f'Loss Sup: {loss_sup_log.avg:.6f}, ' +
                  f'Acc@1: {acc1_log.avg:.2f}, ' +
                  f'Acc@5: {acc5_log.avg:.2f}'
                  )
            
        # Reset optimizers
        optimizer_condgen.zero_grad()
        optimizer_classifier.zero_grad()

        return None
    
    def REM_sleep(self,
                  view_encoder,
                  classifier,
                  cond_generator,
                  optimizers,
                  schedulers,
                  criterions,
                  task_id,
                  scaler,
                  writer,
                  is_main,
                  ddp):
        '''
        Train view encoder using generated input episodes
        '''
       
        # unfreeze view encoder
        view_encoder.train()
        for param in view_encoder.parameters():
            param.requires_grad = True
        # freeze classifier
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        # freeze conditional generator
        cond_generator.eval() 
        for param in cond_generator.parameters():
            param.requires_grad = False

        # optimizers, schedulers, and criterions
        optimizer_encoder, _, _ = optimizers
        scheduler_encoder, _, _ = schedulers
        criterion_sup, _ = criterions

        # Loss Plateau Detector
        loss_plateau_detector = SlopePlateauDetector(window_size=self.window, patience=self.patience, 
                                                     slope_threshold=self.threshold_nrem, use_smooth_loss=True, 
                                                     smooth_loss_alpha=self.smooth_loss_alpha)
        
        loss_sup_log = MetricLogger("Loss_Sup")
        acc1_log = MetricLogger("Acc@1")
        acc5_log = MetricLogger("Acc@5")

        # Sample episodes idxes to use for training
        sampled_episodes_idxs = self.replay_sampling_uniform(self.num_episodes_per_sleep)

        # Train model a mini-bacth at a time
        batch_idx=0
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Load episodes --- ####
            batch_episodes_idxs = sampled_episodes_idxs[batch_idx*self.episode_batch_size:(batch_idx+1)*self.episode_batch_size]
            tensors_loaded = []
            labels_loaded = []
            actions_loaded = []
            for i in batch_episodes_idxs:
                tensor_ = torch.tensor(np.load(self.episodic_memory_tensors[i]))  # Load tensors
                label_ = torch.load(self.episodic_memory_labels[i])  # Load labels
                with open(self.episodic_memory_actions[i], 'rb') as f:
                    action_ = torch.load(f)  # Load actions
                tensors_loaded.append(tensor_)
                labels_loaded.append(label_)
                actions_loaded.append(action_)
            batch_episodes_tensor = torch.stack(tensors_loaded, dim=0).to(self.device)  # (B, V, T, D)
            batch_episodes_labels = torch.stack(labels_loaded, dim=0).to(self.device)  # (B, V)
            batch_episodes_actions = actions_loaded  # list of length B, each element is a list of actions for each view

            # Random policy for action sampling (randomly shuffle the actions so we use actions from a different episode on another one)
            random.shuffle(batch_episodes_actions)  # Shuffle actions in the mini-batch

            #### --- Forward pass --- ####
            B, V, T, D = batch_episodes_tensor.shape
            
            # Get flat versions labels and actions
            flat_labels = batch_episodes_labels.view(B*V) # (B*V)
            flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V
            
            # Get flat version of first view tensors
            first_view_feats = batch_episodes_tensor[:, 0, 1:, :]  # (B, T-1, D) Get first views and exclude cls token
            flat_first_feats = first_view_feats.unsqueeze(1)  # (B, 1,  T-1, D)
            flat_first_feats = flat_first_feats.expand(-1, V, -1, -1) # (B, V, T-1, D)
            flat_first_feats = flat_first_feats.reshape(B * V, T-1, D)   # (B*V, T-1, D) Expand first views

            # Generate inputs with conditional generator
            flat_gen_imgs, _ = cond_generator(flat_first_feats, flat_actions) # (B*V, C, H, W)

            with (autocast()):
                #--- View Encoder and Classifier---#
                ## View Encoder forward pass
                flat_out_tensors = view_encoder(flat_gen_imgs)  # (B*V, T, D)
                ## Classifier forward pass (ignore v=0) ##
                flat_cls = flat_out_tensors[:, 0, :]  # (B*V, D) Get cls token from all views
                mask = torch.ones(B, V, dtype=torch.bool, device=self.device)                          # Create mask for views (B, V)
                mask[:, 0] = False                                                                     # zero out first view
                flat_mask = mask.reshape(-1)                                                           # (B*V)
                sup_logits = classifier(flat_cls[flat_mask])                                           # (B*(V-1), num_classes)
                sup_labels = flat_labels[flat_mask]                                                    # (B*(V-1),)
                ## Classifier losses and accuracy
                loss_sup  = criterion_sup(sup_logits, sup_labels)
                acc1, acc5 = accuracy(sup_logits, sup_labels, topk=(1, 5))

                #--- Total Loss ---#
                loss = loss_sup  # Total loss (only classifier loss in REM phase)

            #### --- Backward Pass --- ####
            optimizer_encoder.zero_grad()
    
            scaler.scale(loss).backward()

            # Sanitycheck: check that gradients for classifier are zeros or None (since it is frozen)
            for name, param in classifier.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)
            # Sanitycheck: check that gradients for conditional generator are zeros or None (since it is frozen)
            for name, param in cond_generator.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)

            # Clip gradients and perform step for conditional generator
            scaler.unscale_(optimizer_encoder)
            torch.nn.utils.clip_grad_norm_(view_encoder.parameters(), 1.0)  # Clip gradients for view encoder
            scaler.step(optimizer_encoder)

            # Update scaler and schedulers
            scaler.update()
            scheduler_encoder.step()
            
            #### --- Update sleep counter --- ####
            self.sleep_episode_counter += self.episode_batch_size

            ### --- Update total number of seen episodes --- ###
            self.total_num_seen_episodes += self.episode_batch_size

            #### --- Print metrics --- ####
            if ddp:
                loss_sup_reduced = reduce_tensor(torch.tensor(loss_sup.item(), device=self.device), mean=True).item()
                acc1_reduced = reduce_tensor(torch.tensor(acc1.item(), device=self.device), mean=True).item()
                acc5_reduced = reduce_tensor(torch.tensor(acc5.item(), device=self.device), mean=True).item()
                sleep_episode_counter_reduced = reduce_tensor(torch.tensor(self.sleep_episode_counter, device=self.device), mean=False).item()
                num_episodes_per_sleep_reduced = self.num_episodes_per_sleep*torch.distributed.get_world_size()
                total_batch_size = self.episode_batch_size * torch.distributed.get_world_size()
                total_num_seen_episodes_reduced = self.total_num_seen_episodes * torch.distributed.get_world_size()
            else:
                loss_sup_reduced = loss_sup.item()
                acc1_reduced = acc1.item()
                acc5_reduced = acc5.item()
                sleep_episode_counter_reduced = self.sleep_episode_counter
                num_episodes_per_sleep_reduced = self.num_episodes_per_sleep
                total_batch_size = self.episode_batch_size
                total_num_seen_episodes_reduced = self.total_num_seen_episodes

            if is_main:
                if batch_idx % self.print_freq == 0 :
                    print(f'Episode [{sleep_episode_counter_reduced}/{num_episodes_per_sleep_reduced}]' +
                        f' -- REM' +
                        f' -- Loss Sup: {loss_sup_reduced:.6f}' +
                        f' -- Acc@1: {acc1_reduced:.2f}' +
                        f' -- Acc@5: {acc5_reduced:.2f}'
                        )
            batch_idx += 1
            
            ### --- Log metrics --- ###
            loss_sup_log.update(loss_sup_reduced, total_batch_size)
            acc1_log.update(acc1_reduced, total_batch_size)
            acc5_log.update(acc5_reduced, total_batch_size)

            #### --- Write metrics per batch for REM --- ###
            if is_main:
                writer.add_scalar('NREM_REM_indicator_per_batch', self.rem_indicator, total_num_seen_episodes_reduced)
                writer.add_scalar('Loss_Sup_per_batch', loss_sup_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Acc@1_per_batch', acc1_reduced, total_num_seen_episodes_reduced)
                writer.add_scalar('Acc@5_per_batch', acc5_reduced, total_num_seen_episodes_reduced)
                # write learning rate per batch
                writer.add_scalar('Encoder_LR_per_batch', scheduler_encoder.get_last_lr()[0], total_num_seen_episodes_reduced)
        
        ### --- Write accumulated metrics after REM --- ###
        if is_main:
            writer.add_scalar('NREM_REM_indicator', self.rem_indicator, total_num_seen_episodes_reduced)
            writer.add_scalar('Loss_Sup', loss_sup_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Acc@1', acc1_log.avg, total_num_seen_episodes_reduced)
            writer.add_scalar('Acc@5', acc5_log.avg, total_num_seen_episodes_reduced)
        
        # Print accumulated metrics at the end of REM
        if is_main:
            print(f'Training metrics -- ' +
                  f'Loss_Sup: {loss_sup_log.avg:.6f}, ' +
                  f'Acc@1: {acc1_log.avg:.2f}, ' +
                  f'Acc@5: {acc5_log.avg:.2f}'
                  )
                
        # Reset optimizers
        optimizer_encoder.zero_grad()

        return None
    
    def recalculate_episodic_memory_with_gen_imgs(self,
                                                  view_encoder, 
                                                  cond_generator):
        '''
        Recalculate episodic memory with generated images from the conditional generator.
        This is used to update the episodic memory with the generated images after the REM phase.
        '''

        # freeze view encoder
        view_encoder.eval()
        cond_generator.eval()

        for start in tqdm(range(0, len(self.episodic_memory_tensors), self.episode_batch_size), desc="Recal. Tensors"):
            ### Load mini-bacth tensors and actions
            end = min(start + self.episode_batch_size, len(self.episodic_memory_tensors))
            batch_episodes_tensor_paths = self.episodic_memory_tensors[start:end]
            batch_episodes_actions_paths = self.episodic_memory_actions[start:end]
            batch_episodes_tensor = torch.stack([torch.tensor(np.load(p)) for p in batch_episodes_tensor_paths], dim=0).to(self.device) # (B, V, T, D)
            batch_episodes_actions = []
            for p in batch_episodes_actions_paths:
                with open(p, 'rb') as f:
                    batch_episodes_actions.append(torch.load(f))

            ### Forward pass to get updated tensors            
            B, V, T, D = batch_episodes_tensor.shape
            batch_first_view_tensors = batch_episodes_tensor[:, 0, 1:, :] # (B, T-1, D) Get first views and exclude cls token
            flat_first_view_tensors = batch_first_view_tensors.unsqueeze(1)  # (B, 1, T-1, D)
            # Get first view tensors
            batch_first_view_tensors = batch_episodes_tensor[:, 0, 1:, :]  # (B, T-1, D) Get first views and exclude cls token
            flat_first_view_tensors = batch_first_view_tensors.unsqueeze(1)  # (B, 1, T-1, D)
            flat_first_view_tensors = flat_first_view_tensors.expand(-1, V, -1, -1)  # (B, V, T-1, D)
            flat_first_view_tensors = flat_first_view_tensors.reshape(B * V, T-1, D)  # (B*V, T-1, D) Expand first views
            # Get flat actions
            flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V
            # Get updated tensors
            with torch.no_grad():
                flat_gen_imgs, _ = cond_generator(flat_first_view_tensors, flat_actions)
                flat_updated_tensors = view_encoder(flat_gen_imgs)  # (B*V, T, D)
            batch_updated_episodes_tensor = flat_updated_tensors.view(B, V, T, D).cpu().detach()  # (B, V, T, D)

            ### Overwrite the same files on disk
            for i in range(B):
                np.save(batch_episodes_tensor_paths[i], batch_updated_episodes_tensor[i].numpy())  # Save updated tensors

        return None  


def eval_classification_performance(view_encoder, 
                                    classifier, 
                                    val_loader, 
                                    criterion, 
                                    writer, 
                                    dataset_name,
                                    ddp,
                                    is_main,
                                    device,
                                    total_num_seen_episodes):
    """
    Evaluate the classification performance of the model on the validation set.
    Note: Validation set has no view. It is just a single view center crop.
    """
    
    view_encoder.eval()
    classifier.eval()

    val_loss_sup_log = MetricLogger("Val_Loss_Sup")
    val_acc1_log = MetricLogger("Val_Acc@1")
    val_acc5_log = MetricLogger("Val_Acc@5")

    with torch.no_grad():
        for batch_imgs, batch_labels, _ in val_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            # View Encoder Forward pass
            tokens = view_encoder(batch_imgs)  # (B, T, D)
            cls_tokens = tokens[:, 0, :]
            # Classifier forward pass
            logits = classifier(cls_tokens)

            # Calculate loss and accuracy
            loss_sup = criterion(logits, batch_labels)
            acc1, acc5 = accuracy(logits, batch_labels, topk=(1, 5))

            # reduce metrics if using DDP
            if ddp:
                loss_sup = reduce_tensor(loss_sup, mean=True)
                acc1 = reduce_tensor(acc1, mean=True)
                acc5 = reduce_tensor(acc5, mean=True)
    
            # Update logs
            val_loss_sup_log.update(loss_sup.item(), batch_imgs.size(0))
            val_acc1_log.update(acc1.item(), batch_imgs.size(0))
            val_acc5_log.update(acc5.item(), batch_imgs.size(0))

    if ddp:
         total_num_seen_episodes_reduced = total_num_seen_episodes * torch.distributed.get_world_size()
    else:
        total_num_seen_episodes_reduced = total_num_seen_episodes


    # Print and log metrics
    if is_main:
        print(f'\t{dataset_name} -- ' +
              f'Loss_Sup: {val_loss_sup_log.avg:.6f}, '
              f'Acc@1: {val_acc1_log.avg:.2f}, '
              f'Acc@5: {val_acc5_log.avg:.2f}')
        # Log metrics
        writer.add_scalar(f'{dataset_name}_Loss_Sup', val_loss_sup_log.avg, total_num_seen_episodes_reduced)
        writer.add_scalar(f'{dataset_name}_Acc@1', val_acc1_log.avg, total_num_seen_episodes_reduced)
        writer.add_scalar(f'{dataset_name}_Acc@5', val_acc5_log.avg, total_num_seen_episodes_reduced)   

    return None