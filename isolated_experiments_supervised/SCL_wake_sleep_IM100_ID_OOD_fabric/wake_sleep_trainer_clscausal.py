import os
import random

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler
from torch.amp import autocast

import numpy as np
from PIL import Image

from tqdm import tqdm
from utils import MetricLogger, accuracy, reduce_tensor

def make_token_weights_fractional(T: int, alpha: float, device, dtype=None):
    """
    Weighted average scheme:
      - uniform weight u = 1/T
      - first token gets w0 = alpha * u   (alpha < 1 => down-weight)
      - others share the remaining mass equally
    Returns w with sum(w) == 1, shape (T,)
    """
    u = 1.0 / T
    w0 = alpha * u
    if T == 1:
        return torch.tensor([1.0], device=device, dtype=dtype)
    w_rest = (1.0 - w0) / (T - 1)
    w = torch.full((T,), w_rest, device=device, dtype=dtype)
    w[0] = w0
    return w

class Wake_Sleep_trainer:
    def __init__(self,
                 episode_batch_size,
                 num_episodes_per_sleep,
                 num_views,
                 dataset_mean,
                 dataset_std,
                 save_dir,
                 print_freq=10
                 ):
        self.episode_batch_size = episode_batch_size
        self.num_episodes_per_sleep = num_episodes_per_sleep
        self.num_views = num_views

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.save_dir = save_dir
        self.print_freq = print_freq

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
        self.sampled_indices_budget = []  # Store sampled indices for logging

        self.plot_freq = 100 # Frequency to plot generated images
        self.batch_idx=0
    
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
        self.batch_idx = 0  # Reset batch index for the next sleep phase

        return None

    def replay_sampling_uniform(self, num_samples):
        """
        Sample a number of episodes uniformly from the episodic memory.
        """
        if len(self.episodic_memory_tensors) == 0:
            raise ValueError("Episodic memory is empty. Please collect data in the wake phase before sampling.")
        
        sampling_weights = torch.ones(len(self.episodic_memory_tensors)) # Uniform sampling across the whole episodic_memory
        sampled_indices = list(WeightedRandomSampler(sampling_weights, num_samples, replacement=True)) # We can repeat samples since num_samples>> len(self.episodic_memory_tensors)

        return sampled_indices
    
    def replay_sampling_uniform_class_balanced(self, num_samples):

        class_labels_per_episode = torch.tensor([torch.load(label_path)[0].item() for label_path in self.episodic_memory_labels])
        seen_classes = class_labels_per_episode.unique()
        num_seen_classes = len(seen_classes)

        # Create a list indicating the number of samples we will get for each class. It is balanced across classes.
        # If num_samples is not divisible by num_seen_classes, gives the remaining samples to a random class.
        num_samples_per_class = [num_samples // num_seen_classes] * num_seen_classes
        remaining_samples = num_samples % num_seen_classes
        if remaining_samples > 0:
            random_class = random.choice(range(num_seen_classes))
            num_samples_per_class[random_class] += remaining_samples

        # Sample indices for each class and concatenate them
        sampled_indices = []
        for class_label, num_samples in zip(seen_classes, num_samples_per_class):
            class_indices = torch.where(class_labels_per_episode == class_label)[0]
            if len(class_indices) == 0:
                continue
            # Sample uniformly from the class indices
            class_sampling_weights = torch.ones(len(class_indices))
            class_sampled_indices = list(WeightedRandomSampler(class_sampling_weights, num_samples, replacement=True))
            sampled_indices.extend(class_indices[class_sampled_indices].tolist())
        
        # Shuffle sampled indices to ensure randomness because we sampled them in class order
        random.shuffle(sampled_indices)

        return sampled_indices
    
    def replay_sampling_GRASP(self, num_samples):

        # Calculate each class mean tensor
        class_labels_per_episode = torch.tensor([torch.load(label_path)[0].item() for label_path in self.episodic_memory_labels])
        seen_classes = class_labels_per_episode.unique()
        class_means = {}
        for class_label in seen_classes:
            class_indices = torch.where(class_labels_per_episode == class_label)[0]
            class_clstokens = [torch.tensor(np.load(self.episodic_memory_tensors[i]))[:, 0, :] for i in class_indices]
            class_clstokens = torch.stack(class_clstokens) # (B, V, D)
            class_mean = class_clstokens.mean(dim=(0,1))  # Mean across all episodes and all views for this class (D,)
            class_means[class_label.item()] = class_mean

        # Calculate the distance of each episode to the class means
        # Do this by calculating the distance of each view cls token to the class mean, then take the mean distance across views
        distances = []
        for tensor_path, label_path in zip(self.episodic_memory_tensors, self.episodic_memory_labels):
            tensor = torch.tensor(np.load(tensor_path)) # (V, T, D)
            label = torch.load(label_path)[0].item()
            class_mean = class_means[label]
            cls_tokens = tensor[:, 0, :] # Get cls tokens from all views (V, D)
            # Calculate the cosine distance to the class mean for each view
            cos_similarities = F.cosine_similarity(cls_tokens, class_mean.unsqueeze(0), dim=1)  # (V,)
            cos_distances = 1 - cos_similarities  # Convert cosine similarity to distance (V,)
            views_mean_distance = cos_distances.mean().item()
            distances.append(views_mean_distance)
        distances = torch.tensor(distances).numpy()  # (num_episodes,)

        # Let's sample episodes indices
        count = 0
        sampled_indices = []
        while count < num_samples:
            classes_rand_list = list(range(len(seen_classes)))
            random.shuffle(classes_rand_list)  # Shuffle the classes to sample them in a random order
            for i in classes_rand_list:
                c = seen_classes[i].item() # class
                dist_class_c = distances[class_labels_per_episode == c]  # grabing distances of all samples of class c
                ixs_class_c = np.array(torch.where(class_labels_per_episode == c)[0])  # indices of all samples of class c
                probas = 1 / (dist_class_c + 1e-7) # min distances get highest scores/ priorities > easy examples
                p_class_c = probas / np.linalg.norm(probas, ord=1)  # sum to 1
                # sample
                sel_idx = np.random.choice(ixs_class_c, size=1, replace=False, p=p_class_c).item() # replace does not matter for 1 sample
                count += 1
                # make the distance of the sampled point the biggest one, so it won't be sampled again until all other points are sampled
                distances[sel_idx] += np.max(dist_class_c)
                # append the sampled index
                sampled_indices.append(sel_idx)
                if count >= num_samples:
                    break
        
        assert len(sampled_indices) == num_samples, f"Sampled {len(sampled_indices)} indices, expected {num_samples}."

        # no shuffle here. We want to keep the easy to hard order of the samples

        return sampled_indices
    
    def sampling_idxs_for_sleep(self, num_samples, sampling_method="uniform"):
        """
        Sample a number of episodes for sleep phase.
        :param num_samples: Number of samples to sample.
        :param sampling_method: Sampling method to use. Options are "uniform", "uniform_class_balanced", "GRASP".
        :return: List of sampled indices.
        """
        if sampling_method == "uniform":
            sampled_indices = self.replay_sampling_uniform(num_samples)
        elif sampling_method == "uniform_class_balanced":
            sampled_indices = self.replay_sampling_uniform_class_balanced(num_samples)
        elif sampling_method == "GRASP":
            sampled_indices = self.replay_sampling_GRASP(num_samples)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        self.sampled_indices_budget = sampled_indices  # Store sampled indices for logging

        return None
    
    def wake_phase(self, fabric, view_encoder, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        view_encoder.eval()

        next_idx = len(self.episodic_memory_tensors)  # next filename index

        new_tensors_path = []
        new_labels_path = []
        new_actions_path = []

        for batch_episodes, batch_labels , _  in tqdm(incoming_dataloader, desc="Wake Phase"):
            batch_episodes_imgs = batch_episodes[0]
            batch_episodes_actions = batch_episodes[1]
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1))

            B, V, C, H, W = batch_episodes_imgs.shape
            # forward pass to get tensors
            with torch.no_grad():
                flat_episodes_imgs = batch_episodes_imgs.view(B*V, C, H, W) # (B*V, C, H, W)
                flat_episodes_tensors = view_encoder(flat_episodes_imgs) # (B*V, T, D)
                T = flat_episodes_tensors.size(1) # number of tokens (includes the cls token and feature tokens)
                D = flat_episodes_tensors.size(2) # feature dimension
                batch_episodes_tensors = flat_episodes_tensors.view(B, V, T, D).cpu().detach() # (B, V, T, D)

            # Save each episode individualy 
            for i in range(B):
                idx = next_idx
                basename = f"rank{fabric.local_rank}_{idx:08d}"
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

        # Append new adquired data to the episodic memory
        self.append_memory(new_tensors_path, new_labels_path, new_actions_path)

        return None
    
    def NREM_sleep(self,
                   fabric,
                   view_encoder,
                   classifier,
                   cond_generator,
                   optimizers,
                   schedulers,
                   criterions,
                   task_id,
                   mean,
                   std,
                   save_dir,
                   view_order,
                   clstype):
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
        
        loss_condgen_tracker = MetricLogger("Loss CondGen")
        lossgen_1_tracker = MetricLogger("Loss Gen_1 FT tensor")
        lossgen_2_tracker = MetricLogger("Loss Gen_2 DecEnc tensor")
        lossgen_3_tracker = MetricLogger("Loss Gen_3 DecEnc direct tensor")
        loss_sup_tracker = MetricLogger("Loss Sup")
        acc1_tracker = MetricLogger("Acc@1")
        acc5_tracker = MetricLogger("Acc@5")

        # Train model a mini-bacth at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep/2: # first half in NREM
            #### --- Load episodes --- ####
            batch_episodes_idxs = self.sampled_indices_budget[self.batch_idx*self.episode_batch_size:(self.batch_idx+1)*self.episode_batch_size]
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
            batch_episodes_tensor = torch.stack(tensors_loaded, dim=0).to(fabric.device)  # (B, V, T, D)
            batch_episodes_labels = torch.stack(labels_loaded, dim=0).to(fabric.device)  # (B, V)
            batch_episodes_actions = actions_loaded  # list of length B, each element is a list of actions for each view

            #### --- Forward pass --- ####
            B, V, T, D = batch_episodes_tensor.shape
            # Get flat versions of tensors, labels, and actions
            flat_tensors = batch_episodes_tensor.view(B*V, T, D) # (B*V, T, D)
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
            flat_gen_dec_feats_and_cls = view_encoder(flat_gen_imgs)
            flat_dec_gen_feats = flat_gen_dec_feats_and_cls[:, 1:, :]  # (B*V, T, D) # Discard the CLS token
            flat_dec_gen_cls = flat_gen_dec_feats_and_cls[:, 0, :].detach()  # (B*V, D) # CLS token for generated images
            ## Conditional Generator direct forward pass
            flat_dir_gen_imgs = cond_generator(flat_feats, None, skip_conditioning=True)  # (B*V, C, H, W)
            flat_gen_dir_feats = view_encoder(flat_dir_gen_imgs)[:, 1:, :]                  # (B*V, T-1, D)

            with fabric.autocast():
                ## CondGen losses
                lossgen_1 = criterion_condgen(flat_ftn_gen_feats, flat_feats)  # FT tensor loss
                lossgen_2 = criterion_condgen(flat_dec_gen_feats, flat_feats)  # DecEnc tensor loss
                lossgen_3 = criterion_condgen(flat_gen_dir_feats, flat_feats)  # DecEnc direct tensor loss
                loss_condgen = lossgen_1 + lossgen_2 + lossgen_3  # Total CondGen loss

            #--- Classifier Causal ---#
            if clstype=='storedcls':
                notflat_cls = flat_cls.reshape(B, V, D)  # (B, V, D) Reshape to (B, V, D)
            elif clstype=='gencls':
                notflat_cls = flat_dec_gen_cls.reshape(B, V, D)  # (B, V, D) Reshape to (B, V, D)
            else:
                raise ValueError(f"Unknown cls type: {clstype}. Choose from 'storedcls' or 'gencls'.")

            if view_order=="ori":
                notflat_cls=notflat_cls
            elif view_order=="rev":
                notflat_cls = notflat_cls.flip(dims=[1])
            elif view_order=='rev50':
                # 50% of the time, reverse the views
                if torch.rand(1) < 0.5:
                    notflat_cls = notflat_cls.flip(dims=[1])
            elif view_order=="rand":
                perms = torch.argsort(torch.rand(B, notflat_cls.size(1), device=notflat_cls.device), dim=1)
                batch_idx = torch.arange(B, device=notflat_cls.device).unsqueeze(1).expand(-1, notflat_cls.size(1))
                notflat_cls = notflat_cls[batch_idx, perms]
            elif view_order=="rand50":
                # 50% of the time, randomize the views
                if torch.rand(1) < 0.5:
                    perms = torch.argsort(torch.rand(B, notflat_cls.size(1), device=notflat_cls.device), dim=1)
                    batch_idx = torch.arange(B, device=notflat_cls.device).unsqueeze(1).expand(-1, notflat_cls.size(1))
                    notflat_cls = notflat_cls[batch_idx, perms]
            else:
                raise ValueError(f"Unknown view order: {view_order}. Choose from 'original', 'reverse', or 'random'.")

            ## Classifier forward pass (use all views) ##
            sup_logits = classifier(notflat_cls).reshape(B * V, -1) # (B*V, num_classes)
            sup_labels = batch_episodes_labels.reshape(-1)     

            # Calculate losses and accuracy
            with fabric.autocast():
                ## Classifier losses and accuracy
                loss_sup  = criterion_sup(sup_logits, sup_labels).mean()
                acc1, acc5 = accuracy(sup_logits, sup_labels, topk=(1, 5))

                #--- Total Loss ---#
                loss = loss_condgen + loss_sup  # Total loss

            #### --- Backward Pass --- ####
            optimizer_condgen.zero_grad()
            optimizer_classifier.zero_grad()
            fabric.backward(loss)

            # Sanitycheck: check that gradients for view encoder are zeros or None (since it is frozen)
            for name, param in view_encoder.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)

            # Clip gradients and perform step for conditional generator
            fabric.clip_gradients(cond_generator, optimizer_condgen, max_norm=1.0)
            optimizer_condgen.step()

            # Clip gradients and perform step for classifier
            fabric.clip_gradients(classifier, optimizer_classifier, max_norm=1.0)
            optimizer_classifier.step()

            # Update schedulers
            scheduler_condgen.step()
            scheduler_classifier.step()
            
            #### --- Update sleep counter --- ####
            self.sleep_episode_counter += self.episode_batch_size

            ### --- Update total number of seen episodes --- ###
            self.total_num_seen_episodes += self.episode_batch_size

            #### --- Track metrics --- ####
            episode_batch_size_allranks = fabric.all_reduce(torch.tensor(self.episode_batch_size, device=fabric.device), reduce_op="sum").item()
            loss_condgen_tracker.update(fabric.all_reduce(loss_condgen.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            lossgen_1_tracker.update(fabric.all_reduce(lossgen_1.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            lossgen_2_tracker.update(fabric.all_reduce(lossgen_2.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            lossgen_3_tracker.update(fabric.all_reduce(lossgen_3.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            loss_sup_tracker.update(fabric.all_reduce(loss_sup.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            acc1_tracker.update(fabric.all_reduce(acc1.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            acc5_tracker.update(fabric.all_reduce(acc5.detach(), reduce_op="mean").item(), episode_batch_size_allranks)

            ### --- Print metrics per batch--- ####
            if self.batch_idx % self.print_freq == 0 :
                sleep_episode_counter_allranks = fabric.all_reduce(torch.tensor(self.sleep_episode_counter, device=fabric.device), reduce_op="sum").item()
                num_episodes_per_sleep_allranks = fabric.all_reduce(torch.tensor(self.num_episodes_per_sleep, device=fabric.device), reduce_op="sum").item()
                fabric.print(f'Episode [{sleep_episode_counter_allranks}/{num_episodes_per_sleep_allranks}]' +
                            f' -- NREM' +
                            f' -- Loss Gen_1: {lossgen_1_tracker.val:.6f}' +
                            f' -- Loss Gen_2: {lossgen_2_tracker.val:.6f}' +
                            f' -- Loss Gen_3: {lossgen_3_tracker.val:.6f}' +
                            f' -- Loss CondGen: {loss_condgen_tracker.val:.6f}' +
                            f' -- Loss Sup: {loss_sup_tracker.val:.6f}' +
                            f' -- Acc@1: {acc1_tracker.val:.2f}' +
                            f' -- Acc@5: {acc5_tracker.val:.2f}'
                            )
            self.batch_idx += 1

            ### --- Log metrics per batch--- ###
            total_num_seen_episodes_allranks = fabric.all_reduce(torch.tensor(self.total_num_seen_episodes, device=fabric.device), reduce_op="sum").item()
            fabric.log(name=f'NREM_REM_indicator_per batch', value=self.nrem_indicator, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_CondGen_per batch', value=loss_condgen_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_Gen_1_per batch', value=lossgen_1_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_Gen_2_per batch', value=lossgen_2_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_Gen_3_per batch', value=lossgen_3_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_Sup_per batch', value=loss_sup_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Acc@1_per batch', value=acc1_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Acc@5_per batch', value=acc5_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'CondGen_LR_per batch', value=scheduler_condgen.get_last_lr()[0], step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Classifier_LR_per batch', value=scheduler_classifier.get_last_lr()[0], step=total_num_seen_episodes_allranks)

            ### --- Plot generated images --- ###
            if fabric.is_global_zero and self.batch_idx % self.plot_freq == 0:
                batch_episodes_gen_imgs = flat_gen_imgs.view(B, V, *flat_gen_imgs.shape[1:])  # (B, V, C, H, W)
                episode_i_gen_imgs = batch_episodes_gen_imgs[0]
                episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_i_gen_imgs]
                episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0)
                episode_i_gen_imgs = torch.clamp(episode_i_gen_imgs, 0, 1) # Clip values to [0, 1]
                grid = torchvision.utils.make_grid(episode_i_gen_imgs, nrow=episode_i_gen_imgs.shape[0])
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
                save_plot_dir = os.path.join(save_dir, 'generated_images_NREM', f'Learned_taskid_{task_id}')
                image_name = f'episode0_batch{self.batch_idx}.png'
                # create folder if it doesn't exist
                if not os.path.exists(save_plot_dir):
                    os.makedirs(save_plot_dir)
                grid.save(os.path.join(save_plot_dir, image_name))
        
        ### --- Log metrics at the end of every NREM sleep session --- ###
        total_num_seen_episodes_allranks = fabric.all_reduce(torch.tensor(self.total_num_seen_episodes, device=fabric.device), reduce_op="sum").item()
        fabric.log(name=f'NREM_REM_indicator', value=self.nrem_indicator, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_CondGen', value=loss_condgen_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_Gen_1', value=lossgen_1_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_Gen_2', value=lossgen_2_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_Gen_3', value=lossgen_3_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_Sup', value=loss_sup_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Acc@1', value=acc1_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Acc@5', value=acc5_tracker.avg, step=total_num_seen_episodes_allranks)

        # Print accumulated metrics at the end of NREM sleep session
        fabric.print(f'Training metrics -- ' +
                  f'Loss Gen_1: {lossgen_1_tracker.avg:.6f}, ' +
                  f'Loss Gen_2: {lossgen_2_tracker.avg:.6f}, ' +
                  f'Loss Gen_3: {lossgen_3_tracker.avg:.6f}, ' +
                  f'Loss CondGen: {loss_condgen_tracker.avg:.6f}, ' +
                  f'Loss Sup: {loss_sup_tracker.avg:.6f}, ' +
                  f'Acc@1: {acc1_tracker.avg:.2f}, ' +
                  f'Acc@5: {acc5_tracker.avg:.2f}'
                  )
            
        # Reset optimizers
        optimizer_condgen.zero_grad()
        optimizer_classifier.zero_grad()

        return None
    
    def REM_sleep(self,
                  fabric,
                  view_encoder,
                  classifier,
                  cond_generator,
                  optimizers,
                  schedulers,
                  criterions,
                  task_id,
                  mean,
                  std,
                  save_dir,
                  viewstouse,
                  lambda_negshannonent,
                  lambda_firstviewpenalty,
                  firstviewCEweight,
                  cls_firstviewdroprate
                  ):
        '''
        Train view encoder using generated input episodes
        '''
        original_plot_flag = True
       
        # unfreeze view encoder
        view_encoder.train()
        for param in view_encoder.parameters():
            param.requires_grad = True
        # freeze classifier
        # classifier.eval()
        classifier.train()
        for param in classifier.parameters():
            # param.requires_grad = False
            param.requires_grad = True
        # freeze conditional generator
        cond_generator.eval() 
        for param in cond_generator.parameters():
            param.requires_grad = False

        # optimizers, schedulers, and criterions
        optimizer_encoder, optimizer_classifier, optimizer_condgen = optimizers
        scheduler_encoder, scheduler_classifier, scheduler_condgen = schedulers
        criterion_sup, criterion_condgen, criterion_attn_div = criterions
        
        loss_sup_tracker = MetricLogger("Loss_Sup")
        loss_neg_shannon_entropy_tracker = MetricLogger("Loss_Neg_Shannon_Entropy")
        loss_first_key_penalty_tracker = MetricLogger("Loss_First_Key_Penalty")
        acc1_tracker = MetricLogger("Acc@1")
        acc5_tracker = MetricLogger("Acc@5")

        # Train model a mini-batch at a time
        while self.sleep_episode_counter < self.num_episodes_per_sleep:
            #### --- Load episodes --- ####
            batch_episodes_idxs = self.sampled_indices_budget[self.batch_idx*self.episode_batch_size:(self.batch_idx+1)*self.episode_batch_size]
            tensors_loaded = []
            labels_loaded = []
            original_actions_loaded = []
            for i in batch_episodes_idxs:
                tensor_ = torch.tensor(np.load(self.episodic_memory_tensors[i]))  # Load tensors
                label_ = torch.load(self.episodic_memory_labels[i])  # Load labels
                with open(self.episodic_memory_actions[i], 'rb') as f:
                    action_ = torch.load(f)  # Load actions
                tensors_loaded.append(tensor_)
                labels_loaded.append(label_)
                original_actions_loaded.append(action_)
            batch_episodes_tensor = torch.stack(tensors_loaded, dim=0).to(fabric.device)  # (B, V, T, D)
            batch_episodes_labels = torch.stack(labels_loaded, dim=0).to(fabric.device)  # (B, V)
            batch_episodes_original_actions = original_actions_loaded

            # Actions are randomly sampled from the episodic_memory_actions. (Sample a batch B of episodes actions)(Random policy for action sampling)
            random_indices = random.sample(range(len(self.episodic_memory_actions)), self.episode_batch_size)
            # make sure none of the batch_episodes_idxs are in the random_indices (to avoid using the same episode actions)
            while any(idx in random_indices for idx in batch_episodes_idxs):
                random_indices = random.sample(range(len(self.episodic_memory_actions)), self.episode_batch_size)
            batch_episodes_actions = [torch.load(self.episodic_memory_actions[i]) for i in random_indices]

            #### --- Forward pass --- ####
            B, V, T, D = batch_episodes_tensor.shape

            ## Get flat version of first view tensors
            first_view_feats = batch_episodes_tensor[:, 0, 1:, :]  # (B, T-1, D) Get first views and exclude cls token
            flat_first_feats = first_view_feats.unsqueeze(1)  # (B, 1,  T-1, D)
            flat_first_feats = flat_first_feats.expand(-1, V, -1, -1) # (B, V, T-1, D)
            flat_first_feats = flat_first_feats.reshape(B * V, T-1, D)   # (B*V, T-1, D) Expand first views

            ## Get Flat actions
            flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V

            #--- Conditional Generator forward pass ---#
            flat_gen_imgs, _ = cond_generator(flat_first_feats, flat_actions) # (B*V, C, H, W)

            #--- View Encoder forward pass ---#
            flat_feats_and_cls = view_encoder(flat_gen_imgs)  # (B*V, T, D)

            #--- Classifier foward pass ---#
            notflat_cls = flat_feats_and_cls[:, 0, :].view(B, V, D)  # Get cls tokens (B, V, D)
            if viewstouse=='nofirstview':
                notflat_cls = notflat_cls[:, 1:, :]  # (B, V-1, D) Exclude first view
                batch_episodes_labels = batch_episodes_labels[:, 1:]  # (B, V-1) Exclude first view labels
            elif viewstouse=='allviews':
                notflat_cls = notflat_cls
                batch_episodes_labels = batch_episodes_labels
            else:
                raise ValueError(f"Unknown first view usage: {viewstouse}. Choose from 'nofirstview' or 'allviews'.")
            notflat_logits = classifier(notflat_cls, first_token_droprate=cls_firstviewdroprate)     # (B, V, num_classes) or (B, V-1, num_classes)
            sup_logits = notflat_logits.reshape(B * notflat_logits.size(1), -1) # (B*V, num_classes) or (B*(V-1), num_classes)
            sup_labels = batch_episodes_labels.reshape(-1)

            with fabric.autocast():
                # Classifier losses and accuracy
                loss_sup  = criterion_sup(sup_logits, sup_labels)
                loss_sup = loss_sup.view(B, notflat_logits.size(1))
                w = make_token_weights_fractional(notflat_logits.size(1), alpha=firstviewCEweight, device=sup_logits.device, dtype=sup_logits.dtype)  # sum=1
                loss_sup = (loss_sup * w).sum(dim=1).mean()
                acc1, acc5 = accuracy(sup_logits, sup_labels, topk=(1, 5))

                # Attention diversification loss (compute it outside mixed precision)
                attn_probs = classifier.module.transf.layers[0].last_attn_probs
                neg_shannon_entropy_loss, firstkeypenalty_loss = criterion_attn_div(attn_probs)

                #--- Total Loss ---#
                loss = loss_sup + lambda_negshannonent * neg_shannon_entropy_loss + lambda_firstviewpenalty * firstkeypenalty_loss

            #### --- Backward Pass --- ####
            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            fabric.backward(loss)

            # Sanitycheck: check that gradients for conditional generator are zeros or None (since it is frozen)
            for name, param in cond_generator.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)

            # Clip gradients and perform step for conditional generator
            fabric.clip_gradients(view_encoder, optimizer_encoder, max_norm=1.0)
            optimizer_encoder.step()

            # Clip gradients and perform step for classifier
            fabric.clip_gradients(classifier, optimizer_classifier, max_norm=1.0)
            optimizer_classifier.step()

            # Update schedulers
            scheduler_encoder.step()
            scheduler_classifier.step()
            
            #### --- Update sleep counter --- ####
            self.sleep_episode_counter += self.episode_batch_size

            ### --- Update total number of seen episodes --- ###
            self.total_num_seen_episodes += self.episode_batch_size

            #### --- Track metrics --- ####
            episode_batch_size_allranks = fabric.all_reduce(torch.tensor(self.episode_batch_size, device=fabric.device), reduce_op="sum").item()
            loss_sup_tracker.update(fabric.all_reduce(loss_sup.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            loss_neg_shannon_entropy_tracker.update(fabric.all_reduce(neg_shannon_entropy_loss.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            loss_first_key_penalty_tracker.update(fabric.all_reduce(firstkeypenalty_loss.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            acc1_tracker.update(fabric.all_reduce(acc1.detach(), reduce_op="mean").item(), episode_batch_size_allranks)
            acc5_tracker.update(fabric.all_reduce(acc5.detach(), reduce_op="mean").item(), episode_batch_size_allranks)

            ### --- Print metrics per batch--- ####
            if self.batch_idx % self.print_freq == 0 :
                sleep_episode_counter_allranks = fabric.all_reduce(torch.tensor(self.sleep_episode_counter, device=fabric.device), reduce_op="sum").item()
                num_episodes_per_sleep_allranks = fabric.all_reduce(torch.tensor(self.num_episodes_per_sleep, device=fabric.device), reduce_op="sum").item()
                fabric.print(f'Episode [{sleep_episode_counter_allranks}/{num_episodes_per_sleep_allranks}]' +
                            f' -- REM' +
                            f' -- Loss Sup: {loss_sup_tracker.val:.6f}' +
                            f' -- Loss Neg Shannon Entropy: {loss_neg_shannon_entropy_tracker.val:.6f}' +
                            f' -- Loss First Key Penalty: {loss_first_key_penalty_tracker.val:.6f}' +
                            f' -- Acc@1: {acc1_tracker.val:.2f}' +
                            f' -- Acc@5: {acc5_tracker.val:.2f}'
                            )
            self.batch_idx += 1

            ### --- Log metrics per batch--- ###
            total_num_seen_episodes_allranks = fabric.all_reduce(torch.tensor(self.total_num_seen_episodes, device=fabric.device), reduce_op="sum").item()
            fabric.log(name=f'NREM_REM_indicator_per batch', value=self.rem_indicator, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_Sup_per batch', value=loss_sup_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_Neg_Shannon_Entropy_per batch', value=loss_neg_shannon_entropy_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Loss_First_Key_Penalty_per batch', value=loss_first_key_penalty_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Acc@1_per batch', value=acc1_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Acc@5_per batch', value=acc5_tracker.val, step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Encoder_LR_per batch', value=scheduler_encoder.get_last_lr()[0], step=total_num_seen_episodes_allranks)
            fabric.log(name=f'Classifier_LR_per batch', value=scheduler_classifier.get_last_lr()[0], step=total_num_seen_episodes_allranks)

            ### --- Plot generated images --- ###
            batch_episodes_gen_imgs = flat_gen_imgs.view(B, V, *flat_gen_imgs.shape[1:])  # (B, V, C, H, W)
            if fabric.is_global_zero and self.batch_idx % self.plot_freq == 0:
                episode_i_gen_imgs = batch_episodes_gen_imgs[0]
                episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_i_gen_imgs]
                episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0)
                episode_i_gen_imgs = torch.clamp(episode_i_gen_imgs, 0, 1) # Clip values to [0, 1]
                grid = torchvision.utils.make_grid(episode_i_gen_imgs, nrow=episode_i_gen_imgs.shape[0])
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
                save_plot_dir = os.path.join(save_dir, 'generated_images_REM', f'Learned_taskid_{task_id}')
                image_name = f'episode0_batch{self.batch_idx}_sampled.png'
                # create folder if it doesn't exist
                if not os.path.exists(save_plot_dir):
                    os.makedirs(save_plot_dir)
                grid.save(os.path.join(save_plot_dir, image_name))

            ### --- One time plot of original episodes --- ###
            if self.batch_idx % self.plot_freq == 0:
                if original_plot_flag:
                    with torch.no_grad():
                        original_plot_flag=False
                        flat_original_actions = [batch_episodes_original_actions[b][v] for b in range(B) for v in range(V)]
                        flat_original_gen_imgs, _ = cond_generator(flat_first_feats, flat_original_actions)
                        batch_episodes_original_gen_imgs = flat_original_gen_imgs.view(B, V, *flat_original_gen_imgs.shape[1:])  # (B, V, C, H, W)
                        episode_i_original_gen_imgs = batch_episodes_original_gen_imgs[0]
                        episode_i_original_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_i_original_gen_imgs]
                        episode_i_original_gen_imgs = torch.stack(episode_i_original_gen_imgs, dim=0)
                        episode_i_original_gen_imgs = torch.clamp(episode_i_original_gen_imgs, 0, 1) # Clip values to [0, 1]
                        if fabric.is_global_zero:
                            grid = torchvision.utils.make_grid(episode_i_original_gen_imgs, nrow=episode_i_original_gen_imgs.shape[0])
                            grid = grid.permute(1, 2, 0).cpu().numpy()
                            grid = (grid * 255).astype(np.uint8)
                            grid = Image.fromarray(grid)
                            save_plot_dir = os.path.join(save_dir, 'generated_images_REM', f'Learned_taskid_{task_id}')
                            image_name = f'episode0_batch{self.batch_idx}_original.png'
                            grid.save(os.path.join(save_plot_dir, image_name))
        
        ### --- Log metrics at the end of every REM sleep session --- ###
        total_num_seen_episodes_allranks = fabric.all_reduce(torch.tensor(self.total_num_seen_episodes, device=fabric.device), reduce_op="sum").item()
        fabric.log(name=f'NREM_REM_indicator', value=self.rem_indicator, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_Sup', value=loss_sup_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_Neg_Shannon_Entropy', value=loss_neg_shannon_entropy_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Loss_First_Key_Penalty', value=loss_first_key_penalty_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Acc@1', value=acc1_tracker.avg, step=total_num_seen_episodes_allranks)
        fabric.log(name=f'Acc@5', value=acc5_tracker.avg, step=total_num_seen_episodes_allranks)
        
        # Print accumulated metrics at the end of REM sleep session
        fabric.print(f'Training metrics -- ' +
                     f'Loss_Sup: {loss_sup_tracker.avg:.6f}, ' +
                     f'Loss_Neg_Shannon_Entropy: {loss_neg_shannon_entropy_tracker.avg:.6f}, ' +
                     f'Loss_First_Key_Penalty: {loss_first_key_penalty_tracker.avg:.6f}, ' +
                     f'Acc@1: {acc1_tracker.avg:.2f}, ' +
                     f'Acc@5: {acc5_tracker.avg:.2f}'
                     )
                
        # Reset optimizers
        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()

        return None
    
    def recalculate_episodic_memory_with_gen_imgs(self,
                                                  fabric,
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
            batch_episodes_tensor = torch.stack([torch.tensor(np.load(p)) for p in batch_episodes_tensor_paths], dim=0).to(fabric.device) # (B, V, T, D)
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

def eval_classification_performance(fabric,
                                    view_encoder, 
                                    classifier, 
                                    val_loader, 
                                    criterion, 
                                    dataset_type,
                                    total_num_seen_episodes):

    assert dataset_type in ['Val_ID', 'Val_OOD']
    view_encoder.eval()
    classifier.eval()

    val_loss_sup_tracker = MetricLogger("Val_Loss_Sup")
    val_acc1_tracker = MetricLogger("Val_Acc@1")
    val_acc5_tracker = MetricLogger("Val_Acc@5")
    with torch.no_grad():
        for batch_imgs, batch_labels, _ in val_loader:

            # View Encoder Forward pass
            tokens = view_encoder(batch_imgs)  # (B, T, D)
            cls_tokens = tokens[:, 0, :] # (B, D) Get cls token from all views
            # Classifier forward pass (causal transformer)
            cls_tokens = cls_tokens.unsqueeze(1) # (B, 1, D) Unsqueeze cls token to make it a sequence of 1 view
            logits = classifier(cls_tokens).squeeze(1)  # (B, num_classes)

            with fabric.autocast():
                # Calculate loss and accuracy
                loss_sup = criterion(logits, batch_labels).mean()
                acc1, acc5 = accuracy(logits, batch_labels, topk=(1, 5))
    
            # Track validation metrics
            val_loss_sup_tracker.update(fabric.all_reduce(loss_sup.detach(), reduce_op="mean").item(), batch_imgs.size(0))
            val_acc1_tracker.update(fabric.all_reduce(acc1.detach(), reduce_op="mean").item(), batch_imgs.size(0))
            val_acc5_tracker.update(fabric.all_reduce(acc5.detach(), reduce_op="mean").item(), batch_imgs.size(0))

    fabric.print(f'\t{dataset_type} -- ' +
                 f'Loss_Sup: {val_loss_sup_tracker.avg:.6f}, '
                 f'Acc@1: {val_acc1_tracker.avg:.2f}, '
                 f'Acc@5: {val_acc5_tracker.avg:.2f}')
    # Log validation metrics
    total_num_seen_episodes_allranks = fabric.all_reduce(torch.tensor(total_num_seen_episodes, device=fabric.device), reduce_op="sum").item()
    fabric.log(name=f'{dataset_type}_Loss_Sup', value=val_loss_sup_tracker.avg, step=total_num_seen_episodes_allranks)
    fabric.log(name=f'{dataset_type}_Acc@1', value=val_acc1_tracker.avg, step=total_num_seen_episodes_allranks)
    fabric.log(name=f'{dataset_type}_Acc@5', value=val_acc5_tracker.avg, step=total_num_seen_episodes_allranks) 

    return None
