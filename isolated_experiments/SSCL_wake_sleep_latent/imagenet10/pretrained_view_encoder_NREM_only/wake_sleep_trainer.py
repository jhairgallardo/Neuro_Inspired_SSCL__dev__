import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler

from evaluate_cluster import evaluate as eval_pred

import einops
import numpy as np
from PIL import Image

class Wake_Sleep_trainer:
    def __init__(self, view_encoder, semantic_memory, view_generator, episode_batch_size):
        self.view_encoder = view_encoder
        self.semantic_memory = semantic_memory
        self.view_generator = view_generator
        self.episodic_memory = torch.empty(0)
        self.episode_batch_size = episode_batch_size
    
    def wake_phase(self, incoming_dataloader, device):
        ''' 
        Collect tensors in episodic memory
        '''
        self.view_encoder.eval() 
        aux_memory = {}
        for i, (episode_batch, _ , _ ) in enumerate(incoming_dataloader):
            b = episode_batch.size(0)
            v = episode_batch.size(1)
            # forward pass
            episode_batch = episode_batch.to(device)
            episode_batch = einops.rearrange(episode_batch, 'b v c h w -> (b v) c h w').contiguous() # all episodes and all views in one batch of (b v)
            with torch.no_grad(): episode_batch = self.view_encoder(episode_batch)
            episode_batch = einops.rearrange(episode_batch, '(b v) c h w -> b v c h w', b=b, v=v).contiguous() # back to batch of episodes
            aux_memory[i] = episode_batch.cpu()

        aux_memory = list(aux_memory.values())
        aux_memory = torch.cat(aux_memory, dim=0)
        self.episodic_memory = torch.cat([self.episodic_memory, aux_memory], dim=0)

        return None
    
    def sleep_phase_NREM(self, 
                        num_episodes_per_sleep, 
                        optimizer_semantic_memory, criterion_semantic_memory, scheduler_semantic_memory, 
                        optimizer_generator, criterion_generator, scheduler_generator, 
                        device, writer=None, task_id=None):
        '''
        Train model using episodic memory
        '''
        self.semantic_memory.train()
        self.view_generator.train()
        self.view_encoder.eval()

        # freeze view encoder
        for param in self.view_encoder.parameters():
            param.requires_grad = False
        # unfreeze generator
        for param in self.view_generator.parameters():
            param.requires_grad = True

        # Sample sleep episodes idxs from episodic_memory (Uniform sampling with replacement)
        weights = torch.ones(len(self.episodic_memory))
        sampled_episodes_idxs = list(WeightedRandomSampler(weights, num_episodes_per_sleep, replacement=True))

        # Train model on sleep episodes a bacth at a time
        for i in range(0, num_episodes_per_sleep, self.episode_batch_size):
            batch_idxs = sampled_episodes_idxs[i:i+self.episode_batch_size]
            batch_episodes = self.episodic_memory[batch_idxs]
            batch_embeddings = einops.rearrange(batch_episodes, 'b v c h w -> (b v) c h w').contiguous() # all episodes and all views in one batch of (b v)
            batch_embeddings = batch_embeddings.to(device)

            # Forward pass on semantic memory
            batch_logits = self.semantic_memory(batch_embeddings)

            # Forward pass on view generator
            batch_recon_imgs = self.view_generator(batch_embeddings)
            batch_recon_embs = self.view_encoder(batch_recon_imgs)

            # Backwad pass on semantic memory
            optimizer_semantic_memory.zero_grad()
            consis_loss, sharp_loss, div_loss = criterion_semantic_memory(batch_logits)
            semantic_loss = consis_loss + sharp_loss - div_loss
            semantic_loss.backward()
            optimizer_semantic_memory.step()
            scheduler_semantic_memory.step()

            # Backward pass on view generator
            optimizer_generator.zero_grad()
            recon_loss = criterion_generator(batch_recon_embs, batch_embeddings)
            recon_loss.backward()
            optimizer_generator.step()
            scheduler_generator.step()

            if i==0 or (i//self.episode_batch_size) % 10 == 0:
                current_episode_idx = min(i+self.episode_batch_size,num_episodes_per_sleep)
                print(f'Episode [{current_episode_idx}/{num_episodes_per_sleep}] ' +
                      f'-- SM lr: {scheduler_semantic_memory.get_last_lr()[0]:.6f} -- Consis: {consis_loss.item():.6f} -- Sharp: {sharp_loss.item():.6f} -- Div: {div_loss.item():.6f} -- SM Total loss: {semantic_loss.item():.6f} ' + 
                      f'-- VG lr: {scheduler_generator.get_last_lr()[0]:.6f} -- VG Loss: {recon_loss.item():.6f}')
                
                # print(f'Episode [{current_episode_idx}/{num_episodes_per_sleep}] ' +
                #       f'-- SM lr: {scheduler_semantic_memory.get_last_lr()[0]:.6f} -- Consis: {consis_loss.item():.6f} -- Sharp: {sharp_loss.item():.6f} -- Div: {div_loss.item():.6f} -- SM Total loss: {semantic_loss.item():.6f} ')
                
                # print(f'Episode [{current_episode_idx}/{num_episodes_per_sleep}] -- VG lr: {scheduler_generator.get_last_lr()[0]:.6f} -- VG Loss: {recon_loss.item():.6f}')
                
                if writer is not None and task_id is not None:
                    writer.add_scalar('Consistency Loss', consis_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Sharpness Loss', sharp_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Diversity Loss', div_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('SM Total Loss', semantic_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('VG Loss', recon_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)

        return None
    
    def evaluate_semantic_memory(self, val_loader, device, measure_cluster_acc=False,
                       plot_clusters=False, save_dir=None, task_id=None, mean=None, std=None):
        '''
        Evaluate semantic memory model on validation set
        '''
        self.semantic_memory.eval()
        self.view_encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(val_loader):
                images = images.to(device)
                logits = self.semantic_memory(self.view_encoder(images))
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_labels.append(targets)
        
        all_preds = torch.cat(all_preds).numpy()
        all_indices = np.arange(len(all_preds))
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        if measure_cluster_acc:
            nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(all_labels.astype(int), all_preds.astype(int), calc_acc=True, total_probs=all_probs)
            print(f'Metrics on all val data -- NMI: {nmi:.4f}, AMI: {ami:.4f}, ARI: {ari:.4f}, F: {fscore:.4f}, ACC: {adjacc:.4f}, ACC-Top5: {top5:.4f}')

        if plot_clusters:
            os.makedirs(save_dir, exist_ok=True)
            assert save_dir is not None
            assert task_id is not None
            assert mean is not None
            assert std is not None
            # plot 25 images per cluster
            for i in range(10): # Only for the first 10 pseudoclasses
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
                image_name = f'pseudoclass_{i}_taskid_{task_id}.png'
                grid.save(os.path.join(save_dir, image_name))

    def evaluate_generator(self, val_loader, device, save_dir=None, task_id=None, mean=None, std=None):
        '''
        Evaluate view generator model on validation set (plot original images and reconstructed images)
        '''
        self.view_generator.eval()
        self.view_encoder.eval()

        os.makedirs(save_dir, exist_ok=True)

        all_labels = []
        all_images = []
        with torch.no_grad():
            for i, (images, targets, _ ) in enumerate(val_loader):
                all_images.append(images)
                all_labels.append(targets)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels).numpy()

        num_imgs = 8
        for i in range(10): # Only for the first 10 classes
            class_imgs = all_images[all_labels==i][:num_imgs]
            class_recon_imgs = self.view_generator(self.view_encoder(class_imgs.to(device))).cpu()
            if len(class_imgs) > 0:
                class_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in class_imgs]
                class_recon_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in class_recon_imgs]
                class_imgs = torch.stack(class_imgs, dim=0)
                class_recon_imgs = torch.stack(class_recon_imgs, dim=0)
                grid = torchvision.utils.make_grid(torch.cat([class_imgs, class_recon_imgs], dim=0), nrow=num_imgs)
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
            else:
                grid = Image.new('RGB', (224*8, 224*2), (0, 0, 0))
            image_name = f'class_{i}_original_reconstructed_images_taskid_{task_id}.png'
            grid.save(os.path.join(save_dir, image_name))

        return None