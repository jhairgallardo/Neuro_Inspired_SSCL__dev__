import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import WeightedRandomSampler

from evaluate_cluster import evaluate as eval_pred
from utils import statistics, encode_label

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import einops

class Wake_Sleep_trainer:
    def __init__(self, view_encoder, semantic_memory, episode_batch_size, args=None):
        self.view_encoder = view_encoder
        self.semantic_memory = semantic_memory

        self.episodic_memory_tensors = torch.empty(0)
        self.episodic_memory_labels = torch.empty(0)

        self.episode_batch_size = episode_batch_size

        self.args = args
        self.tau_t = args.tau_t
        self.tau_s = args.tau_s
        self.beta = args.beta

        self.firstwake=True
    
    def wake_phase(self, incoming_dataloader):
        ''' 
        Collect data in episodic memory
        '''
        self.view_encoder.eval()

        aux_memory_tensors = {}
        aux_memory_labels = {}
        for i, (episode_batch, labels , _ ) in enumerate(tqdm(incoming_dataloader)):
            episode_batch_imgs = episode_batch
            episode_labels = labels.unsqueeze(1).repeat(1, episode_batch.size(1))

            # forwards pass to get tensors
            episode_batch_imgs = episode_batch_imgs.to(self.args.device)
            batch_tensors = torch.empty(0).to(self.args.device)
            with torch.no_grad():
                for v in range(self.args.num_views):
                    img_batch = episode_batch_imgs[:,v,:,:,:]
                    tensors_out = self.view_encoder(img_batch, before_avgpool=True)
                    batch_tensors = torch.cat([batch_tensors, tensors_out.unsqueeze(1)], dim=1)
            # collect episodes
            aux_memory_tensors[i] = batch_tensors.cpu()
            aux_memory_labels[i] = episode_labels

        # Concatenate all
        aux_memory_tensors = list(aux_memory_tensors.values())
        aux_memory_tensors = torch.cat(aux_memory_tensors, dim=0)
        self.episodic_memory_tensors = torch.cat([self.episodic_memory_tensors, aux_memory_tensors], dim=0)

        aux_memory_labels = list(aux_memory_labels.values())
        aux_memory_labels = torch.cat(aux_memory_labels, dim=0)
        self.episodic_memory_labels = torch.cat([self.episodic_memory_labels, aux_memory_labels], dim=0).type(torch.LongTensor)

        return None
    
    def sleep_phase(self, 
                    num_episodes_per_sleep, 
                    optimizer, 
                    criterions, 
                    scheduler,
                    classes_list=None, 
                    writer=None, 
                    task_id=None):
        '''
        Train model on episodic memory
        '''

        self.view_encoder.eval()
        self.semantic_memory.train()

        # freeze view encoder
        for param in self.view_encoder.parameters():
            param.requires_grad = False
        # unfreeze semantic memory
        for param in self.semantic_memory.parameters():
            param.requires_grad = True

        criterion_crossentropyswap = criterions[0]
        criterion_crosscosinesim = criterions[1]
        critetion_entropyreg = criterions[2]
        # criterion_koleo = criterions[3]

        # Sample sleep episodes idxs from episodic_memory (Uniform sampling with replacement)
        weights = torch.ones(len(self.episodic_memory_tensors)) # All with weight 1 (uniform)
        sampled_episodes_idxs = list(WeightedRandomSampler(weights, num_episodes_per_sleep, replacement=True))

        train_logits = []
        train_probs = []
        train_preds = []
        train_gtlabels = []

        # Train model on sleep episodes a bacth at a time
        for i in range(0, num_episodes_per_sleep, self.episode_batch_size):

            batch_idxs = sampled_episodes_idxs[i:i+self.episode_batch_size]
            
            #### Data ####
            batch_episodes = self.episodic_memory_tensors[batch_idxs]
            batch_episodes = batch_episodes.to(self.args.device)

            #### Forward pass to get logits ####
            batch_logits = torch.empty(0).to(self.args.device)
            # batch_proj_logits = torch.empty(0).to(device)
            for v in range(self.args.num_views):
                tensor_batch = batch_episodes[:,v,:,:,:]
                logits = self.semantic_memory(tensor_batch)
                # logits, proj_logits = self.semantic_memory(img_batch, proj_out=True)
                batch_logits = torch.cat([batch_logits, logits.unsqueeze(1)], dim=1)
                # batch_proj_logits = torch.cat([batch_proj_logits, proj_logits.unsqueeze(1)], dim=1)

            #### Get Pseudo-labels ####

            # Oracle labels
            # batch_episodes_labels = self.episodic_memory_labels[batch_idxs]
            # batch_labels = einops.rearrange(batch_episodes_labels, 'b v -> (b v)')
            # batch_labels = encode_label(batch_labels, classes_list, self.args.num_pseudoclasses)
            # batch_labels = einops.rearrange(batch_labels, '(b v) c -> b v c', v=self.args.num_views)
            # batch_labels = batch_labels.to(self.args.device) # all episodes labels and all views in one batch of (b v)

            # MIRA pseudolabels
            batch_labels = mira_pseudolabeling(batch_logits = batch_logits, 
                                               num_views = self.args.num_views,
                                               tau=self.tau_t, 
                                               beta=self.beta, 
                                               iters=30)

            optimizer.zero_grad()

            #### Losses ####
            crossentropyswap_loss = criterion_crossentropyswap(batch_logits/self.tau_s, batch_labels)
            # crosscosinesim_loss = criterion_crosscosinesim(batch_logits, sim_threshold=0.8)

            # num_gt_classes =  (task_id+1)*self.args.class_increment
            # entropy_threshold = critetion_entropyreg.entropy(torch.ones(num_gt_classes)/num_gt_classes, dim=0)
            # entropyreg_loss, entropy_val = critetion_entropyreg(batch_logits/self.tau_s, entropy_threshold=entropy_threshold)
            # koleo_loss = criterion_koleo(batch_proj_logits)
            
            #### Total Loss ####
            loss = crossentropyswap_loss
            # loss = crossentropyswap_loss + 0.3*crosscosinesim_loss
            # loss = crossentropyswap_loss + 0.8*entropyreg_loss
            # loss = crossentropyswap_loss + 0.3*crosscosinesim_loss + 0.7*entropyreg_loss

            # loss = crossentropyswap_loss + 0.3*koleo_loss
            
            loss.backward()
            # if i < int(num_episodes_per_sleep/5): # Don't update linear head for a few iterations (From MIRA)
            #     for param in self.model.module.linear_head.parameters():
            #         param.grad = None

            optimizer.step()
            scheduler.step()

            #### Accumulate for metrics #### (only first view)
            first_view_logits = batch_logits[:,0].detach().cpu()
            first_view_probs = F.softmax(first_view_logits/self.tau_s, dim=1)
            first_view_preds = torch.argmax(first_view_probs, dim=1)
            first_view_gtlabels = self.episodic_memory_labels[batch_idxs][:,0]
            train_logits.append(first_view_logits)
            train_probs.append(first_view_probs)
            train_preds.append(first_view_preds)
            train_gtlabels.append(first_view_gtlabels)

            if i==0 or (i//self.episode_batch_size) % 5 == 0 or i==num_episodes_per_sleep-self.episode_batch_size:
                ps = F.softmax(batch_logits[:,0] / self.tau_s, dim=1).detach().cpu()
                pt = F.softmax(batch_logits[:,0] / self.tau_t, dim=1).detach().cpu()
                _, _, mi_ps = statistics(ps)
                _, _, mi_pt = statistics(pt)
                current_episode_idx = min(i+self.episode_batch_size,num_episodes_per_sleep)
                print(f'Episode [{current_episode_idx}/{num_episodes_per_sleep}] -- lr: {scheduler.get_last_lr()[0]:.6f}' +
                      f' -- mi_ps: {mi_ps.item():.6f} -- mi_pt: {mi_pt.item():.6f}' +
                      f' -- CrossEntropySwap: {crossentropyswap_loss.item():.6f}' +
                    #   f' -- CrossCosineSim: {crosscosinesim_loss.item():.6f}' +
                    #   f' -- EntropyReg: {entropyreg_loss.item():.6f}' +
                    #   f' -- EntropyValue: {entropy_val.item():.6f}' +
                    #   f' -- KoLeo: {koleo_loss.item():.6f}' +
                      f' -- Total: {loss.item():.6f}'
                      )
                
                if writer is not None and task_id is not None:
                    lr = scheduler.get_last_lr()[0]
                    writer.add_scalar('CrossEntropySwap Loss', crossentropyswap_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    # writer.add_scalar('CrossCosineSim Loss', crosscosinesim_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    # writer.add_scalar('EntropyReg Loss', entropyreg_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    # writer.add_scalar('Entropy Value', entropy_val.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    # writer.add_scalar('KoLeo Loss', koleo_loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Total Loss', loss.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('Learning Rate', lr, task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('MI_ps', mi_ps.item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    writer.add_scalar('MI_pt', mi_pt.item(), task_id*num_episodes_per_sleep + current_episode_idx)

                    # Pseudolabel stability. Get first batch of episodes
                    sample_episodes = self.episodic_memory_tensors[:128] # First batch of saved episodes
                    sample_episodes = sample_episodes.to(self.args.device)
                    self.semantic_memory.eval()
                    with torch.no_grad():
                        sample_episodes_logits = torch.empty(0).to(self.args.device)
                        for v in range(self.args.num_views):
                            tensor_batch = sample_episodes[:,v,:,:,:]
                            logits = self.semantic_memory(tensor_batch)
                            sample_episodes_logits = torch.cat([sample_episodes_logits, logits.unsqueeze(1)], dim=1)
                        sample_episodes_labels = mira_pseudolabeling(batch_logits = sample_episodes_logits, 
                                                                    num_views = self.args.num_views,
                                                                    tau=self.tau_t, 
                                                                    beta=self.beta, 
                                                                    iters=30)
                        sample_episodes_preds = torch.argmax(sample_episodes_labels, dim=2)
                        writer.add_scalar(f'Pseudolabel_Stability_episode{0}_view{0}', sample_episodes_preds[0,0].item(), task_id*num_episodes_per_sleep + current_episode_idx)
                        writer.add_scalar(f'Pseudolabel_Stability_episode{1}_view{0}', sample_episodes_preds[1,0].item(), task_id*num_episodes_per_sleep + current_episode_idx)
                        writer.add_scalar(f'Pseudolabel_Stability_episode{2}_view{0}', sample_episodes_preds[2,0].item(), task_id*num_episodes_per_sleep + current_episode_idx)
                    self.semantic_memory.train()

            if self.args is not None and (i==0 or (i//self.episode_batch_size) % 100 == 0 or i==num_episodes_per_sleep-self.episode_batch_size):
                save_dir_images = os.path.join(self.args.save_dir, 'images_probs_pseudolabels')
                os.makedirs(save_dir_images, exist_ok=True)
                ep_idx=0 # episode index within the batch
                current_episode_idx = min(i+self.episode_batch_size,num_episodes_per_sleep)
                episode_plot = batch_logits[ep_idx]

                ### Plot probabilities and pseudolabel. First 3 views in an episode

                # View probs
                plt.figure(figsize=(12, 4))
                for view_idx in range(3):
                    view_probs = F.softmax(episode_plot[view_idx], dim=0).detach().cpu().numpy()
                    plt.subplot(1, 3, view_idx+1)
                    plt.bar(np.arange(len(view_probs)), view_probs)
                    plt.title(f'Probabilities view {view_idx}')
                    plt.xlabel('Cluster ID')
                    plt.ylabel('Probability')
                    plt.xticks(ticks=np.arange(len(view_probs)), labels=np.arange(len(view_probs)))
                    plt.grid()
                    plt.ylim(0, 1)
                plt.savefig(os.path.join(save_dir_images, f'view_probs_taskid_{task_id}_episode_{current_episode_idx}.png'), bbox_inches='tight')
                plt.close()

                # Image_probs_sharp
                plt.figure(figsize=(12, 4))
                for view_idx in range(3):
                    view_probs_sharp = F.softmax(episode_plot[view_idx]/self.tau_s, dim=0).detach().cpu().numpy()
                    plt.subplot(1, 3, view_idx+1)
                    plt.bar(np.arange(len(view_probs_sharp)), view_probs_sharp)
                    plt.title(f'Sharp Probabilities view {view_idx}')
                    plt.xlabel('Cluster ID')
                    plt.ylabel('Probability')
                    plt.xticks(ticks=np.arange(len(view_probs_sharp)), labels=np.arange(len(view_probs_sharp)))
                    plt.grid()
                    plt.ylim(0, 1)
                plt.savefig(os.path.join(save_dir_images, f'view_probs_sharp_taskid_{task_id}_episode_{current_episode_idx}.png'), bbox_inches='tight')
                plt.close()

                # Image_pseudolabel
                plt.figure(figsize=(12, 4))
                for view_idx in range(3):
                    view_pseudolabel = batch_labels[ep_idx, view_idx].detach().cpu().numpy()
                    plt.subplot(1, 3, view_idx+1)
                    plt.bar(np.arange(len(view_pseudolabel)), view_pseudolabel)
                    plt.title(f'Pseudolabel view {view_idx}')
                    plt.xlabel('Cluster ID')
                    plt.ylabel('Probability')
                    plt.xticks(ticks=np.arange(len(view_pseudolabel)), labels=np.arange(len(view_pseudolabel)))
                    plt.grid()
                    plt.ylim(0, 1)
                plt.savefig(os.path.join(save_dir_images, f'view_pseudolabel_taskid_{task_id}_episode_{current_episode_idx}.png'), bbox_inches='tight')
                plt.close()

                # batchmean view probs sharp across the batch (first view only)
                batchmean_view_probs_sharp = F.softmax((batch_logits[:,0]/self.tau_s).mean(dim=0), dim=0).detach().cpu().numpy()
                plt.figure(figsize=(6, 4))
                plt.bar(np.arange(len(batchmean_view_probs_sharp)), batchmean_view_probs_sharp)
                plt.title(f'Mean Sharp Probabilities')
                plt.xlabel('Cluster ID')
                plt.ylabel('Probability')
                plt.xticks(ticks=np.arange(len(batchmean_view_probs_sharp)), labels=np.arange(len(batchmean_view_probs_sharp)))
                plt.grid()
                plt.ylim(0, 1)
                plt.savefig(os.path.join(save_dir_images, f'batchmean_view_probs_sharp_taskid_{task_id}_batch_{int(current_episode_idx/self.episode_batch_size)}.png'), bbox_inches='tight')
                plt.close()

        #### Track train metrics ####
        save_dir_clusters=os.path.join(self.args.save_dir, 'training_tacking')
        os.makedirs(save_dir_clusters, exist_ok=True)

        print('Train metrics:')
        train_logits = torch.cat(train_logits).numpy()
        train_probs = torch.cat(train_probs).numpy()
        train_preds = torch.cat(train_preds).numpy()
        train_gtlabels = torch.cat(train_gtlabels).numpy()

        calc_cluster_acc = len(np.unique(train_gtlabels)) == self.args.num_pseudoclasses
        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(train_gtlabels.astype(int), train_preds.astype(int), calc_acc=calc_cluster_acc, total_probs=train_probs)
        print(f'NMI: {nmi:.4f}, AMI: {ami:.4f}, ARI: {ari:.4f}, F: {fscore:.4f}, ACC: {adjacc:.4f}, ACC-Top5: {top5:.4f}')

        ### Plot number of samples per cluster with color per class
        labels_IDs = np.unique(train_gtlabels)
        dict_class_vs_clusters = {}
        for i in labels_IDs:
            dict_class_vs_clusters[f'Class {i}'] = []
            for j in range(self.args.num_pseudoclasses):
                indices = (train_gtlabels==i) & (train_preds==j)
                dict_class_vs_clusters[f'Class {i}'].append(np.sum(indices))
        df = pd.DataFrame(dict_class_vs_clusters)
        df.plot.bar(stacked=True, figsize=(10, 8))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
        plt.title(f'Training Task {task_id}\nNumber of samples per cluster per class TaskID: {task_id}')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of samples')
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(save_dir_clusters, f'number_samples_per_cluster_per_class_taskid_{task_id}.png'), bbox_inches='tight')
        plt.close()
        
        ### Measure semantics with entropy
        ## Class entropy (high entropy-> class is spread out across clusters. low entropy-> class is concentrated in few clusters)
        class_entropy_mean = 0
        n=0
        for class_name, num_across_clusters in dict_class_vs_clusters.items():
            if np.sum(num_across_clusters) > 0:
                value_probs = np.array(num_across_clusters)/np.sum(num_across_clusters)
                class_entropy = -np.sum(value_probs * np.log(value_probs + 1e-5))
                class_entropy_mean += class_entropy
                n += 1
        class_entropy_mean = class_entropy_mean / n
        class_entropy_uniform = -np.log(1/self.args.num_pseudoclasses)

        ## Cluster entropy (high entropy-> cluster contains many classes. low entropy-> cluster contains few classes)
        cluster_entropy_mean = 0
        n=0
        for cluster_id in range(self.args.num_pseudoclasses):
            num_across_classes = [dict_class_vs_clusters[f'Class {class_id}'][cluster_id] for class_id in labels_IDs]
            if np.sum(num_across_classes) > 0:
                value_probs = np.array(num_across_classes)/np.sum(num_across_classes)
                cluster_entropy = -np.sum(value_probs * np.log(value_probs + 1e-5))
                cluster_entropy_mean += cluster_entropy
                n += 1
        cluster_entropy_mean = cluster_entropy_mean / n
        cluster_entropy_uniform = -np.log(1/len(labels_IDs))
        # print('\tClass entropy (high entropy-> class is spread out across clusters. low entropy-> class is concentrated in few clusters)')
        print(f'\tClass entropy uniform: {class_entropy_uniform:.4f} -- Class entropy mean: {class_entropy_mean:.4f}')
        # print('\tCluster entropy (high entropy-> cluster contains many classes. low entropy-> cluster contains few classes)')
        print(f'\tCluster entropy uniform: {cluster_entropy_uniform:.4f} -- Cluster entropy mean: {cluster_entropy_mean:.4f}')  
        
        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5,
                        'ClassEntropyUniform': class_entropy_uniform, 'ClassEntropyMean': class_entropy_mean,
                        'ClusterEntropyUniform': cluster_entropy_uniform, 'ClusterEntropyMean': cluster_entropy_mean}

        return task_metrics
    
    def evaluate_model(self, 
                       val_loader,  
                       calc_cluster_acc=False,
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
                images = images.to(self.args.device)
                logits = self.semantic_memory(self.view_encoder(images, before_avgpool=True))
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

        nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(all_labels.astype(int), all_preds.astype(int), calc_acc=calc_cluster_acc, total_probs=all_probs)
        print(f'\tNMI: {nmi:.4f}, AMI: {ami:.4f}, ARI: {ari:.4f}, F: {fscore:.4f}, ACC: {adjacc:.4f}, ACC-Top5: {top5:.4f}')

        dict_class_vs_clusters = {}
        labels_IDs = np.unique(all_labels)
        for i in labels_IDs:
            dict_class_vs_clusters[f'Class {i}'] = []
            for j in range(self.args.num_pseudoclasses):
                indices = (all_labels==i) & (all_preds==j)
                dict_class_vs_clusters[f'Class {i}'].append(np.sum(indices))

        if plot_clusters:
            assert save_dir_clusters is not None
            assert task_id is not None
            assert mean is not None
            assert std is not None

            ### plot 25 images per cluster
            os.makedirs(save_dir_clusters, exist_ok=True)
            for i in range(self.args.num_pseudoclasses): 
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
                grid.save(os.path.join(save_dir_clusters, image_name))
                if i == 19: # Only plot 20 clusters (from 0 to 19) if num_pseudoclasses > 20
                    break

            name = save_dir_clusters.split('/')[-1]

            ### Plot all logits in a 2D space. (PCA, then, t-SNE).
            tsne = TSNE(n_components=2, random_state=0)
            all_logits_2d = tsne.fit_transform(all_logits)
            # Legend is cluster ID
            plt.figure(figsize=(8, 8))
            clusters_IDs = np.unique(all_preds)
            for i in clusters_IDs:
                indices = all_preds==i
                plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Cluster {i}', alpha=0.75, s=20, color=sns.color_palette("husl", self.args.num_pseudoclasses)[i])
            plt.title(f'{name}\nLogits 2D space TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(clusters_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'logits_2d_space_clusters_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()
            # Legend is GT class
            plt.figure(figsize=(8, 8))
            for i in labels_IDs:
                indices = all_labels==i
                plt.scatter(all_logits_2d[indices, 0], all_logits_2d[indices, 1], label=f'Class {i}', alpha=0.75, s=20)
            plt.title(f'{name}\nLogits 2D space TaskID: {task_id}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.savefig(os.path.join(save_dir_clusters, f'logits_2d_space_labels_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

            ### Plot headmap showing cosine similarity matrix of each cluster weights
            clusters_weights = self.semantic_memory.module.linear_head.weight.data.cpu()
            clusters_weights = F.normalize(clusters_weights, p=2, dim=1)
            clusters_cosine_sim = torch.mm(clusters_weights, clusters_weights.T)
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(clusters_cosine_sim, cmap='viridis', ax=ax, annot= self.args.num_pseudoclasses==10 , vmax=1, vmin=-1)
            plt.title(f'{name}\nCosine Similarity Matrix TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Cluster ID')
            plt.savefig(os.path.join(save_dir_clusters, f'cosine_similarity_matrix_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

            ### Plot number of samples per cluster with color per class
            df = pd.DataFrame(dict_class_vs_clusters)
            df.plot.bar(stacked=True, figsize=(10, 8))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol= 1 if len(labels_IDs) < 20 else 2)
            plt.title(f'{name}\nNumber of samples per cluster per class TaskID: {task_id}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of samples')
            plt.xticks(rotation=0)
            plt.savefig(os.path.join(save_dir_clusters, f'number_samples_per_cluster_per_class_taskid_{task_id}.png'), bbox_inches='tight')
            plt.close()

        ### Measure semantics with entropy
        ## Class entropy (high entropy-> class is spread out across clusters. low entropy-> class is concentrated in few clusters)
        class_entropy_mean = 0
        n=0
        for class_name, num_across_clusters in dict_class_vs_clusters.items():
            if np.sum(num_across_clusters) > 0:
                value_probs = np.array(num_across_clusters)/np.sum(num_across_clusters)
                class_entropy = -np.sum(value_probs * np.log(value_probs + 1e-5))
                class_entropy_mean += class_entropy
                n += 1
        class_entropy_mean = class_entropy_mean / n
        class_entropy_uniform = -np.log(1/self.args.num_pseudoclasses)
        ## Cluster entropy (high entropy-> cluster contains many classes. low entropy-> cluster contains few classes)
        cluster_entropy_mean = 0
        n=0
        for cluster_id in range(self.args.num_pseudoclasses):
            num_across_classes = [dict_class_vs_clusters[f'Class {class_id}'][cluster_id] for class_id in labels_IDs]
            if np.sum(num_across_classes) > 0:
                value_probs = np.array(num_across_classes)/np.sum(num_across_classes)
                cluster_entropy = -np.sum(value_probs * np.log(value_probs + 1e-5))
                cluster_entropy_mean += cluster_entropy
                n += 1
        cluster_entropy_mean = cluster_entropy_mean / n
        cluster_entropy_uniform = -np.log(1/len(labels_IDs))
        # print('\tClass entropy (high entropy-> class is spread out across clusters. low entropy-> class is concentrated in few clusters)')
        print(f'\tClass entropy uniform: {class_entropy_uniform:.4f} -- Class entropy mean: {class_entropy_mean:.4f}')
        # print('\tCluster entropy (high entropy-> cluster contains many classes. low entropy-> cluster contains few classes)')
        print(f'\tCluster entropy uniform: {cluster_entropy_uniform:.4f} -- Cluster entropy mean: {cluster_entropy_mean:.4f}')

        #### Gather task metrics ####
        task_metrics = {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5,
                        'ClassEntropyUniform': class_entropy_uniform, 'ClassEntropyMean': class_entropy_mean,
                        'ClusterEntropyUniform': cluster_entropy_uniform, 'ClusterEntropyMean': cluster_entropy_mean}

        return task_metrics

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
def mira_pseudolabeling(batch_logits, num_views, tau, beta, iters):
    targets = torch.empty(0).to(batch_logits.device)
    for t in range(num_views):
        targets_t = mira(batch_logits[:,t], tau, beta, iters)
        targets = torch.cat([targets, targets_t.unsqueeze(1)], dim=1)
    return targets