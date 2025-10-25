import argparse
import os, time

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR

from continuum.datasets import InMemoryDataset
from continuum import InstanceIncremental
from continuum.tasks import TaskType

from models_deit3_clscausal import *
from augmentations import Episode_Transformations, collate_function
from wake_sleep_trainer_clscausal import Wake_Sleep_trainer, eval_classification_performance

from utils import get_imgpath_label, time_duration_print, file_broadcast_list, plot_generated_images_hold_set

import numpy as np
import json
import random
from PIL import Image
from tqdm import tqdm

from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

parser = argparse.ArgumentParser(description='Wake-Sleep Training - Supervised - iid training')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/home/jhair/datasets/Create_ImageNet100_ID_OOD/ImageNet-100_gmedian_quantile@alpha0.9_id_ood_dinov3_vits16plus')
parser.add_argument('--class_idx_file', type=str, default='./IM100_class_index/imagenet100_class_index.json')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--num_tasks', type=int, default=10)
parser.add_argument('--num_tasks_to_run', type=int, default=5)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Pre-trained folder
parser.add_argument('--pretrained_folder', type=str, default='./output/Pretrained_IM100B/Causal_deit_tiny_patch16_LS_100c_4viewsCLSallviews_bs80_epochs100_ENC_lr0.0005wd0.05_CONDGEN_lr0.0003wd0_loss@ent0.1firstview@penalty0.1CEweight0.1_sup1.0_condgen1.0_attndiv1.0_seed0')
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--enc_model_checkpoint', type=str, default='view_encoder_epoch99.pth')
parser.add_argument('--enc_lr', type=float, default=0.0003)
parser.add_argument('--enc_wd', type=float, default=0.05)
parser.add_argument('--drop_path', type=float, default=0.0125)
# Classifier parameters
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
parser.add_argument('--classifier_model_checkpoint', type=str, default='classifier_epoch99.pth')
parser.add_argument('--classifier_lr', type=float, default=0.0008)
parser.add_argument('--classifier_wd', type=float, default=0)
parser.add_argument('--cls_layers', type=int, default=1)
parser.add_argument('--cls_nheads', type=int, default=1)
parser.add_argument('--cls_dropout', type=float, default=0.4)
parser.add_argument('--cls_firstviewdroprate', type=float, default=0.0)
# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--condgen_model_checkpoint', type=str, default='cond_generator_epoch99.pth')
parser.add_argument('--condgen_lr', type=float, default=0.0003)
parser.add_argument('--condgen_wd', type=float, default=0)
parser.add_argument('--cond_dropout', type=float, default=0)
# Training parameters
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--num_episodes_per_sleep', type=int, default=670000)
parser.add_argument('--episode_batch_size', type=int, default=80)
parser.add_argument('--sampling_method', type=str, default='uniform_class_balanced', choices=['uniform', 'uniform_class_balanced', 'GRASP']) # uniform, random, sequential
parser.add_argument('--NREMclstype', type=str, default='storedcls', choices=['storedcls', 'gencls'], help='Type of classifier head. "storedcls" uses the stored cls tokens, "gencls" uses the cls tokens from the generated images.')
parser.add_argument('--NREMview_order', type=str, default='ori', choices=['ori', 'rev', 'rand', 'rev50', 'rand50'], help='Order of views for the conditional generator. "original" keeps the order, "reverse" reverses it, and "random" applies a different random permutation to each element in the batch.')
parser.add_argument('--REMviewstouse', type=str, default='nofirstview', choices=['nofirstview', 'allviews'], help='If "use", the first view is used during REM. If "ignore", the first view is ignored during REM.')
parser.add_argument('--REMskip', action='store_true', help='Skip REM phase')
# REM extra losses and CE weighting parameters
parser.add_argument('--lambda_negshannonent', type=float, default=0.0)
parser.add_argument('--lambda_firstviewpenalty', type=float, default=0.0)
parser.add_argument('--firstviewCEweight', type=float, default=1.0)
# Other parameters
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/wake_sleep_recalwithGenImg/run_debug")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations.
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_ep_plot', type=int, default=10) # number of episodes to plot per task (for debugging purposes)

class AttentionDiversificationLoss(torch.nn.Module):
    def __init__(self, exclude_first_query: bool = True,
                 reduction: str = 'mean', eps: float = 1e-12):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.exclude_first_query = exclude_first_query
        self.reduction = reduction
        self.eps = eps

    def forward(self, attn_probs: torch.Tensor) -> torch.Tensor:
        B, H, T, S = attn_probs.shape
        A = attn_probs

        # If exclude_first_query is True, exclude the first query (causal: t=0 must attend to key 0 only. Don't penalize it)
        if self.exclude_first_query and T > 1:
            A = A[:, :, 1:, :]   # (B, H, T-1, S)

        # -------- 1) Negative Shannon entropy (per head) --------
        neg_shannon_ent_loss = (A * (A + self.eps).log()).sum(dim=-1)   # (B, H, T')
        # Average over queries, then heads -> per-sample
        neg_shannon_ent_loss = neg_shannon_ent_loss.mean(dim=-1).mean(dim=1) # (B,)

        # -------- 2) First-key penalty (attention to key 0 at t>0) --------
        fk = A[..., 0]  # (B,H,T')
        # For causal attention, support size at original query t is (t+1).
        # After excluding t=0, remaining queries correspond to t=1..T-1 with supports 2..T.
        if self.exclude_first_query:
            supports = torch.arange(2, T + 1, device=A.device, dtype=A.dtype)  # [2..T]
        else:
            supports = torch.arange(1, T + 1, device=A.device, dtype=A.dtype)  # [1..T]
        logS = supports.log().view(1, 1, -1)  # (1,1,T')
        fk = fk / (logS + self.eps)
        # mean over queries, heads -> (B,)
        fk = fk.mean(dim=-1).mean(dim=1)

        if self.reduction == 'mean':
            neg_shannon_ent_loss = neg_shannon_ent_loss.mean()
            fk = fk.mean()
            return neg_shannon_ent_loss, fk

        elif self.reduction == 'sum':
            neg_shannon_ent_loss = neg_shannon_ent_loss.sum()
            fk = fk.sum()
            return neg_shannon_ent_loss, fk

        elif self.reduction is None:
            return neg_shannon_ent_loss, fk
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
            
def main():

    ### Parse arguments
    args = parser.parse_args()


    ### Create save dir folder
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
    

    ### Define loggers
    tb_logger = TensorBoardLogger(root_dir=os.path.join(args.save_dir, "logs"), name="tb_logs")
    csv_logger = CSVLogger(root_dir=os.path.join(args.save_dir, "logs"), name="csv_logs",  flush_logs_every_n_steps=1)


    ### Define Fabric and launch it
    fabric = Fabric(accelerator="gpu", strategy="ddp", devices="auto", precision="bf16-mixed", loggers=[tb_logger, csv_logger])
    fabric.launch()


    ### Seed everything
    fabric.seed_everything(args.seed + fabric.local_rank, verbose=False)


    ### Load pre-trained model args
    args_pretrained = json.load(open(os.path.join(args.pretrained_folder, 'args.json'), 'r'))
    args.img_num_tokens = args_pretrained['img_num_tokens']
    args.cond_num_layers = args_pretrained['cond_num_layers']
    args.cond_nhead = args_pretrained['cond_nhead']
    args.cond_dim_ff = args_pretrained['cond_dim_ff']
    args.aug_feature_dim = args_pretrained['aug_feature_dim']
    args.aug_num_tokens_max = args_pretrained['aug_num_tokens_max']
    args.aug_n_layers = args_pretrained['aug_n_layers']
    args.aug_n_heads = args_pretrained['aug_n_heads']
    args.aug_dim_ff = args_pretrained['aug_dim_ff']
    args.upsampling_num_Blocks = args_pretrained['upsampling_num_Blocks']
    args.upsampling_num_out_channels = args_pretrained['upsampling_num_out_channels']


    ### Save args
    if fabric.is_global_zero:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    fabric.barrier()


    ### Recalculate workers, batch size, number of episodes per sleep, and number of batch episodes per sleep
    args.workers = int(args.workers / fabric.world_size)
    args.episode_batch_size = int(args.episode_batch_size / fabric.world_size)
    args.num_episodes_per_sleep = int(np.ceil(args.num_episodes_per_sleep / fabric.world_size))
    args.num_batch_episodes_per_sleep = int(np.ceil(args.num_episodes_per_sleep / args.episode_batch_size))


    ### Load data
    '''Create tasks
    Note: To make continuum work with action_codes, I edited the following file:
    /home/jhair/anaconda3/envs/py39gpu/lib/python3.9/site-packages/continuum/tasks/image_array_task_set.py
    Specifically, line 112. The idea is to be able to pass a tuple variable on x, where the img is on x[0]
    Maybe I should create a fork version of continuum with that change (and install with setup.py)
    def _prepare_data(self, x, y, t):
        if self.trsf is not None:
            x = self.get_task_trsf(t)(x)
        if type(x) is tuple: ### Change to be able to output action vectors
            if not isinstance(x[0], torch.Tensor):
                x[0] = self._to_tensor(x[0])
        elif not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return x, y, t
    '''
    fabric.print('\n==> Preparing data...')
    # Get transforms
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std),
                        ])
    # Get img_paths, labels, and tasks
    traindir_ID = os.path.join(args.data_path, 'train_ID')
    valdir_ID = os.path.join(args.data_path, 'val_ID')
    valdir_OOD = os.path.join(args.data_path, 'val_OOD')
    train_imgpaths_ID, train_labels_array_ID = get_imgpath_label(traindir_ID, args.class_idx_file)

    # Create task IDs for train dataset (equally divide each class into num_tasks)
    train_tasks_array_ID = np.zeros(len(train_labels_array_ID))
    for class_idx in range(args.num_classes):
        class_indices = np.where(train_labels_array_ID == class_idx)[0]
        # equally divide class indices into num_tasks
        class_indices_per_task = np.array_split(class_indices, args.num_tasks)
        for task_idx, cls_indxs_per_task in enumerate(class_indices_per_task):
            train_tasks_array_ID[cls_indxs_per_task] = task_idx

    val_imgpaths_ID, val_labels_array_ID = get_imgpath_label(valdir_ID, args.class_idx_file)
    val_imgpaths_OOD, val_labels_array_OOD = get_imgpath_label(valdir_OOD, args.class_idx_file)

    # Get datasets and loaders
    train_dataset_continuum_ID = InMemoryDataset(train_imgpaths_ID, train_labels_array_ID, train_tasks_array_ID, data_type=TaskType.IMAGE_PATH)
    val_dataset_continuum_ID = InMemoryDataset(val_imgpaths_ID, val_labels_array_ID, data_type=TaskType.IMAGE_PATH)
    val_dataset_continuum_OOD = InMemoryDataset(val_imgpaths_OOD, val_labels_array_OOD, data_type=TaskType.IMAGE_PATH)
    train_tasks = InstanceIncremental(train_dataset_continuum_ID, transformations=[train_transform]) # Task IDs are already in the dataset
    val_tasks_ID = InstanceIncremental(val_dataset_continuum_ID, transformations=[val_transform], nb_tasks=1)
    val_tasks_OOD = InstanceIncremental(val_dataset_continuum_OOD, transformations=[val_transform], nb_tasks=1)


    ### Load Models
    fabric.print('\n==> Prepare models...')
    view_encoder = eval(args.enc_model_name)(drop_path_rate=args.drop_path, output_before_pool=True)
    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.embed_dim, num_classes=args.num_classes, n_heads=args.cls_nheads, n_layers=args.cls_layers, dropout=args.cls_dropout)
    cond_generator = eval(args.condgen_model_name)(img_num_tokens=args.img_num_tokens,
                                                img_feature_dim = view_encoder.head.weight.shape[1],
                                                num_layers = args.cond_num_layers,
                                                nhead = args.cond_nhead,
                                                dim_ff = args.cond_dim_ff,
                                                dropout = args.cond_dropout,
                                                aug_num_tokens_max = args.aug_num_tokens_max,
                                                aug_feature_dim = args.aug_feature_dim,
                                                aug_n_layers = args.aug_n_layers,
                                                aug_n_heads = args.aug_n_heads,
                                                aug_dim_ff = args.aug_dim_ff,
                                                upsampling_num_Blocks = args.upsampling_num_Blocks,
                                                upsampling_num_out_channels = args.upsampling_num_out_channels)
    view_encoder.head = torch.nn.Identity() # remove the head of the encoder


    ### Print models
    fabric.print('\nView encoder')
    fabric.print(view_encoder)
    fabric.print('\nClassifier')
    fabric.print(classifier)
    fabric.print('\nConditional generator')
    fabric.print(cond_generator)
    fabric.print('\n')


    ### Setup models
    view_encoder = fabric.setup_module(view_encoder)
    classifier = fabric.setup_module(classifier)
    cond_generator = fabric.setup_module(cond_generator)


    ### Load pre-trained weights if available 
    # Fully load view encoder
    if args.enc_model_checkpoint is not None:
        fabric.print(f'Loading view encoder from {args.pretrained_folder}/{args.enc_model_checkpoint}')
        fabric.load_raw(os.path.join(args.pretrained_folder, args.enc_model_checkpoint), view_encoder)
    # Fully load conditional generator
    if args.condgen_model_checkpoint is not None:
        fabric.print(f'Loading conditional generator from {args.pretrained_folder}/{args.condgen_model_checkpoint}')
        fabric.load_raw(os.path.join(args.pretrained_folder, args.condgen_model_checkpoint), cond_generator)
    # Partially load classifier (load only projector and causal transformer subnetworks)
    if args.classifier_model_checkpoint is not None:
        fabric.print(f"Loading classifier (partial: projector + transf) from {args.pretrained_folder}/{args.classifier_model_checkpoint}")
        if fabric.is_global_zero:
            ckpt_state = torch.load(os.path.join(args.pretrained_folder, args.classifier_model_checkpoint), map_location="cpu")
        else:
            ckpt_state = None
        ckpt_state = fabric.broadcast(ckpt_state, src=0)
        # Keep only projector.* and transf.* keys
        allowed_prefixes = ('projector.', 'transf.')
        filtered_state = {k: v for k, v in ckpt_state.items() if k.startswith(allowed_prefixes)}
        incompatible = classifier.load_state_dict(filtered_state, strict=False)
        num_loaded = len(filtered_state)
        fabric.print(f"Classifier partial load → loaded keys: {num_loaded}, missing: {len(incompatible.missing_keys)}, unexpected: {len(incompatible.unexpected_keys)}")
    fabric.barrier()


    ### Initialize linear head weights with data driven initialization
    # This makes each class weight vector to be the mean of the data vectors of that class
    all_vectors = []
    all_labels = []
    view_encoder.eval()
    classifier.eval()
    with torch.no_grad():
        fabric.print(f"Initializing linear head weights with data driven initialization")
        train_loader_aux = torch.utils.data.DataLoader(train_tasks[0], batch_size=args.episode_batch_size, shuffle=True, collate_fn=collate_function, num_workers=4)
        train_loader_aux = fabric.setup_dataloaders(train_loader_aux)
        for batch_episodes, batch_labels, _ in train_loader_aux:
            batch_episodes_imgs = batch_episodes[0]
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)) # (B, V, 1)
            B, V, C, H, W = batch_episodes_imgs.shape
            flat_episodes_imgs = batch_episodes_imgs.view(B*V, C, H, W) # (B*V, C, H, W)
            flat_episodes_tensors = view_encoder(flat_episodes_imgs) # (B*V, T, D)
            flat_episodes_clsvectors = flat_episodes_tensors[:, 0, :] # (B*V, D)
            x = classifier.projector(flat_episodes_clsvectors)
            x = x.reshape(B, V, -1)
            x = x + classifier.pos_embed(V)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(V).to(x.device) # (T,T)
            x = classifier.transf(x, mask=causal_mask)
            flat_vectors = x.reshape(B * V, -1)
            flat_episodes_labels = batch_episodes_labels.view(B*V, 1).squeeze() # (B*V)
            all_vectors.append(flat_vectors)
            all_labels.append(flat_episodes_labels)
        # Concatenate all vectors and labels
        all_vectors = torch.cat(all_vectors, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # Gather all vectors and labels from all ranks
        all_vectors = fabric.all_gather(all_vectors).reshape(-1, all_vectors.size(1))
        all_labels = fabric.all_gather(all_labels).reshape(-1)
        # Calculate mean of each class
        for class_idx in range(args.num_classes):
            class_indices = torch.where(all_labels == class_idx)[0]
            class_data_vectors = all_vectors[class_indices]
            class_mean_vector = class_data_vectors.mean(dim=0)
            classifier.classifier_head.weight[class_idx] = class_mean_vector
        del all_vectors, all_labels, train_loader_aux

    ### Define wake-sleep trainer engine ###
    WS_trainer = Wake_Sleep_trainer(episode_batch_size = args.episode_batch_size,
                                    num_episodes_per_sleep = args.num_episodes_per_sleep,
                                    num_views = args.num_views,
                                    dataset_mean = args.mean,
                                    dataset_std = args.std,
                                    save_dir = args.save_dir,
                                    print_freq = args.print_frequency)

    ### Load criterion
    criterion_sup = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_attn_div = AttentionDiversificationLoss(exclude_first_query=True, reduction='mean', eps=1e-12) # Only used during REM
    criterion_condgen = torch.nn.MSELoss()


    ### Save one batch for plot purposes (all tasks)
    fabric.seed_everything(args.seed + fabric.local_rank, verbose=False)  # Seed for reproducibility of the plot
    episodes_plot_dict = {}
    for i in range(1, args.num_tasks_to_run+1):
        train_loader_aux = torch.utils.data.DataLoader(train_tasks[i], batch_size=args.num_ep_plot, shuffle=True, collate_fn=collate_function)
        train_loader_aux = fabric.setup_dataloaders(train_loader_aux)
        episodes_plot, _, _ = next(iter(train_loader_aux))
        episodes_plot_dict[f"taskid_{i}"] = episodes_plot
        del train_loader_aux

    ### Plot generated images with pre-trained model (before wake-sleep training)
    plot_generated_images_hold_set(fabric, view_encoder, cond_generator, episodes_plot_dict, 0, 
                                   args.mean, args.std, args.num_views, args.save_dir, fabric.device)


    ### Load validation sets (ID and OOD)
    val_dataset_ID = val_tasks_ID[:]
    val_loader_ID = torch.utils.data.DataLoader(val_dataset_ID, batch_size=args.episode_batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)
    val_dataset_OOD = val_tasks_OOD[:]
    val_loader_OOD = torch.utils.data.DataLoader(val_dataset_OOD, batch_size=args.episode_batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    ### Setup validation loaders
    val_loader_ID = fabric.setup_dataloaders(val_loader_ID)
    val_loader_OOD = fabric.setup_dataloaders(val_loader_OOD)


    ### Test performance at pre-trained network but random init classifier
    fabric.print(f'\nValidation metrics (pre-trained network and random init classifier)')
    eval_classification_performance(fabric, view_encoder, classifier, val_loader_ID, criterion_sup, 
                                    "Val_ID", WS_trainer.total_num_seen_episodes)
    eval_classification_performance(fabric, view_encoder, classifier, val_loader_OOD, criterion_sup,
                                    "Val_OOD", WS_trainer.total_num_seen_episodes)




    ###########################################
    #### Wake-Sleep Learning with NREM-REM ####
    ###########################################

    fabric.print(f'\n==> Total number of tasks: {args.num_tasks}')
    num_samples_per_task = len(train_tasks[0])
    fabric.print(f'==> Number of episodes loaded per task: {num_samples_per_task}')
    if args.num_tasks != args.num_tasks_to_run:
        fabric.print(f'==> I am not running all tasks, I am running only the first {args.num_tasks_to_run} tasks')
    else:
        fabric.print(f'==> Number of tasks to run: {args.num_tasks}')

    fabric.print('\n==> Start Training on incoming tasks')
    init_time = time.time()

    #### TASKS LOOP ####
    for task_id in range(1, args.num_tasks_to_run+1):
        task_start_time = time.time()

        fabric.print(f"\n\n\n##------ Learning Task {task_id}/{args.num_tasks_to_run} ------##")
        fabric.log(name='Task_boundaries', value=task_id, step=WS_trainer.total_num_seen_episodes)

        #----------------#
        ### WAKE PHASE ###
        #----------------#
        fabric.print(f"\n=== WAKE PHASE ===")
        train_loader_current_task = torch.utils.data.DataLoader(
            train_tasks[task_id-1],
            batch_size=args.episode_batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_function,
            pin_memory=True
        )
        train_loader_current_task = fabric.setup_dataloaders(train_loader_current_task)
        WS_trainer.wake_phase(fabric, view_encoder, train_loader_current_task)
        WS_trainer.reset_sleep_counter() # Reset sleep counter to start sleep session
        del train_loader_current_task


        #-----------------#
        ### SLEEP PHASE ###
        #-----------------#
        fabric.print(f"\n=== SLEEP PHASE ===")
        # Define optimizers
        optimizer_encoder = torch.optim.AdamW(view_encoder.parameters(), lr=args.enc_lr, weight_decay=args.enc_wd)
        optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_wd)
        optimizer_condgen = torch.optim.AdamW(cond_generator.parameters(), lr=args.condgen_lr, weight_decay=args.condgen_wd)
        # Setup optimizers
        optimizer_encoder = fabric.setup_optimizers(optimizer_encoder)
        optimizer_classifier = fabric.setup_optimizers(optimizer_classifier)
        optimizer_condgen = fabric.setup_optimizers(optimizer_condgen)
        # Define schedulers
        scheduler_encoder = OneCycleLR(optimizer_encoder, max_lr=args.enc_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        scheduler_classifier = OneCycleLR(optimizer_classifier, max_lr=args.classifier_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        scheduler_condgen = OneCycleLR(optimizer_condgen, max_lr=args.condgen_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1) # TODO: check if this is correct

        ### Sample indexes
        WS_trainer.sampling_idxs_for_sleep(args.num_episodes_per_sleep, sampling_method=args.sampling_method)

        ### NREM-REM cycles ###
        nrem_rem_cycle_counter = 1 # For printing purposes
        while WS_trainer.sleep_episode_counter < args.num_episodes_per_sleep:

            ### NREM 
            fabric.print(f"=== NREM - sleep cycle {nrem_rem_cycle_counter}")
            WS_trainer.NREM_sleep(fabric, 
                                  view_encoder = view_encoder,
                                  classifier = classifier, 
                                  cond_generator = cond_generator,
                                  optimizers = [optimizer_encoder, optimizer_classifier, optimizer_condgen], 
                                  schedulers = [scheduler_encoder, scheduler_classifier, scheduler_condgen],
                                  criterions = [criterion_sup, criterion_condgen],
                                  task_id = task_id,
                                  mean = args.mean,
                                  std = args.std,
                                  save_dir = args.save_dir,
                                  view_order=args.NREMview_order,
                                  clstype=args.NREMclstype)
            fabric.print(f'Validation metrics')
            eval_classification_performance(fabric, view_encoder, classifier, val_loader_ID, criterion_sup, 
                                            "Val_ID", WS_trainer.total_num_seen_episodes)
            eval_classification_performance(fabric, view_encoder, classifier, val_loader_OOD, criterion_sup, 
                                            "Val_OOD", WS_trainer.total_num_seen_episodes)
            
            ### REM
            if args.REMskip:
                fabric.print(f"=== REM - sleep cycle {nrem_rem_cycle_counter} skipped")
                WS_trainer.sleep_episode_counter += args.num_episodes_per_sleep/2
                WS_trainer.total_num_seen_episodes += args.num_episodes_per_sleep/2
                fabric.print(f'Validation metrics (REM skipped)')
                eval_classification_performance(fabric, view_encoder, classifier, val_loader_ID, criterion_sup, 
                                                "Val_ID", WS_trainer.total_num_seen_episodes)
                eval_classification_performance(fabric, view_encoder, classifier, val_loader_OOD, criterion_sup, 
                                                "Val_OOD", WS_trainer.total_num_seen_episodes)
                total_num_seen_episodes_allranks = fabric.all_reduce(torch.tensor(WS_trainer.total_num_seen_episodes, device=fabric.device), reduce_op="sum").item()
                fabric.log(name=f'NREM_REM_indicator', value=WS_trainer.rem_indicator, step=total_num_seen_episodes_allranks)
            else:
                fabric.print(f"=== REM - sleep cycle {nrem_rem_cycle_counter}")
                WS_trainer.REM_sleep(fabric,
                                     view_encoder = view_encoder,
                                     classifier = classifier, 
                                     cond_generator = cond_generator,
                                     optimizers = [optimizer_encoder, optimizer_classifier, optimizer_condgen], 
                                     schedulers = [scheduler_encoder, scheduler_classifier, scheduler_condgen],
                                     criterions = [criterion_sup, criterion_condgen, criterion_attn_div],
                                     task_id = task_id,
                                     mean = args.mean,
                                     std = args.std,
                                     save_dir = args.save_dir,
                                     viewstouse= args.REMviewstouse,
                                     lambda_negshannonent=args.lambda_negshannonent,
                                     lambda_firstviewpenalty=args.lambda_firstviewpenalty,
                                     firstviewCEweight=args.firstviewCEweight,
                                     cls_firstviewdroprate=args.cls_firstviewdroprate
                                    )
                fabric.print(f'Validation metrics')
                eval_classification_performance(fabric, view_encoder, classifier, val_loader_ID, criterion_sup, 
                                                "Val_ID", WS_trainer.total_num_seen_episodes)
                eval_classification_performance(fabric, view_encoder, classifier, val_loader_OOD, criterion_sup, 
                                                "Val_OOD", WS_trainer.total_num_seen_episodes)
                # Helper to avoid representational drift (do it after a phase that updates the view encoder)
                # Here, I will update the episodic_memory_tensors by generating their inputs again and pass them through view encoder
                fabric.print(f"=== Recalculate episodic memory with Generated Images")
                WS_trainer.recalculate_episodic_memory_with_gen_imgs(fabric, view_encoder, cond_generator)

            nrem_rem_cycle_counter += 1  # Increment cycle counter
            fabric.barrier()

        # Save models after each task
        fabric.print(f"\n==> Saving models after task {task_id}...")
        state = {"view_encoder": view_encoder, "classifier": classifier, "cond_generator": cond_generator}
        fabric.save(os.path.join(args.save_dir, f'models_checkpoint_task{task_id}.pth'), state=state)
        fabric.print("Models saved.")

        # Plot
        plot_generated_images_hold_set(fabric, view_encoder, cond_generator, episodes_plot_dict, task_id, 
                                        args.mean, args.std, args.num_views, args.save_dir, fabric.device)

        # Task finished, print time taken for the task
        task_duration = time.time() - task_start_time
        fabric.print(f"\nTask {task_id} finished in {time_duration_print(task_duration)}")
        fabric.print(f"Total time taken so far: {time_duration_print(time.time() - init_time)}")

    fabric.barrier()

    return None
            

if __name__ == '__main__':
    main()