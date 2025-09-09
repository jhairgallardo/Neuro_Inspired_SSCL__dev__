import argparse
import os, time

import torch
from torchvision import transforms

from torch.amp import GradScaler
from torch.amp import autocast

import torchvision
from torchvision.datasets import ImageFolder

from models_deit3_clscausal import *
from augmentations import Episode_Transformations, collate_function_notaskid
from utils import MetricLogger, reduce_tensor, accuracy, time_duration_print

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/caltech256/256_ObjectCategories_splits')
parser.add_argument('--num_classes', type=int, default=256)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--enc_lr', type=float, default=0.0008)
parser.add_argument('--enc_wd', type=float, default=0.05)
parser.add_argument('--drop_path', type=float, default=0.0125) # 0.0125 for tiny, 0.05 for small, 0.2 for base
# Classifier parameters
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
parser.add_argument('--cls_layers', type=int, default=1)
parser.add_argument('--cls_nheads', type=int, default=1)
parser.add_argument('--cls_dropout', type=float, default=0.4)
parser.add_argument('--cls_firstviewdroprate', type=float, default=0.8)
parser.add_argument('--cls_viewstouse', type=str, default='allviews', choices=['nofirst', 'allviews', 'reverse', 'reverse50']) # 'nofirst' ignores the first view, 'allviews' uses all views, 'reverse' reverses the order of views, 'reverse50' reverses the order of views 50% of the time
# Conditional generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--img_num_tokens', type=int, default=196)
parser.add_argument('--cond_num_layers', type=int, default=8)
parser.add_argument('--cond_nhead', type=int, default=8)
parser.add_argument('--cond_dim_ff', type=int, default=1024)
parser.add_argument('--cond_dropout', type=float, default=0)
parser.add_argument('--aug_feature_dim', type=int, default=64)
parser.add_argument('--aug_num_tokens_max', type=int, default=16)
parser.add_argument('--aug_n_layers', type=int, default=2)
parser.add_argument('--aug_n_heads', type=int, default=4)
parser.add_argument('--aug_dim_ff', type=int, default=256)
parser.add_argument('--upsampling_num_Blocks', type=list, default=[1,1,1,1])
parser.add_argument('--upsampling_num_out_channels', type=int, default=3)
parser.add_argument('--condgen_lr', type=float, default=0.0008)
parser.add_argument('--condgen_wd', type=float, default=0)
# Attention diversification loss weights
parser.add_argument('--lambda_negshannonent', type=float, default=0.1)
parser.add_argument('--lambda_firstviewpenalty', type=float, default=0.1)
# Training parameters
parser.add_argument('--firstviewCEweight', type=float, default=0.1)
parser.add_argument('--sup_coef', type=float, default=1.0)
parser.add_argument('--condgen_coef', type=float, default=1.0)
parser.add_argument('--attndiv_coef', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--episode_batch_size', type=int, default=80) #512
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--label_smoothing', type=float, default=0.0) # Label smoothing for the supervised loss
# Other parameters
parser.add_argument('--workers', type=int, default=32) # 8 for 1 gpu, 48 for 4 gpus
parser.add_argument('--save_dir', type=str, default="output/Pretrained_caltech256/run_debug")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations.
parser.add_argument('--seed', type=int, default=0)
## DDP args
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    return None

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

def validation_step(val_loader, view_encoder, classifier, val_criterion_sup, device, epoch=None):

    view_encoder.eval()
    classifier.eval()
    results = {'acc1': [], 'acc5': [], 'loss_sup': [], 'epoch': []}

    # Forward pass
    acc1_log = MetricLogger('Acc@1')
    acc5_log = MetricLogger('Acc@5')
    loss_sup_log = MetricLogger('Loss')
    with torch.no_grad():
        for i, (batch_imgs, batch_labels) in enumerate(val_loader):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            batch_tokens = view_encoder(batch_imgs)
            batch_cls_tokens = batch_tokens[:, 0, :]
            logits = classifier(batch_cls_tokens.unsqueeze(1)).squeeze(1)
            acc1, acc5 = accuracy(logits, batch_labels, topk=(1, 5))
            loss_sup = val_criterion_sup(logits, batch_labels)
            acc1_log.update(acc1.item(), batch_imgs.size(0))
            acc5_log.update(acc5.item(), batch_imgs.size(0))
            loss_sup_log.update(loss_sup.item(), batch_imgs.size(0))
    
    results['acc1'] = acc1_log.avg
    results['acc5'] = acc5_log.avg
    results['loss_sup'] = loss_sup_log.avg
    results['epoch'] = epoch

    return results

def main():

    ### Parse arguments
    args = parser.parse_args()

    ### DDP init
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://', device_id=device)
        args.ddp = True
        print(f"DDP used, local rank set to {args.local_rank}. {torch.distributed.get_world_size()} GPUs training.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.local_rank = 0
        args.ddp = False
        print("DDP not used, local rank set to 0. 1 GPU training.")

    # Create save dir folders and save args
    if args.local_rank == 0:
        print(args)
        if not os.path.exists(args.save_dir): # create save dir
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Calculate batch size per GPU
    if args.ddp:
        args.episode_batch_size_per_gpu = int(args.episode_batch_size / torch.distributed.get_world_size())
    else:
        args.episode_batch_size_per_gpu = args.episode_batch_size
    # Calculate number of workers per GPU
    if args.ddp:
        args.workers_per_gpu = int(args.workers / torch.distributed.get_world_size())
    else:
        args.workers_per_gpu = args.workers

    ### Seed everything
    final_seed = args.seed + args.local_rank
    seed_everything(seed=final_seed)

    ### Define tensoboard writer
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load data
    if args.local_rank == 0:
        print('\n==> Preparing data...')
    # Get transforms
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std)
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std),
                        ])
    # Get img_paths, labels and tasks
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    train_dataset = ImageFolder(traindir, transform=train_transform)
    val_dataset = ImageFolder(valdir, transform=val_transform)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=True if train_sampler is None else False,
                                               sampler=train_sampler, num_workers=args.workers_per_gpu, pin_memory=True, persistent_workers=True,
                                               collate_fn=collate_function_notaskid)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size_per_gpu, shuffle=False,
                                             sampler=val_sampler, num_workers=args.workers_per_gpu, pin_memory=True)
    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish

    ### Load models
    if args.local_rank == 0:
        print('\n==> Prepare models...')
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
    if args.local_rank == 0:
        print('\nView encoder')
        print(view_encoder)
        print('\nClassifier')
        print(classifier)
        print('\nConditional generator')
        print(cond_generator)
        print('\n')

    ### Dataparallel and move models to device
    view_encoder = view_encoder.to(device)
    classifier = classifier.to(device)
    cond_generator = cond_generator.to(device)
    if args.ddp:
        view_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(view_encoder)
        view_encoder = torch.nn.parallel.DistributedDataParallel(view_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
        cond_generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cond_generator)
        cond_generator = torch.nn.parallel.DistributedDataParallel(cond_generator, device_ids=[args.local_rank], output_device=args.local_rank)

    ### Load optimizer and criterion
    param_groups_encoder = [{'params': view_encoder.parameters(), 'lr': args.enc_lr, 'weight_decay': args.enc_wd},
                    {'params': classifier.parameters(), 'lr': args.enc_lr, 'weight_decay': args.enc_wd}]
    optimizer_encoder = torch.optim.AdamW(param_groups_encoder, lr=0, weight_decay=0)
    linear_warmup_scheduler_encoder = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_encoder, start_factor=1e-6/args.enc_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_encoder = torch.optim.lr_scheduler.SequentialLR(optimizer_encoder, [linear_warmup_scheduler_encoder, cosine_scheduler_encoder], milestones=[args.warmup_epochs*len(train_loader)])

    optimizer_condgen = torch.optim.AdamW(cond_generator.parameters(), lr=args.condgen_lr, weight_decay=args.condgen_wd)
    linear_warmup_scheduler_condgen = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer_condgen, start_factor=1e-6/args.condgen_lr, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler_condgen = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_condgen, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=0)
    scheduler_condgen = torch.optim.lr_scheduler.SequentialLR(optimizer_condgen, [linear_warmup_scheduler_condgen, cosine_scheduler_condgen], milestones=[args.warmup_epochs*len(train_loader)])

    criterion_sup = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none') # mean, none
    criterion_attn_div = AttentionDiversificationLoss(exclude_first_query=True, reduction='mean', eps=1e-12)
    val_criterion_sup = torch.nn.CrossEntropyLoss()
    criterion_condgen = torch.nn.MSELoss()

    ### Save one batch for plot purposes
    seed_everything(final_seed)  # Reset seed to ensure reproducibility for the batch
    if args.local_rank == 0:
        episodes_plot, _ = next(iter(train_loader))

    #######################################
    ### Validation STEP before training ###
    #######################################
    if args.local_rank == 0:
        results = validation_step(val_loader, 
                                view_encoder.module if args.ddp else view_encoder, 
                                classifier.module if args.ddp else classifier, 
                                val_criterion_sup, device, epoch=-1)
        print(f'Epoch [{-1}] Val Loss Total: {results["loss_sup"]:.6f} -- Val Top1: {results["acc1"]:.2f} -- Val Top5: {results["acc5"]:.2f}')
    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish the validation step

    #### Train loop ####
    if args.local_rank == 0:
        print('\n==> Training model')
    init_time = time.time()
    scaler = GradScaler()

    for epoch in range(args.epochs):
        start_time = time.time()

        if args.local_rank == 0:
            print(f'\n==> Epoch {epoch}/{args.epochs}')

        # DDP init
        if args.ddp:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        
        ##################
        ### Train STEP ###
        ##################
        losscondgen_total_log = MetricLogger('LossCondgen Total')
        loss_gen1_log = MetricLogger('Loss Gen1')
        loss_gen2_log = MetricLogger('Loss Gen2')
        loss_gen3_log = MetricLogger('Loss Gen3')

        train_loss_total = MetricLogger('Train Loss Total')
        train_top1 = MetricLogger('Train Top1 ACC')
        train_top5 = MetricLogger('Train Top5 ACC')

        view_encoder.train()
        cond_generator.train()
        classifier.train()
        for i, (batch_episodes, batch_labels) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes[0].to(device, non_blocking=True) # (B, V, C, H, W)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device, non_blocking=True) # (B, V)
            batch_episodes_actions = batch_episodes[1] # (B, V, A)

            ## Forward pass
            loss_gen1 = 0
            loss_gen2 = 0
            loss_gen3 = 0
            loss_sup = 0
            acc1 = 0
            acc5 = 0
            B, V, C, H, W = batch_episodes_imgs.shape
            with (autocast(device_type='cuda', dtype=torch.float16)):
                # Flatten the batch and views
                flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                flat_feats_and_cls = view_encoder(flat_imgs) # (B*V, 1+T, D)
                # Reshape and get first view features
                all_feats = flat_feats_and_cls.view(B, V, flat_feats_and_cls.size(1), -1) # (B, V, 1+T, D)
                first_view_feats = all_feats[:, 0, 1:, :].detach() # (B, T, D) # Discard the CLS token. Shape is (B, T, D)
                # Reshape to get the CLS token and features
                flat_tensors = all_feats[:, :, 1:, :].reshape(B * V, -1, all_feats.size(-1))  # → (B·V, T, D)
                
                # Reshape and expand the first view features
                flat_first_feats = first_view_feats.unsqueeze(1)  # (B, 1,  T, D)
                flat_first_feats = flat_first_feats.expand(-1, V, -1, -1) # (B, V,  T, D)
                flat_first_feats = flat_first_feats.reshape(B * V, *first_view_feats.shape[1:])   # (B*V, T, D)

                # Get actions
                flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V
                # Run the conditional generator
                flat_gen_imgs, flat_gen_feats = cond_generator(flat_first_feats, flat_actions) # (B*V, C, H, W), (B*V, T, D)
                flat_gen_dec_feats = view_encoder(flat_gen_imgs)[:, 1:, :]  # (B*V, T, D)
                # Run the generator directly (skip conditioning)
                flat_gen_imgs_dir = cond_generator(flat_tensors, None, skip_conditioning=True)  # (B*V, C, H, W)
                flat_gen_dir_feats = view_encoder(flat_gen_imgs_dir)[:, 1:, :]                  # (B*V, T, D)
                # Get generator losses
                loss_gen1 = criterion_condgen(flat_gen_feats, flat_tensors.detach())
                loss_gen2 = criterion_condgen(flat_gen_dec_feats, flat_tensors.detach())
                loss_gen3 = criterion_condgen(flat_gen_dir_feats, flat_tensors.detach())

                # Supervised loss & accuracy (Causal Transformer)
                notflat_cls = all_feats[:, :, 0, :]    # (B, V, D)

                # A) Ignoring first view
                if args.cls_viewstouse == 'nofirst':
                    notflat_cls = notflat_cls[:, 1:, :]    # Discard the first view CLS token (non augmented image) to not overfit.
                    batch_episodes_labels = batch_episodes_labels[:, 1:] # Discard the first view labels (non augmented image) to not overfit.
                    notflat_sup_logits = classifier(notflat_cls,first_token_droprate=args.cls_firstviewdroprate)

                # B) Using all views (original order)
                elif args.cls_viewstouse == 'allviews':
                    notflat_sup_logits = classifier(notflat_cls, first_token_droprate=args.cls_firstviewdroprate)

                # C) Using all views (reverse order so first view is only seens no the final token)
                elif args.cls_viewstouse == 'reverse':
                    notflat_cls = notflat_cls.flip(dims=[1])  # Reverse the order of views ########### TEST THIS TO CHECK IF IT FLIPS THE ORDER CORRECTLY
                    batch_episodes_labels = batch_episodes_labels.flip(dims=[1])  # Reverse the order of labels ########### TEST THIS TO CHECK IF IT FLIPS THE ORDER CORRECTLY
                    notflat_sup_logits = classifier(notflat_cls, first_token_droprate=args.cls_firstviewdroprate)

                # D) Using all views (flip views order 50% of the time)
                elif args.cls_viewstouse == 'reverse50':
                    if random.random() < 0.5:
                        notflat_cls = notflat_cls.flip(dims=[1])
                        batch_episodes_labels = batch_episodes_labels.flip(dims=[1])
                    notflat_sup_logits = classifier(notflat_cls, first_token_droprate=args.cls_firstviewdroprate)

                else:
                    raise ValueError(f"Invalid cls_viewstouse: {args.cls_viewstouse}")

                V = notflat_sup_logits.size(1)
                sup_logits = notflat_sup_logits.reshape(B * V, -1) # (B*T, num_classes)
                sup_labels = batch_episodes_labels.reshape(-1) # (B*T,)
                loss_sup  = criterion_sup(sup_logits, sup_labels)

                loss_sup = loss_sup.view(B, V)
                # first view has less weight (alpha is the fraction from the uniform weight) (reduction should be none for this)
                w = make_token_weights_fractional(V, alpha=args.firstviewCEweight, device=sup_logits.device, dtype=sup_logits.dtype)  # sum=1
                loss_sup = (loss_sup * w).sum(dim=1).mean()

                acc1, acc5 = accuracy(sup_logits, sup_labels, topk=(1, 5))

            # Attention diversification loss (compute it outside mixed precision)
            attn_probs = classifier.module.transf.layers[0].last_attn_probs if args.ddp else classifier.transf.layers[0].last_attn_probs
            neg_shannon_entropy_loss, firstkeypenalty_loss = criterion_attn_div(attn_probs)
 
            # Calculate Total loss for the batch
            loss_attn_div = args.lambda_negshannonent * neg_shannon_entropy_loss + args.lambda_firstviewpenalty * firstkeypenalty_loss
            losssup_total = loss_sup
            losscondgen_total = loss_gen1 + loss_gen2 + loss_gen3
            loss_total = args.sup_coef*losssup_total + args.condgen_coef*losscondgen_total + args.attndiv_coef*loss_attn_div

            ## Backward pass with clip norm
            optimizer_encoder.zero_grad()
            optimizer_condgen.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer_encoder)
            torch.nn.utils.clip_grad_norm_(view_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optimizer_encoder)
            scaler.unscale_(optimizer_condgen)
            torch.nn.utils.clip_grad_norm_(cond_generator.parameters(), 1.0)
            scaler.step(optimizer_condgen)
            scaler.update()
            scheduler_encoder.step()
            scheduler_condgen.step()

            ## Track losses for per batch plotting (Encoder)
            train_loss_total.update(losssup_total.item(), batch_episodes_imgs.size(0))
            train_top1.update(acc1.item(), batch_episodes_imgs.size(0))
            train_top5.update(acc5.item(), batch_episodes_imgs.size(0))

            ## Track losses for per epoch plotting (CondGen)
            losscondgen_total_log.update(losscondgen_total.item(), batch_episodes_imgs.size(0))
            loss_gen1_log.update(loss_gen1.item(), batch_episodes_imgs.size(0))
            loss_gen2_log.update(loss_gen2.item(), batch_episodes_imgs.size(0))
            loss_gen3_log.update(loss_gen3.item(), batch_episodes_imgs.size(0))

            if (args.local_rank == 0) and ((i % args.print_frequency) == 0):
                print(
                    f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                    f'lr Encoder: {scheduler_encoder.get_last_lr()[0]:.6f} -- ' +
                    f'Loss Sup: {losssup_total.item():.6f} -- ' +
                    f'lr CondGen: {scheduler_condgen.get_last_lr()[0]:.6f} -- ' +
                    f'Loss Gen1: {loss_gen1.item():.6f} -- ' +
                    f'Loss Gen2: {loss_gen2.item():.6f} -- ' +
                    f'Loss Gen3: {loss_gen3.item():.6f} -- ' +
                    f'LossCondgen Total: {losscondgen_total.item():.6f}'
                    )
            if args.local_rank == 0:
                writer.add_scalar('lr Encoder', scheduler_encoder.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Supervised', losssup_total.item(), epoch*len(train_loader)+i)
                writer.add_scalar('lr CondGen', scheduler_condgen.get_last_lr()[0], epoch*len(train_loader)+i)
                writer.add_scalar('Loss Gen1', loss_gen1.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Gen2', loss_gen2.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Gen3', loss_gen3.item(), epoch*len(train_loader)+i)
                writer.add_scalar('LossCondgen Total', losscondgen_total.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Negative Shannon Entropy', neg_shannon_entropy_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss First Key Penalty', firstkeypenalty_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss Attention Diversification', loss_attn_div.item(), epoch*len(train_loader)+i)
        
        # Train Epoch metrics
        if args.ddp:
            train_loss_total.all_reduce()
            losscondgen_total_log.all_reduce()
            loss_gen1_log.all_reduce()
            loss_gen2_log.all_reduce()
            loss_gen3_log.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Supervised (per epoch)', train_loss_total.avg, epoch)
            writer.add_scalar('Accuracy (per epoch)', train_top1.avg, epoch)
            writer.add_scalar('Accuracy Top5 (per epoch)', train_top5.avg, epoch)
            writer.add_scalar('LossCondgen Total (per epoch)', losscondgen_total_log.avg, epoch)
            writer.add_scalar('Loss Gen1 (per epoch)', loss_gen1_log.avg, epoch)
            writer.add_scalar('Loss Gen2 (per epoch)', loss_gen2_log.avg, epoch)
            writer.add_scalar('Loss Gen3 (per epoch)', loss_gen3_log.avg, epoch)
            print(f'Epoch [{epoch}] Loss Supervised: {train_loss_total.avg:.6f} -- LossCondgen Total: {losscondgen_total_log.avg:.6f} -- Loss Gen1: {loss_gen1_log.avg:.6f} -- Loss Gen2: {loss_gen2_log.avg:.6f} -- Loss Gen3: {loss_gen3_log.avg:.6f}')

        if args.ddp:
            torch.distributed.barrier()  # Wait for all processes to finish the training epoch

        #######################
        ### Validation STEP ###
        #######################

        # This is only for the supervised task
        val_loss_total = MetricLogger('Val Loss Total')
        val_top1 = MetricLogger('Val Top1 ACC')
        val_top5 = MetricLogger('Val Top5 ACC')
        view_encoder.eval()
        cond_generator.eval()
        classifier.eval()
        for i, (batch_imgs, batch_labels) in enumerate(val_loader):
            with torch.no_grad():
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    batch_tensors = view_encoder(batch_imgs)
                    batch_cls_tokens = batch_tensors[:, 0, :].unsqueeze(1) # (B, 1, D)
                    batch_logits = classifier(batch_cls_tokens) # pass cls token to classifier
                    batch_logits = batch_logits.squeeze(1) # (B, num_classes)
                    loss_sup = val_criterion_sup(batch_logits, batch_labels)
                losssup_total_val = loss_sup
                acc1, acc5 = accuracy(batch_logits, batch_labels, topk=(1, 5))
                # Track losses and acc for per epoch plotting
                val_loss_total.update(losssup_total_val.item(), batch_imgs.size(0))
                val_top1.update(acc1.item(), batch_imgs.size(0))
                val_top5.update(acc5.item(), batch_imgs.size(0))
        
        if args.ddp:
            val_loss_total.all_reduce()
            val_top1.all_reduce()
            val_top5.all_reduce()

        if args.local_rank == 0:
            writer.add_scalar('Loss Supervised Validation (per epoch)', val_loss_total.avg, epoch)
            writer.add_scalar('Accuracy Validation (per epoch)', val_top1.avg, epoch)
            writer.add_scalar('Accuracy Validation Top5 (per epoch)', val_top5.avg, epoch)
            print(f'Epoch [{epoch}] Val Loss Total: {val_loss_total.avg:.6f} -- Val Top1: {val_top1.avg:.2f} -- Val Top5: {val_top5.avg:.2f}')

        if args.ddp:
            torch.distributed.barrier() # Wait for all processes to finish the validation epoch

        ### Save model ###
        if (args.local_rank == 0) and (((epoch+1) % 10) == 0) or epoch==0:
            if args.ddp:
                view_encoder_state_dict = view_encoder.module.state_dict()
                classifier_state_dict = classifier.module.state_dict()
                cond_generator_state_dict = cond_generator.module.state_dict()
            else:
                view_encoder_state_dict = view_encoder.state_dict()
                classifier_state_dict = classifier.state_dict()
                cond_generator_state_dict = cond_generator.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(classifier_state_dict, os.path.join(args.save_dir, f'classifier_epoch{epoch}.pth'))
            torch.save(cond_generator_state_dict, os.path.join(args.save_dir, f'cond_generator_epoch{epoch}.pth'))

        ### Plot reconstructions examples ###
        if args.local_rank == 0:
            if (epoch+1) % 5 == 0 or epoch==0:
                view_encoder.eval()
                cond_generator.eval()
                classifier.eval()
                n = 16
                episodes_plot_imgs = episodes_plot[0][:n].to(device, non_blocking=True)
                episodes_plot_actions = episodes_plot[1][:n]
                episodes_plot_gen_imgs = torch.empty(0)
                with torch.no_grad():
                    first_view_tensors = view_encoder(episodes_plot_imgs[:,0])[:, 1:, :] # Discard the CLS token. Shape is (B, T, D)
                    for v in range(args.num_views):
                        actions = [episodes_plot_actions[j][v] for j in range(episodes_plot_imgs.shape[0])]
                        gen_images, _ = cond_generator(first_view_tensors, actions)
                        episodes_plot_gen_imgs = torch.cat([episodes_plot_gen_imgs, gen_images.unsqueeze(1).detach().cpu()], dim=1)
                episodes_plot_imgs = episodes_plot_imgs.detach().cpu()
                # plot each episode
                for i in range(n):
                    episode_i_imgs = episodes_plot_imgs[i]
                    episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_imgs]
                    episode_i_imgs = torch.stack(episode_i_imgs, dim=0)

                    episode_i_gen_imgs = episodes_plot_gen_imgs[i]
                    episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_gen_imgs]
                    episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0)
                    episode_i_gen_imgs = torch.clamp(episode_i_gen_imgs, 0, 1) # Clip values to [0, 1]

                    grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs], dim=0), nrow=args.num_views)
                    grid = grid.permute(1, 2, 0).cpu().numpy()
                    grid = (grid * 255).astype(np.uint8)
                    grid = Image.fromarray(grid)
                    image_name = f'epoch{epoch}_episode{i}.png'
                    save_plot_dir = os.path.join(args.save_dir, 'gen_plots')
                    # create folder if it doesn't exist
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    grid.save(os.path.join(save_plot_dir, image_name))

        if args.local_rank == 0:
            epoch_time = time.time() - start_time
            elapsed_time = time.time() - init_time
            print(f"Epoch [{epoch}] Epoch Time: {time_duration_print(epoch_time)} -- Elapsed Time: {time_duration_print(elapsed_time)}")

    # Close tensorboard writer
    if args.is_main:
        writer.close()

    if args.ddp:
        torch.distributed.barrier()  # Wait for all processes to finish the validation step
        torch.distributed.destroy_process_group()  # Destroy the process group

    return None

if __name__ == '__main__':
    main()