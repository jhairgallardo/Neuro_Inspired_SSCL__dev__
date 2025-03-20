import argparse
import os, time

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

# from models_encGNMish_UCL import *
from models_encBNMish import *
# from models import *
from loss_functions import KoLeoLossViewExpanded
from augmentations import Episode_Transformations

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Episodic offline')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Network parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--pred_model_name', type=str, default='ConditionedPredictor')
parser.add_argument('--pred_hiddendim', type=int, default=512)
parser.add_argument('--pred_num_layers', type=int, default=3)
parser.add_argument('--pred_nhead', type=int, default=4)
parser.add_argument('--pred_dim_feedforward', type=int, default=2048) # it should always be hidden_dim * 4
parser.add_argument('--pred_dropout', type=float, default=0.1)
parser.add_argument('--pred_sequence_length', type=int, default=50) # 1 action token + 49 feature tokens (7x7)
parser.add_argument('--corruption_prob', type=float, default=0.75)
parser.add_argument('--action_code_dim', type=int, default=11)
# Training parameters
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--episode_batch_size', type=int, default=192)
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/run_encoder_pretraining")
parser.add_argument('--print_frequency', type=int, default=10) # batch iterations
parser.add_argument('--seed', type=int, default=0)

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

def main():

    ### Parse arguments
    args = parser.parse_args()
    # args.lr = args.lr * args.episode_batch_size / 128
    print(args)
    if not os.path.exists(args.save_dir): # create save dir
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Seed everything
    seed_everything(seed=args.seed)

    ### Define tensoboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load data
    print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=True)
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.mean, std=args.std),
                        ])
    train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
    val_dataset = datasets.ImageFolder(valdir, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers, pin_memory = True, persistent_workers=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.episode_batch_size, shuffle = False, num_workers = args.workers, pin_memory = True)

    ## Load training data for KNN tracking
    train_knn_dataset = datasets.ImageFolder(traindir, transform=val_transform)
    train_knn_dataloader = torch.utils.data.DataLoader(train_knn_dataset, batch_size = args.episode_batch_size, shuffle = False, num_workers = args.workers)

    ### Load models
    print('\n==> Preparing network...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    predictor_net = eval(args.pred_model_name)(input_dim=view_encoder.fc.weight.shape[1],
                                               hidden_dim=args.pred_hiddendim,
                                               output_dim=view_encoder.fc.weight.shape[1],
                                               action_code_dim=args.action_code_dim,
                                               num_layers=args.pred_num_layers,
                                               nhead=args.pred_nhead,
                                               dim_feedforward=args.pred_dim_feedforward,
                                               dropout=args.pred_dropout,
                                               sequence_length=args.pred_sequence_length)
    view_encoder.fc = torch.nn.Identity()

    ### Print models
    print('\nView encoder')
    print(view_encoder)
    print('\nPredictor')
    print(predictor_net)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    predictor_net = torch.nn.DataParallel(predictor_net)
    view_encoder = view_encoder.to(device)
    predictor_net = predictor_net.to(device)

    ### KNN eval before training
    print('\n==> KNN evaluation before training')
    knn_val = knn_eval(train_knn_dataloader, val_loader, view_encoder, device, k=10, num_classes=args.num_classes, save_dir=args.save_dir, epoch=-1)
    print(f'KNN accuracy: {knn_val}')
    writer.add_scalar('KNN_accuracy_validation', knn_val, -1)

    ### Load optimizer and criterion
    param_groups = [{'params': view_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': predictor_net.parameters(), 'lr': args.lr, 'weight_decay': args.wd}]#, 'fix_lr': True}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
    criterion_mse = torch.nn.MSELoss()
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr*1e-6, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    #### Train loop ####
    print('\n==> Training model')
    init_time = time.time()
    scaler = GradScaler()
    for epoch in range(args.epochs):
        print(f'\n==> Epoch {epoch}/{args.epochs}')
        start_time = time.time()

        ## Train STEP ##
        total_loss=0
        view_encoder.train()
        predictor_net.train()
        for i, (batch_episodes, _) in enumerate(train_loader):

            # Forward pass
            batch_episodes_imgs = batch_episodes[0].to(device)
            batch_episodes_actions = batch_episodes[1].to(device)

            batch_episodes_outputs = torch.empty(0).to(device)
            batch_episodes_outputs_noaction = torch.empty(0).to(device)
            batch_episodes_targets = torch.empty(0).to(device)

            batch_first_view_tensors = view_encoder(batch_episodes_imgs[:,0]) # h1
            batch_noaction = batch_episodes_actions[:,0] # no action is in the first view
            # for v in range(1, args.num_views):
            for t in range(args.num_views-1):
                batch_imgs = batch_episodes_imgs[:,t+1] # V{t+1}
                batch_actions = batch_episodes_actions[:,t+1] # a{t}

                ### Forward pass
                with autocast():
                
                    # Predictor forward pass wit action codes
                    batch_outputs = predictor_net(batch_first_view_tensors, batch_actions) # Q(h1,a{t}) -> hhat{t+1}

                    # Predictor forward pass with no action and corruption
                    batch_tensors_corrupted = corrupt_batch_tensors(view_encoder(batch_imgs), p=args.corruption_prob) # Corrupt( F(V{t+1}) ) -> ~h{t+1}
                    batch_outputs_noaction = predictor_net(batch_tensors_corrupted, batch_noaction) # Q{~h{t+1}, a{0}} -> ~hhat{t+1}

                    # View encoder forward pass (get targets)
                    batch_targets = view_encoder(batch_imgs) # F(V{t+1}) -> h{t+1}

                # collect outputs
                batch_episodes_outputs = torch.cat([batch_episodes_outputs, batch_outputs.unsqueeze(1)], dim=1)
                batch_episodes_outputs_noaction = torch.cat([batch_episodes_outputs_noaction, batch_outputs_noaction.unsqueeze(1)], dim=1)
                batch_episodes_targets = torch.cat([batch_episodes_targets, batch_targets.unsqueeze(1)], dim=1)
                 
            loss_rep = criterion_mse(batch_episodes_outputs, batch_episodes_targets.detach())
            loss_rep_noaction = criterion_mse(batch_episodes_outputs_noaction, batch_episodes_targets.detach())

            # loss = loss_rep
            loss = loss_rep + loss_rep_noaction

            # Backward pass
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

            # Update metrics
            total_loss += loss.item()
            if i % args.print_frequency == 0:
                print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                      f'lr: {scheduler.get_last_lr()[0]:.6f} -- ' +
                      f'Loss rep: {loss_rep.item():.6f} -- ' +
                      f'Loss rep no action: {loss_rep_noaction.item():.6f} -- ' +
                      f'Loss Total: {loss.item():.6f}'
                    )
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch*len(train_loader)+i)
            writer.add_scalar('Loss Rep', loss_rep.item(), epoch*len(train_loader)+i)
            writer.add_scalar('Loss Rep No Action', loss_rep_noaction.item(), epoch*len(train_loader)+i)          
            writer.add_scalar('Loss Total', loss.item(), epoch*len(train_loader)+i)
            
            # Update lr scheduler
            scheduler.step()
        
        # Epoch metrics
        total_loss /= len(train_loader)
        writer.add_scalar('Loss Total (per epoch)', total_loss, epoch)
        print(f'Epoch [{epoch}] Total Train Loss per Epoch: {total_loss:.6f}')

        ## KNN eval ##
        knn_val = knn_eval(train_knn_dataloader, val_loader, view_encoder, device, k=10, num_classes=args.num_classes, save_dir=args.save_dir, epoch=epoch)
        print(f'Epoch [{epoch}] KNN accuracy - validation: {knn_val}')
        writer.add_scalar('KNN_accuracy_validation', knn_val, epoch)

        ## Save model ##
        if (epoch+1) % 10 == 0 or epoch==0:
            view_encoder_state_dict = view_encoder.module.state_dict()
            predictor_net_state_dict = predictor_net.module.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(predictor_net_state_dict, os.path.join(args.save_dir, f'predictor_net_epoch{epoch}.pth'))

        print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    return None

def corrupt_batch_tensors(tensors, p=0.75):
    """
    Corrupt feature maps by randomly zeroing out complete channel vectors at spatial locations.

    Args:
        tensors (torch.Tensor): Input tensor of shape (B, C, H, W).
        p (float): Probability of zeroing out a spatial location (default is 0.75).

    Returns:
        torch.Tensor: Corrupted tensor.
    """
    # Generate a mask with shape (B, H, W): 1 means keep, 0 means zero out.
    # With probability 'p' we want to zero out, so we keep if random value >= p.
    mask = (torch.rand(tensors.shape[0], tensors.shape[2], tensors.shape[3], device=tensors.device) >= p).float()
    # Add a channel dimension to match tensors: shape becomes (B, 1, H, W).
    mask = mask.unsqueeze(1)
    # Multiply the input tensor by the mask. Broadcasting will zero out the entire channel vector where needed.
    return tensors * mask

def plot_feature_space(features, labels, num_classes, title, save_dir):
    """
    Plot feature space using PCA and t-SNE.
    """
    # PCA
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features.cpu().numpy())
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
    features_tsne = tsne.fit_transform(features_pca)
    # Plot
    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], label=f'Class {i}')
    plt.title(title)
    plt.legend()
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()
    return None

def plot_feature_space_PCA(features, labels, num_classes, title, save_dir):
    """
    Plot feature space using PCA and t-SNE.
    """
    # PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features.cpu().numpy())
    # Plot
    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        plt.scatter(features_pca[labels == i, 0], features_pca[labels == i, 1], label=f'Class {i}')
    plt.title(title)
    plt.legend()
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()
    return None


def knn_eval(train_loader, val_loader, view_encoder, device, k, num_classes, save_dir, epoch):
    view_encoder.eval()

    ### Get train features and labels
    train_features = []
    train_labels = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            features = view_encoder(imgs)
            # global average pooling
            features = torch.mean(features, dim=(2,3))
            # flattening
            features = torch.flatten(features, 1)
            train_features.append(features)
            train_labels.append(labels)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    ### Get val features and labels
    val_features = []
    val_labels = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            features = view_encoder(imgs)
            features = torch.mean(features, dim=(2,3))
            features = torch.flatten(features, 1)
            val_features.append(features)
            val_labels.append(labels)
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    if epoch == -1 or epoch == 0 or (epoch+1) % 5 == 0 or epoch==99:
        # plot feature space (Applying PCA and t-SNE)
        # plot_feature_space(train_features, train_labels, num_classes, title='Train Feature Space', save_dir=os.path.join(save_dir, 'train_feature_space.png'))
        plot_feature_space(val_features, val_labels, num_classes, title=f'Validation Feature Space PCA+TSNE Epoch {epoch}', save_dir=os.path.join(save_dir, f'val_feature_space_TSNE_{epoch}.png'))
        plot_feature_space_PCA(val_features, val_labels, num_classes, title=f'Validation Feature Space PCA Epoch {epoch}', save_dir=os.path.join(save_dir, f'val_feature_space_PCA_{epoch}.png'))

    ### KNN
    top1 = knn_classifier(train_features, train_labels, val_features, val_labels, num_classes=num_classes, k=k)

    return top1

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, num_classes, k=20, batch_size=256):
    """
    KNN classification in batches to reduce memory usage.
    """
    device = train_features.device
    num_test = test_features.size(0)
    predictions = []
    for start_idx in range(0, num_test, batch_size):
        end_idx = min(start_idx + batch_size, num_test)
        test_batch = test_features[start_idx:end_idx]  # shape: (B, D)
        distances = torch.cdist(test_batch, train_features, p=2)  # shape: (B, N)
        _, knn_indices = distances.topk(k, dim=1, largest=False)
        nn_labels = train_labels[knn_indices]
        batch_preds = []
        for row in nn_labels:
            counts = row.bincount(minlength=num_classes)
            pred_label = torch.argmax(counts)
            batch_preds.append(pred_label)
        batch_preds = torch.stack(batch_preds)
        predictions.append(batch_preds)
    predictions = torch.cat(predictions, dim=0)  # shape: (num_test,)
    correct = (predictions == test_labels).sum().item()
    accuracy = correct / num_test
    return accuracy

if __name__ == '__main__':
    main()