import argparse
import os, time

import torch
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from models_encBNMish_zca_infodrop import *
from loss_functions import CrossEntropyViewExpanded, KoLeoLossViewExpanded
from augmentations import Episode_Transformations
from function_zca import calculate_ZCA_weights, scaled_filters

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

parser = argparse.ArgumentParser(description='View Encoder Pretraining - Supervised Episodic offline with ZCA with InfoDrop')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-5-random')
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# Network parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--classifier_model_name', type=str, default='Classifier_Model')
# ZCA parameters
parser.add_argument('--zca_channels', type=int, default=3)
parser.add_argument('--zca_kernel_size', type=int, default=3)
parser.add_argument('--zca_eps', type=float, default=1e-4)
parser.add_argument('--zca_init_imgs', type=int, default=10000)
# InfoDrop parameters
parser.add_argument('--infodrop_blocks', type=float, default=0.5)
# Training parameters
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=6)
parser.add_argument('--koleo_gamma', type=float, default=0)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--save_dir', type=str, default="output/run_encoder_offline")
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
    args.lr = args.lr * args.episode_batch_size / 128
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
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=False)
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
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool=True, infodrop_blocks=args.infodrop_blocks)
    classifier = eval(args.classifier_model_name)(input_dim=view_encoder.fc.weight.shape[1], num_classes=args.num_classes)
    zca_layer = eval("ZCA_layer")(in_channels=3, out_channels=args.zca_channels, kernel_size=args.zca_kernel_size, epsilon=args.zca_eps)
    view_encoder.fc = torch.nn.Identity() # remove last layer
                                                  
    ### Print models
    print('\n==> ZCA layer')
    print(zca_layer)
    print('\nView encoder')
    print(view_encoder)
    print('\nClassifier')
    print(classifier)
    print('\n')

    ### Calculate weights of ZCA layer
    print('\n==> Calculating ZCA weights...')
    train_dataset_for_zcaInit = datasets.ImageFolder(traindir, transform=val_transform)
    train_dataloader_for_zcaInit = torch.utils.data.DataLoader(train_dataset_for_zcaInit, batch_size = args.zca_init_imgs, shuffle = True)
    imgs_to_init_zca,_ = next(iter(train_dataloader_for_zcaInit))
    zca_layer.init_ZCA_layer_weights(imgs_to_init_zca)
    for param in zca_layer.parameters(): 
        param.requires_grad = False # freeze ZCA layer
    print(f'Actual number of images used to Initialized ZCA layer: {imgs_to_init_zca.size(0)}')
    print(f'ZCA layer weights calculated and stored in the model')
    del train_dataset_for_zcaInit, train_dataloader_for_zcaInit, imgs_to_init_zca
    # Save zca layer
    torch.save(zca_layer.state_dict(), os.path.join(args.save_dir, 'zca_layer.pth'))

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    classifier = torch.nn.DataParallel(classifier)
    zca_layer = torch.nn.DataParallel(zca_layer)
    view_encoder = view_encoder.to(device)
    classifier = classifier.to(device)
    zca_layer = zca_layer.to(device)

    ### KNN eval before training
    print('\n==> KNN evaluation before training')
    knn_val = knn_eval(train_knn_dataloader, val_loader, [zca_layer, view_encoder], device, k=10, num_classes=args.num_classes, save_dir=args.save_dir, epoch=-1)
    print(f'KNN accuracy: {knn_val}')
    writer.add_scalar('KNN_accuracy_validation', knn_val, -1)

    ### Load optimizer and criterion
    param_groups = [{'params': view_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': classifier.parameters(), 'lr': args.lr, 'weight_decay': args.wd}]
    optimizer = torch.optim.AdamW(param_groups, lr=0, weight_decay=0)
    criterion_sup = CrossEntropyViewExpanded(num_views=args.num_views).to(device)
    criterion_koleo = KoLeoLossViewExpanded(num_views=args.num_views).to(device)
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
        zca_layer.eval()
        view_encoder.train()
        classifier.train()
        for i, (batch_episodes_imgs, batch_labels) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes_imgs.to(device) # (B, V, C, H, W)
            batch_episodes_labels = batch_labels.unsqueeze(1).repeat(1, batch_episodes_imgs.size(1)).to(device) # (B, V)

            # Forward pass
            batch_episodes_tensors = torch.empty(0).to(device) # (B, V, c, h, w)
            batch_episodes_logits = torch.empty(0).to(device) # (B, V, num_class)
            for v in range(args.num_views):
                batch_imgs = batch_episodes_imgs[:,v]
                with autocast():
                    batch_tensors = view_encoder(zca_layer(batch_imgs))
                    batch_logits = classifier(batch_tensors)
                batch_episodes_tensors = torch.cat([batch_episodes_tensors, batch_tensors.unsqueeze(1)], dim=1)
                batch_episodes_logits = torch.cat([batch_episodes_logits, batch_logits.unsqueeze(1)], dim=1)
            loss_sup = criterion_sup(batch_episodes_logits, batch_episodes_labels)
            if args.koleo_gamma != 0: loss_koleo = criterion_koleo(batch_episodes_tensors.mean(dim=(3,4))) # pass the average pooled version (koleo works on vectors) 
            else: loss_koleo = torch.tensor(0).to(device)
            loss = loss_sup + args.koleo_gamma*loss_koleo

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            # sanity check (check that zca layer is not updated)
            for name, param in zca_layer.named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0)
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            # Update metrics
            total_loss += loss.item()
            if i % args.print_frequency == 0:
                print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- ' +
                      f'lr: {scheduler.get_last_lr()[0]:.6f} -- ' +
                      f'Loss Sup: {loss_sup.item():.6f} -- ' +
                      f'{f"Loss Koleo: {loss_koleo.item():.6f}" if args.koleo_gamma > 0 else ""} -- ' +
                      f'Loss Total: {loss.item():.6f}'
                    )
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch*len(train_loader)+i)
            writer.add_scalar('Loss Supervised', loss_sup.item(), epoch*len(train_loader)+i)
            if args.koleo_gamma > 0: writer.add_scalar('Loss Koleo', loss_koleo.item(), epoch*len(train_loader)+i)            
            writer.add_scalar('Loss Total', loss.item(), epoch*len(train_loader)+i)
            
            # Update lr scheduler
            scheduler.step()

        # Train Epoch metrics
        total_loss /= len(train_loader)
        writer.add_scalar('Loss Total (per epoch)', total_loss, epoch)
        print(f'Epoch [{epoch}] Total Train Loss per Epoch: {total_loss:.6f}')


        ## Validation STEP ##
        zca_layer.eval()
        view_encoder.eval()
        classifier.eval()
        total_loss=0
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(val_loader):
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                with autocast():
                    batch_tensors = view_encoder(zca_layer(batch_imgs))
                    batch_logits = classifier(batch_tensors)
                loss = criterion_sup.crossentropy(batch_logits, batch_labels)
                total_loss += loss.item()
                # Calculate accuracy
                _, predicted = batch_logits.max(1)
                correct = predicted.eq(batch_labels).sum().item()
                # accumulate for the whole validation run
                if i == 0:
                    total_correct = correct
                    total_samples = batch_labels.size(0)
                else:
                    total_correct += correct
                    total_samples += batch_labels.size(0)
        accuracy = total_correct / total_samples

        # Validation Epoch metrics
        total_loss /= len(val_loader)
        writer.add_scalar('Loss Total Validation (per epoch)', total_loss, epoch)
        print(f'Epoch [{epoch}] Total Validation Loss per Epoch: {total_loss:.6f}, Accuracy: {accuracy:.6f}')

        
        ## KNN eval ##
        knn_val = knn_eval(train_knn_dataloader, val_loader, [zca_layer, view_encoder], device, k=10, num_classes=args.num_classes, save_dir=args.save_dir, epoch=epoch)
        print(f'Epoch [{epoch}] KNN accuracy - validation: {knn_val}')
        writer.add_scalar('KNN_accuracy_validation', knn_val, epoch)

        ## Save model ##
        if (epoch+1) % 10 == 0 or epoch==0:
            view_encoder_state_dict = view_encoder.module.state_dict()
            classifier_state_dict = classifier.module.state_dict()
            torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_epoch{epoch}.pth'))
            torch.save(classifier_state_dict, os.path.join(args.save_dir, f'classifier_epoch{epoch}.pth'))

        print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    return None

def knn_eval(train_loader, val_loader, models, device, k, num_classes, save_dir, epoch):
    zca_layer = models[0]
    view_encoder = models[1]

    zca_layer.eval()
    view_encoder.eval()

    ### Get train features and labels
    train_features = []
    train_labels = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            with autocast():
                features = view_encoder(zca_layer(imgs))
            features = torch.mean(features, dim=(2,3))
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
            with autocast():
                features = view_encoder(zca_layer(imgs))
            features = torch.mean(features, dim=(2,3))
            features = torch.flatten(features, 1)
            val_features.append(features)
            val_labels.append(labels)
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

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