import argparse
import os, time
import random
import warnings
import einops
import json

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.optim import lr_scheduler

from resnet_gn_mish import *
from evaluate_cluster import evaluate as eval_pred

from PIL import ImageFilter
from tqdm import tqdm


from PIL import Image, ImageOps, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from tensorboardX import SummaryWriter

from function_zca import calculate_ZCA_conv0_weights
import numpy as np

def main(args, device, writer):

    print('\n==> Preparing data...')

    ### Load data
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    transform = Transformations(num_views=args.num_views, aug_type=args.aug_type, zca=args.zca)
    train_dataset = datasets.ImageFolder(traindir, transform=transform)
    val_dataset = datasets.ImageFolder(valdir, transform=transform.no_aug)
    train_dataset = Datasetwithindex(train_dataset)
    val_dataset = Datasetwithindex(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                            shuffle=True, num_workers=args.workers, pin_memory=True,
                                            persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=args.workers, pin_memory=True)
    
    print('\n==> Building and loading model')

    ### Load model
    encoder = eval(args.model_name)(num_classes=100, zero_init_residual=args.zero_init_res, conv0_flag=args.zca)
    if args.zca:
        print('\n      Calculating ZCA layer ...')
        zca_dataset = datasets.ImageFolder(traindir, transform=transform.aug)
        weight, bias = calculate_ZCA_conv0_weights(model = encoder, dataset = zca_dataset,
                                            addgray = True, save_dir = args.save_dir,
                                            nimg = 10000, zca_epsilon=5e-4)
        encoder.conv0.weight = torch.nn.Parameter(weight)
        encoder.conv0.bias = torch.nn.Parameter(bias)
    for param in encoder.conv0.parameters(): # freeze conv0 layer
        param.requires_grad = False
    if args.pretrained_model is not None:
        encoder.load_state_dict(torch.load(args.pretrained_model), strict=True)
        for param in encoder.parameters(): # freeze pretrained encoder
            param.requires_grad = False
    model = SSL_epmodel(encoder, args.num_pseudoclasses)
    if args.dp:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    print('\n==> Setting optimizer and scheduler')

    ### Load optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = EntLoss(num_views=args.num_views, downweight=args.loss_downweight, gamma=args.gamma, anchor=args.anchor_based_loss).to(device)
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.lr*1e-6, total_iters=args.warmup_epochs*len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs)*len(train_loader), eta_min=args.lr*0.001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs*len(train_loader)])

    print('\n==> Validation on random model')
    val_metrics = validate(args, model, val_loader, device, writer, init=True)
    nmi = val_metrics['NMI']
    ami = val_metrics['AMI']
    ari = val_metrics['ARI']
    fscore = val_metrics['F']
    adjacc = val_metrics['ACC']
    top5 = val_metrics['ACC-Top5']
    print(f'Validation -- NMI: {nmi:.4f} -- AMI: {ami:.4f} -- ARI: {ari:.4f} -- F: {fscore:.4f} -- ACC: {adjacc:.4f} -- ACC-Top5: {top5:.4f}')

    # Save model at random init
    if args.dp: state_dict = model.module.state_dict()
    else: state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(args.save_dir, f'episodicSSL_model_init.pth'))
    # save encoder
    if args.dp: encoder_state_dict = model.module.encoder.state_dict()
    else: encoder_state_dict = model.encoder.state_dict()
    torch.save(encoder_state_dict, os.path.join(args.save_dir, f'encoder_init.pth'))

    print('\n==> Training model')

    ### Train model
    # save rand init model
    if args.dp: state_dict = model.module.state_dict()
    else: state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(args.save_dir, f'episodicSSL_model_init.pth'))
    # train
    init_time = time.time()
    train_loss_all = []
    nmi_all = []
    ACC_all = []
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_step(args, model, train_loader, optimizer, criterion, scheduler, epoch, device, writer)
        val_metrics = validate(args, model, val_loader, device, writer)

        # TODO: knn tracker on every epoch to test representations

        # Print results
        nmi = val_metrics['NMI']
        ami = val_metrics['AMI']
        ari = val_metrics['ARI']
        fscore = val_metrics['F']
        adjacc = val_metrics['ACC']
        top5 = val_metrics['ACC-Top5']
        print(f'Epoch [{epoch}] Total Train Loss per Epoch: {train_loss:.6f}')
        print(f'Epoch [{epoch}] Epoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')
        print(f'NMI: {nmi:.4f}, AMI: {ami:.4f}, ARI: {ari:.4f}, F: {fscore:.4f}, ACC: {adjacc:.4f}, ACC-Top5: {top5:.4f}')
        
        # Save model per epoch
        if args.dp: state_dict = model.module.state_dict()
        else: state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(args.save_dir, f'episodicSSL_model_epoch{epoch}.pth'))
        # save encoder
        if args.dp: encoder_state_dict = model.module.encoder.state_dict()
        else: encoder_state_dict = model.encoder.state_dict()
        torch.save(encoder_state_dict, os.path.join(args.save_dir, f'encoder_epoch{epoch}.pth'))

        # Add to tensorboard
        writer.add_scalar('Total Loss per epoch', train_loss, epoch)
        writer.add_scalar('Metric NMI', nmi, epoch)
        writer.add_scalar('Metric AMI', ami, epoch)
        writer.add_scalar('Metric ARI', ari, epoch)
        writer.add_scalar('Metric F', fscore, epoch)
        writer.add_scalar('Metric ACC', adjacc, epoch)
        writer.add_scalar('Metric ACC-Top5', top5, epoch)

        # Save results
        train_loss_all.append(train_loss)
        nmi_all.append(nmi)
        ACC_all.append(adjacc)
        np.save(os.path.join(args.save_dir, 'train_loss.npy'), np.array(train_loss_all))
        np.save(os.path.join(args.save_dir, 'nmi.npy'), np.array(nmi_all))
        np.save(os.path.join(args.save_dir, 'ACC.npy'), np.array(ACC_all))

        if epoch+1 == args.stop_epoch:
            break

    print('\n==> Training finished')

    return None
        

def train_step(args, model, train_loader, optimizer, criterion, scheduler, epoch, device, writer=None):
    model.train()
    total_loss = 0
    all_SKL = []
    max_SKL = 0
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        # forward pass
        episodes_images = batch[0].to(device)
        X = einops.rearrange(episodes_images, 'b v c h w -> (b v) c h w').contiguous() # all episodes views in one batch
        episodes_logits = model(X)
        consis_loss, sharp_loss, div_loss, SKL = criterion(episodes_logits)
        loss = consis_loss + args.lam1*sharp_loss - args.lam2*div_loss
        # backward pass
        loss.backward()
        optimizer.step()
        # accumulate loss
        total_loss += loss.item()
        # print each loss and total loss
        if i % 10 == 0:
            print(f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- lr: {scheduler.get_last_lr()[0]:.6f} -- Consis: {consis_loss.item():.6f}' + 
                  f' -- Sharp: {sharp_loss.item():.6f} -- Div: {div_loss.item():.6f} -- Total: {loss.item():.6f}')
            if writer is not None:
                writer.add_scalar('Consistency Loss', consis_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Sharpness Loss', sharp_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Diversity Loss', div_loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Total Loss', loss.item(), epoch*len(train_loader)+i)
        scheduler.step()
        if i<100: 
            batch_SKL = SKL.cpu().detach()
            all_SKL.append(batch_SKL)
            if max_SKL < torch.max(batch_SKL):
                max_SKL = torch.max(batch_SKL)
                max_index_in_batch = torch.argmax(batch_SKL)
                if args.anchor_based_loss:
                    views_maxskl = batch[0][max_index_in_batch][[0,0,-1]].cpu().detach() # original, view, view
                else:
                    views_maxskl = batch[0][max_index_in_batch][[0,-2,-1]].cpu().detach()# original, view, view

    total_loss /= len(train_loader)

    all_SKL = torch.stack(all_SKL, dim=0).view(-1).numpy()
    np.save(os.path.join(args.save_dir, f'SKL_epoch{epoch}.npy'), all_SKL)

    # plot views with max SKL
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    if args.zca: std = [1.0, 1.0, 1.0]
    views_maxskl = [transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in views_maxskl]
    views_maxskl = torch.stack(views_maxskl, dim=0)
    grid = torchvision.utils.make_grid(views_maxskl, nrow=3)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = (grid * 255).astype(np.uint8)
    grid = Image.fromarray(grid)
    grid.save(os.path.join(args.save_dir, f'views_epoch{epoch}maxSKL{max_SKL:.4f}.png'))
    
    return total_loss

def validate(args, model, val_loader, device, writer=None, init=False):
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_preds = []
        all_probs = []
        all_indices = []
        for i, batch in enumerate(val_loader):
            imgs = batch[0].to(device)
            labels = batch[1].to(device)
            img_index = batch[2]
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_indices.append(img_index)
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_indices = torch.cat(all_indices).numpy()
    nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(all_labels.astype(int), all_preds.astype(int), calc_acc=(args.num_pseudoclasses==100), total_probs=all_probs)

    if writer is not None:
        os.makedirs(os.path.join(args.save_dir, f'pseudo_classes_clusters'), exist_ok=True)
        # plot 25 images per cluster
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if args.zca: std = [1.0, 1.0, 1.0]
        for i in range(8): # only for 8 pseudo classes
            pseudoclass_imgs_indices = all_indices[all_preds==i]
            if len(pseudoclass_imgs_indices) > 0:
                pseudoclass_imgs_indices = pseudoclass_imgs_indices[:25]
                pseudoclass_imgs = [val_loader.dataset.data[j][0] for j in pseudoclass_imgs_indices]
                # psudoclass images are the output of the transform already. So we need to reverse the normalization
                pseudoclass_imgs = [transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in pseudoclass_imgs]
                # stack images
                pseudoclass_imgs = torch.stack(pseudoclass_imgs, dim=0)
                # plot a grid of images 5x5
                grid = torchvision.utils.make_grid(pseudoclass_imgs, nrow=5)
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                grid = Image.fromarray(grid)
            else: # Save a black image
                grid = Image.new('RGB', (224*5, 224*5), (0, 0, 0))
            image_name = f'pseudoclass_{i}.png'
            if init: image_name = f'pseudoclass_{i}_init.png'
            grid.save(os.path.join(args.save_dir, f'pseudo_classes_clusters', image_name))

    return {'NMI': nmi, 'AMI': ami, 'ARI': ari, 'F': fscore, 'ACC': adjacc, 'ACC-Top5': top5}
    

class EntLoss(nn.Module):
    def __init__(self, num_views=4, tau=1, eps=1e-5, gamma=1, downweight=False, anchor=False):
        super(EntLoss, self).__init__()
        self.eps = eps
        self.tau = tau
        self.N = num_views
        self.gamma = gamma
        self.downweight = downweight
        self.anchor = anchor

    def forward(self, episodes_logits):
        episodes_probs = F.softmax(episodes_logits, dim=1)
        episodes_probs = einops.rearrange(episodes_probs, '(b v) c -> b v c', v=self.N).contiguous()
        episodes_sharp_probs = F.softmax(episodes_logits/self.tau, dim=1)
        episodes_sharp_probs = einops.rearrange(episodes_sharp_probs, '(b v) c -> b v c', v=self.N).contiguous()
        B = episodes_probs.size(0)

        consis_loss = 0
        sharp_loss = 0
        div_loss = 0

        if self.downweight: # loss with downweighting
            w_sum=0
            for t in range(self.N):
                if t < self.N-1:
                    if self.anchor:
                        SKL = 0.5 * (self.KL(episodes_probs[:,0], episodes_probs[:,t+1]) + self.KL(episodes_probs[:,t+1], episodes_probs[:,0])) # Simetrized KL
                    else:
                        SKL = 0.5 * (self.KL(episodes_probs[:,t], episodes_probs[:,t+1]) + self.KL(episodes_probs[:,t+1], episodes_probs[:,t])) # Simetrized KL
                    w = torch.exp(-self.gamma * SKL) # **2 = downweightingV2 , **4 = downweightingV3
                    consis_loss += SKL * w
                    w_sum += w
                sharp_loss += self.entropy(episodes_sharp_probs[:,t]).mean() #### Sharpening loss
                mean_across_episodes = episodes_sharp_probs[:,t].mean(dim=0)
                div_loss += self.entropy(mean_across_episodes, dim=0) #### Diversity loss
            consis_loss = consis_loss / w_sum # mean over views with downweighting
            consis_loss = consis_loss.mean() # mean over episodes
            sharp_loss = sharp_loss / self.N
            div_loss = div_loss / self.N

        else: # loss without downweighting
            for t in range(self.N):
                if t < self.N-1:
                    if self.anchor:
                        SKL = 0.5 * (self.KL(episodes_probs[:,0], episodes_probs[:,t+1]) + self.KL(episodes_probs[:,t+1], episodes_probs[:,0])) # Simetrized KL
                    else:
                        SKL = 0.5 * (self.KL(episodes_probs[:,t], episodes_probs[:,t+1]) + self.KL(episodes_probs[:,t+1], episodes_probs[:,t])) # Simetrized KL
                    consis_loss += SKL
                sharp_loss += self.entropy(episodes_sharp_probs[:,t]).mean() #### Sharpening loss
                mean_across_episodes = episodes_sharp_probs[:,t].mean(dim=0)
                div_loss += self.entropy(mean_across_episodes, dim=0) #### Diversity loss
            consis_loss = consis_loss / (self.N-1) # mean over views without downweighting
            consis_loss = consis_loss.mean() # mean over episodes
            sharp_loss = sharp_loss / self.N
            div_loss = div_loss / self.N

        return consis_loss, sharp_loss, div_loss, SKL

    def KL(self, probs1, probs2, eps = 1e-5):
        kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum(dim=1)
        return kl

    def entropy(self, probs, eps = 1e-5, dim=1):
        H = - (probs * (probs + eps).log()).sum(dim=dim)
        return H

class SSL_epmodel(torch.nn.Module):
    def __init__(self, encoder, num_pseudoclasses):
        super().__init__()
        self.encoder = encoder
        features_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = torch.nn.Identity()
        # Projector (R) 
        self.projector = nn.Sequential(
            nn.Linear(features_dim, features_dim*4),
            nn.GroupNorm(32, features_dim*4),
            nn.Mish(),
            nn.Linear(features_dim*4, features_dim*4),
            nn.GroupNorm(32, features_dim*4),
            nn.Mish()
        )
        # Linear head (F)
        self.linear_head = nn.Linear(features_dim*4, num_pseudoclasses, bias=True)
        self.norm = nn.BatchNorm1d(num_pseudoclasses, affine=False)

    def forward(self, x):
        # encoder
        x = self.encoder(x)
        x = self.projector(x)
        x = self.linear_head(x)
        x = self.norm(x)
        return x

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Transformations:
    def __init__(self, num_views, aug_type='all', zca=False):
        self.num_views = num_views
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if zca: std = [1.0, 1.0, 1.0]
        self.no_aug = transforms.Compose([
                transforms.Resize(256),# interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ])
        if aug_type == 'all':
            self.aug = transforms.Compose([
                transforms.RandomResizedCrop(224),# interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                    ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif aug_type == 'noflips':
            self.aug = transforms.Compose([
                transforms.RandomResizedCrop(224),# interpolation=Image.BICUBIC),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                    ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif aug_type=='onlycrops':
            self.aug = transforms.Compose([
                transforms.RandomResizedCrop(224),# interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    def __call__(self, x):
        # initialize views tensor
        views = torch.zeros(self.num_views, 3, 224, 224)
        # first view is not augmented
        views[0] = self.no_aug(x)
        # the rest of the views are augmented
        for i in range(1, self.num_views):
            views[i] = self.aug(x)
        return views
    
class Datasetwithindex(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y, index


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SSL on episodes offline Training')
    parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-100')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--zero_init_res', action='store_true', default=True)
    parser.add_argument('--num_pseudoclasses', type=int, default=100)
    parser.add_argument('--num_views', type=int, default=4)
    parser.add_argument('--lam1', type=float, default=1)
    parser.add_argument('--lam2', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--stop_epoch', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--wd', type=float, default=1.5e-6)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--loss_downweight', action='store_true')
    parser.add_argument('--anchor_based_loss', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--aug_type', type=str, default='all', choices=['all', 'noflips', 'onlycrops'])
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--dp', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default="output/run_SSL_on_episodes")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate learning rate
    args.lr = args.lr * args.batch_size/256 

    # Seed Everything
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    # Define folder to save results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # print and save args
    print(args)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # define tensoboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    # run main function
    main(args, device, writer)

    # Close tensorboard writer
    writer.close()