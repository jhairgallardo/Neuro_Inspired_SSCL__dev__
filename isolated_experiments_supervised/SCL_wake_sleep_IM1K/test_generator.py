import argparse
import os, time

import torch
import torch.nn.functional as F
from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental
import torchvision
from torch.cuda.amp import GradScaler, autocast


from models_GNMish_lossgen1only import *
from augmentations import Episode_Transformations

from tensorboardX import SummaryWriter
import numpy as np
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

from copy import deepcopy

parser = argparse.ArgumentParser(description='Conditional Generator Test')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet2012')
parser.add_argument('--num_pretraining_classes', type=int, default=10)
parser.add_argument('--data_order_file_name', type=str, default='./IM1K_data_class_orders/imagenet_class_order_siesta.txt')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
# View encoder parameters (pretrained)
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--enc_pretrained_file_path', type=str, default='./output/Pretrained_encoders/PreEnc100c_resnet18_views@4no1stview_epochs@100_lr@0.01_wd@0.05_bs@512_koleo@0.01_seed@0/view_encoder_epoch99.pth')
# Conditional Generator parameters
parser.add_argument('--condgen_model_name', type=str, default='ConditionalGenerator')
parser.add_argument('--dec_num_Blocks', type=list, default=[1,1,1,1])
parser.add_argument('--dec_num_out_channels', type=int, default=3)
parser.add_argument('--action_code_dim', type=int, default=12)
parser.add_argument('--ft_num_layers', type=int, default=8)
parser.add_argument('--ft_nhead', type=int, default=8)
parser.add_argument('--ft_dim_feedforward', type=int, default=1024)
parser.add_argument('--ft_dropout', type=float, default=0.1)
# Training parameters
parser.add_argument('--episode_batch_size', type=int, default=150)
parser.add_argument('--num_views', type=int, default=4) 
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--save_dir', type=str, default="output/run_cond_generator_pretraining_testing")
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
    print(args)
    if not os.path.exists(args.save_dir): # create save dir
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Seed everything
    seed_everything(seed=args.seed)

    ### Load data
    print('\n==> Preparing data...')
    traindir = os.path.join(args.data_path, 'train')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=True)
    train_dataset_continuum = ImageFolderDataset(traindir)

    # Get data classfolder order from file (the file has order using folder names)
    with open(args.data_order_file_name, 'r') as f:
        data_classfolder_order = f.read().splitlines()
    # Create a classfolder to class index mapping
    classfolder_to_idx={}
    img_paths, classidxes, _ = train_dataset_continuum.get_data()
    for img_path, classidx in zip(img_paths, classidxes):
        class_name = str(img_path).split('/')[-2]
        if class_name not in classfolder_to_idx:
            classfolder_to_idx[class_name] = classidx
    idx_to_classfolder = {v: k for k, v in classfolder_to_idx.items()}
    # Transform the class folder order to class index order
    data_class_order = []
    for class_name in data_classfolder_order:
        if class_name in classfolder_to_idx:
            data_class_order.append(classfolder_to_idx[class_name])
        else:
            raise ValueError(f"Class {class_name} not found in the dataset.")
    
    train_tasks = ClassIncremental(train_dataset_continuum, increment=1, initial_increment=args.num_pretraining_classes, transformations=[train_transform], class_order=data_class_order)
    train_dataset = train_tasks[0] # Create the train dataset taking only the first task (the first 100 classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size, shuffle=True,
                                               num_workers=args.workers, drop_last=True)
    
    ### Load pretrained view_encoder
    print('\n==> Load pre-trained view encoder...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    missing_keys, unexpected_keys = view_encoder.load_state_dict(torch.load(args.enc_pretrained_file_path), strict=False)
    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    feature_dim = view_encoder.fc.weight.shape[1]
    view_encoder.fc = torch.nn.Identity() # remove last fc layer from view_encoder network (it is not trained)
    for param in view_encoder.parameters(): # freeze view_encoder
        param.requires_grad = False

    ### Load Conditional generator
    print('\n==> Load Conditional Generator')
    cond_generator = eval(args.condgen_model_name)(action_code_dim=args.action_code_dim,
                                            feature_dim=feature_dim,
                                            ft_num_layers=args.ft_num_layers,
                                            ft_nhead=args.ft_nhead,
                                            ft_dim_feedforward=args.ft_dim_feedforward,
                                            ft_dropout=args.ft_dropout,
                                            dec_num_Blocks=args.dec_num_Blocks,
                                            dec_num_out_channels=args.dec_num_out_channels)
    ## Sanity check (load trained generator)
    cond_generator_state_dict = torch.load('/home/jhair/Research/DOING/Neuro_Inspired_SSCL__dev__/isolated_experiments_supervised/SCL_wake_sleep_IM1K/output/Pretrained_condgenerators/FOR_PreEnc100c_resnet18_views@4no1stview_epochs@100_lr@0.01_wd@0.05_bs@512_koleo@0.01_seed@0/lossgen1only_PreCondGenGELUEncoder10cTokenConcat_Bilinear_8layers1024dim8nheads_views@2_epochs@100_lr@0.001_wd@0_bs@104_seed@0/cond_generator_epoch99.pth')
    cond_generator.load_state_dict(cond_generator_state_dict, strict=True)
    del cond_generator_state_dict
    # freeze generator
    for param in cond_generator.parameters():
        param.requires_grad = False

    ### Print models
    print('\nView encoder')
    print(view_encoder)
    print('\nConditional Generator')
    print(cond_generator)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    cond_generator = torch.nn.DataParallel(cond_generator)
    view_encoder = view_encoder.to(args.device)
    cond_generator = cond_generator.to(args.device)

    ### Load optimizer and criterion
    criterion = torch.nn.MSELoss(reduction='none') #.to(args.device)

    ### Save one batch for plot purposes
    # episodes_plot, _, _ = next(iter(train_loader))

    ### Testing
    print('\n==> Testing model')

    ## Val STEP ##
    accum_loss=0
    view_encoder.eval()
    cond_generator.eval()

    loss_acumm = []
    lossgen_1_acumm = []
    # lossgen_2_acumm = []
    # lossgen_3_acumm = []

    with torch.no_grad():
        for i, (batch_episode, _, _) in enumerate(train_loader):
            batch_episode_imgs = batch_episode[0].to(args.device)
            batch_episode_actions = batch_episode[1].to(args.device)

            # Pass obtained tensors through conditional generator
            batch_episode_tensors = torch.empty(0).to(args.device)
            batch_episode_gen_FTtensors = torch.empty(0).to(args.device)
            batch_episode_gen_DecEnctensors = torch.empty(0).to(args.device)
            batch_episode_gen_DecEnctensors_direct = torch.empty(0).to(args.device)

            batch_first_view_tensors = view_encoder(batch_episode_imgs[:,0])

            for v in range(args.num_views):
                batch_imgs = batch_episode_imgs[:,v]
                batch_actions = batch_episode_actions[:,v]

                # Forward pass on view encoder
                batch_tensors = view_encoder(batch_imgs)

                # Conditional forward pass (When v=0, the action codes are "no action". Using first view to predict the same first view)
                batch_gen_images, batch_gen_FTtensors = cond_generator(batch_first_view_tensors, batch_actions)
                # batch_gen_DecEnctensors = view_encoder(batch_gen_images)

                # Direct forward pass (skip FTN to boost training of generator)
                # batch_gen_images_direct = cond_generator(batch_tensors, None, skip_FTN=True)
                # batch_gen_DecEnctensors_direct = view_encoder(batch_gen_images_direct)

                # Concatenate tensors
                batch_episode_tensors = torch.cat([batch_episode_tensors, batch_tensors.unsqueeze(1)], dim=1)
                batch_episode_gen_FTtensors = torch.cat([batch_episode_gen_FTtensors, batch_gen_FTtensors.unsqueeze(1)], dim=1)
                # batch_episode_gen_DecEnctensors = torch.cat([batch_episode_gen_DecEnctensors, batch_gen_DecEnctensors.unsqueeze(1)], dim=1)
                # batch_episode_gen_DecEnctensors_direct = torch.cat([batch_episode_gen_DecEnctensors_direct, batch_gen_DecEnctensors_direct.unsqueeze(1)], dim=1)

            # conditional loss (FT) (first view + action --> other views)
            lossgen_1_per_sample = criterion(batch_episode_gen_FTtensors, batch_episode_tensors).mean(dim=(2,3,4))
            lossgen_1 = lossgen_1_per_sample.mean()
            # conditional loss (DecEnc) (first view + action --> other views)
            # lossgen_2_per_sample = criterion(batch_episode_gen_DecEnctensors, batch_episode_tensors).mean(dim=(2,3,4))
            # lossgen_2 = lossgen_2_per_sample.mean()
            # direct loss (DecEnc) (views --> views)
            # lossgen_3_per_sample = criterion(batch_episode_gen_DecEnctensors_direct, batch_episode_tensors).mean(dim=(2,3,4))
            # lossgen_3 = lossgen_3_per_sample.mean()
            # total loss
            loss = lossgen_1 #+ lossgen_2 + lossgen_3

            accum_loss += loss.item() # accumulate loss

            # Loss results per batch
            print(f'Batch {i+1}/{len(train_loader)} -- ' +
                f'Loss 1 (FT): {lossgen_1.item():.6f} -- ' +
                # f'Loss 2 (DecEnc): {lossgen_2.item():.6f} -- ' +
                # f'Loss 3 (DecEnc_direct): {lossgen_3.item():.6f} -- '
                f'Loss: {loss.item():.6f}'
                )

            # Append all training data loss
            lossgen_1_acumm.append(lossgen_1_per_sample.detach().cpu())
            # lossgen_2_acumm.append(lossgen_2_per_sample.detach().cpu())
            # lossgen_3_acumm.append(lossgen_3_per_sample.detach().cpu())
            loss_acumm.append(lossgen_1_per_sample.detach().cpu())# + lossgen_2_per_sample.detach().cpu() + lossgen_3_per_sample.detach().cpu())

            # For 1 batch_episode_tensor and 1 batch_episode_gen_DecEnctensors (view 1), I want to plot each of their range values and the range of its MSE
            if i==0: #only for first batch
                for n_img in range(6):
                    for view_num in range(4):
                        target_tensor = batch_episode_tensors[n_img, view_num]
                        # gen_tensor = batch_episode_gen_DecEnctensors[n_img, 1]
                        gen_tensor = batch_episode_gen_FTtensors[n_img, view_num]
                        sqerror_tensor = criterion(target_tensor, gen_tensor)
                        mse_value = sqerror_tensor.mean().item()

                        target_image = batch_episode_imgs[n_img, view_num]
                        # gen_image = batch_gen_images[n_img]

                        # Let's do a q row 3 columns plot (matplotlib subplot)
                        # First plot should have the histogram of the target tensor
                        # let's use a wide figure
                        plt.figure(figsize=(16, 5))
                        plt.subplot(1, 3, 1)
                        plt.hist(target_tensor.flatten().detach().cpu(), bins=100)
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        plt.title('Target Tensor Histogram')
                        # Second plot should have the histogram of the generated tensor
                        plt.subplot(1, 3, 2)
                        plt.hist(gen_tensor.flatten().detach().cpu(), bins=100)
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        plt.title('Generated Tensor Histogram')
                        # Third plot should have the histogram of the squared error
                        plt.subplot(1, 3, 3)
                        plt.hist(sqerror_tensor.flatten().detach().cpu(), bins=100)
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        plt.title(f'Squared Error Histogram (MSE={mse_value:.6f})')
                        # Save the plot
                        plt.savefig(os.path.join(args.save_dir, f'Comparison_batch_{i+1}_image_{n_img}_view{view_num}.png'), bbox_inches='tight')
                        plt.close()

                        # I want to do a similar 3 columns plot but instead I want to show the box plot of the target tensor, generated tensor and squared error
                        plt.figure(figsize=(16, 5))
                        plt.subplot(1, 3, 1)
                        plt.boxplot(target_tensor.flatten().detach().cpu(), vert=False)
                        plt.xlabel('Value')
                        plt.title('Target Tensor Boxplot')
                        plt.subplot(1, 3, 2)
                        plt.boxplot(gen_tensor.flatten().detach().cpu(), vert=False)
                        plt.xlabel('Value')
                        plt.title('Generated Tensor Boxplot')
                        plt.subplot(1, 3, 3)
                        plt.boxplot(sqerror_tensor.flatten().detach().cpu(), vert=False)
                        plt.xlabel('Value')
                        plt.title(f'Squared Error Boxplot (MSE={mse_value:.6f})')
                        # Save the plot
                        plt.savefig(os.path.join(args.save_dir, f'Comparison_boxplot_batch_{i+1}_image_{n_img}_view{view_num}.png'), bbox_inches='tight')
                        plt.close()

                        # Based on the sqerror tensor, I want to create a heatmap where if shows high values in red and low values in blue
                        heatmap_error = sqerror_tensor.mean(dim=0).detach().cpu().numpy()
                        # heatmap_error = np.clip(heatmap_error, 0, 1) # clip values to [0, 1]
                        # heatmap_error = (heatmap_error - np.min(heatmap_error)) / (np.max(heatmap_error) - np.min(heatmap_error)) # normalize values to [0, 1]
                        plt.imshow(heatmap_error, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        plt.title(f'Heatmap of Squared Error (MSE={mse_value:.6f})')
                        plt.savefig(os.path.join(args.save_dir, f'Heatmap_batch_{i+1}_image_{n_img}_view{view_num}.png'), bbox_inches='tight')
                        plt.close()

                        # Here plot the image and the predicted image Subplot 1 row 2 columns
                        plt.figure(figsize=(16, 8))
                        plt.subplot(1, 2, 1)
                        plt.imshow(torchvision.transforms.functional.normalize(target_image, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]).permute(1, 2, 0).detach().cpu().numpy())
                        plt.title('Target Image')
                        plt.axis('off')
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(torchvision.transforms.functional.normalize(gen_image, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]).permute(1, 2, 0).detach().cpu().numpy())
                        # plt.title('Generated Image')
                        # plt.axis('off')
                        # Save the plot
                        plt.savefig(os.path.join(args.save_dir, f'Image_batch_{i+1}_image_{n_img}_view{view_num}.png'), bbox_inches='tight')
                        plt.close()




    # Plot loss histogram per view (on all training set)
    loss_acumm = torch.stack(loss_acumm, dim=0)
    lossgen_1_acumm = torch.stack(lossgen_1_acumm, dim=0)
    # lossgen_2_acumm = torch.stack(lossgen_2_acumm, dim=0)
    # lossgen_3_acumm = torch.stack(lossgen_3_acumm, dim=0)

    for v in range(args.num_views):

        # Total loss histogram
        lossgen1_accum_view = loss_acumm[:,:, v]
        lossgen1_accum_view = lossgen1_accum_view.flatten()
        plt.hist(lossgen1_accum_view, bins=100)
        plt.xlabel(f'MSE')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of view {v} loss')
        plt.savefig(os.path.join(args.save_dir, f'Hist_LossgenTotal_view{v}.png'))
        plt.close()

        # Lossgen1 histogram
        lossgen1_accum_view = lossgen_1_acumm[:,:, v]
        lossgen1_accum_view = lossgen1_accum_view.flatten()
        plt.hist(lossgen1_accum_view, bins=100)
        plt.xlabel(f'MSE')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of view {v} lossgen1')
        plt.savefig(os.path.join(args.save_dir, f'Hist_Lossgen1_view{v}.png'))
        plt.close()

        # # Lossgen2 histogram
        # lossgen2_accum_view = lossgen_2_acumm[:,:, v]
        # lossgen2_accum_view = lossgen2_accum_view.flatten()
        # plt.hist(lossgen2_accum_view, bins=100)
        # plt.xlabel(f'MSE')
        # plt.ylabel('Frequency')
        # plt.title(f'Histogram of view {v} lossgen2')
        # plt.savefig(os.path.join(args.save_dir, f'Hist_Lossgen2_view{v}.png'))
        # plt.close()

        # # Lossgen3 histogram
        # lossgen3_accum_view = lossgen_3_acumm[:,:, v]
        # lossgen3_accum_view = lossgen3_accum_view.flatten()
        # plt.hist(lossgen3_accum_view, bins=100)
        # plt.xlabel(f'MSE')
        # plt.ylabel('Frequency')
        # plt.title(f'Histogram of view {v} lossgen3')
        # plt.savefig(os.path.join(args.save_dir, f'Hist_Lossgen3_view{v}.png'))
        # plt.close()   

    accum_loss /= len(train_loader)
    print(f'Total Train Loss: {accum_loss:.6f}')

    # ## Plot reconctructions examples
    # view_encoder.eval()
    # cond_generator.eval()
    # n = 8
    # episodes_plot_imgs = episodes_plot[0][:n].to(args.device)
    # episodes_plot_actions = episodes_plot[1][:n].to(args.device)
    # episodes_plot_gen_imgs = torch.empty(0)
    # with torch.no_grad():
    #     first_view_tensors = view_encoder(episodes_plot_imgs[:,0])
    #     for v in range(args.num_views):
    #         actions = episodes_plot_actions[:,v]
    #         gen_images, _ = cond_generator(first_view_tensors, actions)
    #         episodes_plot_gen_imgs = torch.cat([episodes_plot_gen_imgs, gen_images.unsqueeze(1).detach().cpu()], dim=1)
    # episodes_plot_imgs = episodes_plot_imgs.detach().cpu()
    # # plot each episode
    # for i in range(n):
    #     episode_i_imgs = episodes_plot_imgs[i]
    #     episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_imgs]
    #     episode_i_imgs = torch.stack(episode_i_imgs, dim=0)

    #     episode_i_gen_imgs = episodes_plot_gen_imgs[i]
    #     episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_gen_imgs]
    #     episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0)

    #     grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs], dim=0), nrow=args.num_views)
    #     grid = grid.permute(1, 2, 0).cpu().numpy()
    #     grid = (grid * 255).astype(np.uint8)
    #     grid = Image.fromarray(grid)
    #     image_name = f'episode{i}.png'
    #     save_plot_dir = os.path.join(args.save_dir, 'gen_plots')
    #     # create folder if it doesn't exist
    #     if not os.path.exists(save_plot_dir):
    #         os.makedirs(save_plot_dir)
    #     grid.save(os.path.join(save_plot_dir, image_name))

    return None

if __name__ == '__main__':
    main()