import argparse
import os, time

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental, InstanceIncremental

from models import *
from loss_functions import SimSiamLossViewExpanded, KoLeoLossViewExpanded
from augmentations import Episode_Transformations
from wake_sleep_trainer import Wake_Sleep_trainer

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

# turn off warnings
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='NREM train encoder wake_sleep - SimSiam')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--data_order_file_name', type=str, default='./../../IM10_data_class_orders/IM10_data_class_order0.txt')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
parser.add_argument('--iid', action='store_true')
# View encoder parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
# Projector
parser.add_argument('--proj_model_name', type=str, default='Projector_Model')
parser.add_argument('--proj_dim', type=int, default=2048)
# Predictor
parser.add_argument('--pred_model_name', type=str, default='Predictor_Model')
parser.add_argument('--pred_dim', type=int, default=512)
# Training parameters
parser.add_argument('--rep_lr', type=float, default=0.0008)
parser.add_argument('--rep_wd', type=float, default=0)
parser.add_argument('--koleo_gamma', type=float, default=0.03)
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=12) 
parser.add_argument('--num_episodes_per_sleep', type=int, default=64000)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--save_dir', type=str, default="output/run_CSSL")
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
    args.num_batch_episodes_per_sleep = int(np.ceil(args.num_episodes_per_sleep/args.episode_batch_size))
    # args.rep_lr = args.rep_lr * args.episode_batch_size / 128 #256
    if not os.path.exists(args.save_dir): # Create save directory
        os.makedirs(args.save_dir)
    print(args)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f: # Save arguments
        json.dump(args.__dict__, f, indent=2)

    ### Seed everything, define device, define writer
    seed_everything(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'Tensorboard_Results'))

    ### Load data tasks
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
    print('\n==> Preparing data...')
    episode_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, return_actions=False)
    train_tranform = [episode_transform]
    val_transform = [transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = args.mean, std = args.std)])]
    train_parent_dataset = ImageFolderDataset(data_path = os.path.join(args.data_path, 'train'))
    val_parent_dataset = ImageFolderDataset(data_path = os.path.join(args.data_path, 'val'))
    data_class_order = list(np.loadtxt(args.data_order_file_name, dtype=int))
    if args.iid: # iid
        train_tasks = InstanceIncremental(train_parent_dataset, nb_tasks=args.num_tasks, transformations=train_tranform)
        val_tasks = InstanceIncremental(val_parent_dataset, nb_tasks=args.num_tasks, transformations=val_transform)
    else: # non-iid (class incremental)
        assert args.num_classes % args.num_tasks == 0, "Number of classes must be divisible by number of tasks"
        args.class_increment = int(args.num_classes // args.num_tasks)
        train_tasks = ClassIncremental(train_parent_dataset, increment = args.class_increment, transformations = train_tranform, class_order = data_class_order)
        val_tasks = ClassIncremental(val_parent_dataset, increment = args.class_increment, transformations = val_transform, class_order = data_class_order)

    ### Create train loader and val laoder for KNN
    train_tasks_knn = ClassIncremental(train_parent_dataset, increment = args.class_increment, transformations = val_transform, class_order = data_class_order)
    train_knn_dataloader = torch.utils.data.DataLoader(train_tasks_knn[:], batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers)
    val_knn_dataloader = torch.utils.data.DataLoader(val_tasks[:], batch_size = args.episode_batch_size, shuffle = False, num_workers = args.workers)

    ### Load models
    print('\n==> Preparing models...')
    view_encoder = eval(args.enc_model_name)(zero_init_residual = True, output_before_avgpool = True)
    projector_rep = eval(args.proj_model_name)(input_dim = view_encoder.fc.weight.shape[1], hidden_dim = args.proj_dim, output_dim = args.proj_dim)
    predictor_rep = eval(args.pred_model_name)(input_dim = args.proj_dim, hidden_dim = args.pred_dim, output_dim = args.proj_dim)
    view_encoder.fc = torch.nn.Identity()

    ### Print models
    print('\nView encoder')
    print(view_encoder)
    print('\nProjector for representation learning')
    print(projector_rep)
    print('\nPredictor for representation learning')
    print(predictor_rep)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    projector_rep = torch.nn.DataParallel(projector_rep)
    predictor_rep = torch.nn.DataParallel(predictor_rep)
    view_encoder = view_encoder.to(device)
    projector_rep = projector_rep.to(device)
    predictor_rep = predictor_rep.to(device)

    ### Define wake-sleep trainer
    print('\n==> Preparing Wake-Sleep trainer...')
    WS_trainer = Wake_Sleep_trainer(episode_batch_size = args.episode_batch_size,
                                    num_episodes_per_sleep = args.num_episodes_per_sleep,
                                    num_views = args.num_views,
                                    dataset_mean = args.mean,
                                    dataset_std = args.std,
                                    device = device,
                                    save_dir = args.save_dir,
                                    koleo_gamma = args.koleo_gamma)
    ### KNN eval before training
    print('\n==> KNN evaluation before training')
    knn_val_all = knn_eval(train_knn_dataloader, val_knn_dataloader, view_encoder, device, k=10, num_classes=args.num_classes)
    print(f'KNN accuracy on all classes: {knn_val_all}')
    writer.add_scalar('KNN_accuracy_all_seen_classes', knn_val_all, 0)

    ### Save view encoder at random init
    view_encoder_state_dict = view_encoder.module.state_dict()
    torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_taskid_randinit.pth'))

    ### Save projector_rep at random init
    projector_rep_state_dict = projector_rep.module.state_dict()
    torch.save(projector_rep_state_dict, os.path.join(args.save_dir, f'projector_rep_taskid_randinit.pth'))

    ### Save predictor_rep at random init
    predictor_rep_state_dict = predictor_rep.module.state_dict()
    torch.save(predictor_rep_state_dict, os.path.join(args.save_dir, f'predictor_rep_taskid_randinit.pth'))

    ### Loop over tasks
    print('\n==> Start wake-sleep training')
    init_time = time.time()
    scaler = GradScaler()
    for task_id in range(len(train_tasks)):
        print(f"\n\n\n#------ Task {task_id+1}/{len(train_tasks)} ------#")
        start_time = time.time()
        writer.add_scalar('Task_boundaries', task_id+1, task_id*args.num_episodes_per_sleep)
        
        # Load data
        train_loader_current = torch.utils.data.DataLoader(train_tasks[task_id], batch_size = args.episode_batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
        val_loader_seen = torch.utils.data.DataLoader(val_tasks[:task_id+1], batch_size = 128, shuffle = False, num_workers = args.workers, pin_memory = True)
        
        ######-- WAKE PHASE --######
        print("\n######-- Wake Phase --######")
        print(f'Seen Tasks: {list(range(task_id+1))} -- Seen classes: {train_tasks[:task_id+1].get_classes()}')
        WS_trainer.wake_phase(train_loader_current)
        del train_loader_current

        ######-- SLEEP PHASE --######
        print("\n######-- Sleep Phase --######")
        param_groups_rep_learning = [{'params': view_encoder.parameters(), 'lr': args.rep_lr, 'weight_decay': args.rep_wd},
                                    {'params': projector_rep.parameters(), 'lr': args.rep_lr, 'weight_decay': args.rep_wd},
                                    {'params': predictor_rep.parameters(), 'lr': args.rep_lr, 'weight_decay': args.rep_wd}]
        
        optimizer_rep = torch.optim.AdamW(param_groups_rep_learning, lr = 0, weight_decay = 0)
        scheduler_rep = torch.optim.lr_scheduler.OneCycleLR(optimizer_rep, max_lr = [args.rep_lr, args.rep_lr, args.rep_lr], steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        criterion_simsiam = SimSiamLossViewExpanded(num_views = args.num_views).to(device)
        criterion_koleo = KoLeoLossViewExpanded(num_views = args.num_views).to(device)
        ### NREM Step ####
        print(f"### NREM step -- task {task_id+1} ##")
        WS_trainer.sleep(view_encoder,
                        projector_rep,
                        predictor_rep,
                        optimizers = [optimizer_rep], 
                        schedulers = [scheduler_rep],
                        criterions = [criterion_simsiam, criterion_koleo],
                        task_id = task_id,
                        scaler=scaler,
                        writer = writer)
        
        ######-- KNN EVALUATION --######
        knn_val_all = knn_eval(train_knn_dataloader, val_knn_dataloader, view_encoder, device, k=10, num_classes=args.num_classes)
        print(f'KNN accuracy on all classes: {knn_val_all}')
        writer.add_scalar('KNN_accuracy_all_seen_classes', knn_val_all, task_id*args.num_episodes_per_sleep + WS_trainer.sleep_episode_counter)
        
        ### Save view encoder
        view_encoder_state_dict = view_encoder.module.state_dict()
        torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_taskid_{task_id}.pth'))

        ### Save projector_rep
        projector_rep_state_dict = projector_rep.module.state_dict()
        torch.save(projector_rep_state_dict, os.path.join(args.save_dir, f'projector_rep_taskid_{task_id}.pth'))

        ### Save predictor_rep
        predictor_rep_state_dict = predictor_rep.module.state_dict()
        torch.save(predictor_rep_state_dict, os.path.join(args.save_dir, f'predictor_rep_taskid_{task_id}.pth'))

        ### Print time
        print(f'Task {task_id} Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} -- Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-init_time))}')

    # Close tensorboard writer
    writer.close()

    print('\n==> END')

    return None

def knn_eval(train_loader, val_loader, view_encoder, device, k, num_classes):
    view_encoder.eval()

    ### Get train features and labels
    train_features = []
    train_labels = []
    with torch.no_grad():
        for i, (imgs, labels, _) in enumerate(train_loader):
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
        for i, (imgs, labels, _) in enumerate(val_loader):
            imgs = imgs.to(device)
            features = view_encoder(imgs)
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