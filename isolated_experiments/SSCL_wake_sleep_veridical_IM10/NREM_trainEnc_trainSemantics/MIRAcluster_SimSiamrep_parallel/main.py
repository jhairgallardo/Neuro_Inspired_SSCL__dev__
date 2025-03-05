import argparse
import os, time

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

from continuum.datasets import ImageFolderDataset
from continuum import ClassIncremental, InstanceIncremental

from models import *
from loss_functions import SwapLossViewExpanded, SimSiamLossViewExpanded
from augmentations import Episode_Transformations
from wake_sleep_trainer import Wake_Sleep_trainer, evaluate_semantic_memory

from tensorboardX import SummaryWriter
import numpy as np
import json
import random

# turn off warnings
# import warnings
# warnings.filterwarnings("ignore")

# Have only GPUs 1 2 3 visible
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

parser = argparse.ArgumentParser(description='Semantic Memory training analysis - MIRAcluster')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/ImageNet-10')
parser.add_argument('--data_order_file_name', type=str, default='./../../IM10_data_class_orders/IM10_data_class_order0.txt')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
parser.add_argument('--iid', action='store_true')
# Representation Learning parameters
parser.add_argument('--enc_model_name', type=str, default='resnet18')
parser.add_argument('--rep_proj_model_name', type=str, default='Projector_Model')
parser.add_argument('--rep_proj_dim', type=int, default=2048)
parser.add_argument('--rep_pred_model_name', type=str, default='Predictor_Model')
parser.add_argument('--rep_pred_dim', type=int, default=512)
parser.add_argument('--rep_lr', type=float, default=0.0008)
parser.add_argument('--rep_wd', type=float, default=0)
# Semantic Learning parameters
parser.add_argument('--semantic_model_name', type=str, default='Semantic_Memory_Model')
parser.add_argument('--num_pseudoclasses', type=int, default=32)
parser.add_argument('--sem_lr', type=float, default=0.01)
parser.add_argument('--sem_wd', type=float, default=0)
parser.add_argument('--sem_proj_dim', type=int, default=2048)
parser.add_argument('--sem_out_dim', type=int, default=1024)
parser.add_argument('--tau_t', type=float, default=0.225)
parser.add_argument('--tau_s', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.75)
# Training parameters
parser.add_argument('--episode_batch_size', type=int, default=128)
parser.add_argument('--num_views', type=int, default=12) 
parser.add_argument('--num_episodes_per_sleep', type=int, default=64000)
parser.add_argument('--workers', type=int, default=32)
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
    args.sem_lr = args.sem_lr * args.episode_batch_size / 128
    args.rep_lr = args.rep_lr * args.episode_batch_size / 128
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
    projector_rep = eval(args.rep_proj_model_name)(input_dim = view_encoder.fc.weight.shape[1], 
                                                   hidden_dim = args.rep_proj_dim, 
                                                   output_dim = args.rep_proj_dim)
    predictor_rep = eval(args.rep_pred_model_name)(input_dim = args.rep_proj_dim, 
                                                   hidden_dim = args.rep_pred_dim, 
                                                   output_dim = args.rep_proj_dim)
    semantic_memory = eval(args.semantic_model_name)(input_dim = view_encoder.fc.weight.shape[1], 
                                                     num_pseudoclasses = args.num_pseudoclasses,
                                                     hidden_dim = args.sem_proj_dim,
                                                     output_dim = args.sem_out_dim)
    view_encoder.fc = torch.nn.Identity()

    ### Print models
    print('\nView encoder model')
    print(view_encoder)
    print('\nProjector representation model')
    print(projector_rep)
    print('\nPredictor representation model')
    print(predictor_rep)
    print('\nSemantic Memory model')
    print(semantic_memory)
    print('\n')

    ### Dataparallel and move models to device
    view_encoder = torch.nn.DataParallel(view_encoder)
    projector_rep = torch.nn.DataParallel(projector_rep)
    predictor_rep = torch.nn.DataParallel(predictor_rep)
    semantic_memory = torch.nn.DataParallel(semantic_memory)
    view_encoder = view_encoder.to(device)
    projector_rep = projector_rep.to(device)
    predictor_rep = predictor_rep.to(device)
    semantic_memory = semantic_memory.to(device)

    ### Define wake-sleep trainer
    print('\n==> Preparing Wake-Sleep trainer...')
    WS_trainer = Wake_Sleep_trainer(view_encoder = view_encoder,
                                    projector_rep = projector_rep,
                                    predictor_rep = predictor_rep,
                                    semantic_memory = semantic_memory, 
                                    episode_batch_size = args.episode_batch_size,
                                    num_episodes_per_sleep = args.num_episodes_per_sleep,
                                    num_views = args.num_views,
                                    tau_t = args.tau_t,
                                    tau_s = args.tau_s,
                                    beta = args.beta,
                                    dataset_mean = args.mean,
                                    dataset_std = args.std,
                                    device = device,
                                    save_dir = args.save_dir)
    
    ### KNN eval before training
    print('\n==> KNN evaluation before training')
    knn_val_all = knn_eval(train_knn_dataloader, val_knn_dataloader, view_encoder, device, k=10, num_classes=args.num_classes)
    print(f'KNN accuracy on all classes: {knn_val_all}')
    writer.add_scalar('KNN_accuracy_all_seen_classes', knn_val_all, 0)

    ### Save view encoder model
    view_encoder_state_dict = view_encoder.module.state_dict()
    torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_taskid_randinit.pth'))

    ### Save projector representation model
    projector_rep_state_dict = projector_rep.module.state_dict()
    torch.save(projector_rep_state_dict, os.path.join(args.save_dir, f'projector_rep_taskid_randinit.pth'))

    ### Save predictor representation model
    predictor_rep_state_dict = predictor_rep.module.state_dict()
    torch.save(predictor_rep_state_dict, os.path.join(args.save_dir, f'predictor_rep_taskid_randinit.pth'))

    ### Save semantic memory model
    semantic_memory_state_dict = semantic_memory.module.state_dict()
    torch.save(semantic_memory_state_dict, os.path.join(args.save_dir, f'semantic_memory_taskid_randinit.pth'))

    ### Loop over tasks
    print('\n==> Start wake-sleep training')
    init_time = time.time()
    scaler = GradScaler()
    val_seen_metrics = {}
    val_each_task_metrics = {}
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
        scheduler_rep = torch.optim.lr_scheduler.OneCycleLR(optimizer_rep, max_lr = [args.rep_lr, args.rep_lr, args.rep_lr], 
                                                            steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        optimizer_sem = torch.optim.AdamW(semantic_memory.parameters(), lr = args.sem_lr, weight_decay = args.sem_wd)
        scheduler_sem = torch.optim.lr_scheduler.OneCycleLR(optimizer_sem, max_lr = args.sem_lr, steps_per_epoch = args.num_batch_episodes_per_sleep, epochs=1)
        criterion_rep_learning = SimSiamLossViewExpanded(num_views = args.num_views).to(device)
        criterion_sem_learning = SwapLossViewExpanded(num_views = args.num_views).to(device)
        ### NREM Step ####
        print(f"### NREM step -- task {task_id+1} ##")
        WS_trainer.sleep(optimizers = [optimizer_rep, optimizer_sem], 
                            schedulers = [scheduler_rep, scheduler_sem],
                            criterions = [criterion_rep_learning, criterion_sem_learning],
                            task_id = task_id,
                            scaler = scaler,
                            writer = writer)
        
        ######-- KNN EVALUATION --######
        knn_val_all = knn_eval(train_knn_dataloader, val_knn_dataloader, view_encoder, device, k=10, num_classes=args.num_classes)
        print(f'KNN accuracy on all classes: {knn_val_all}')
        writer.add_scalar('KNN_accuracy_all_seen_classes', knn_val_all, task_id*args.num_episodes_per_sleep + WS_trainer.sleep_episode_counter)

        ### Validation step on Val Seen data
        val_seen_stats = evaluate_semantic_memory(val_loader_seen, 
                                                  view_encoder, 
                                                  projector_rep,
                                                  predictor_rep,
                                                  semantic_memory, 
                                                  args.tau_s, args.num_pseudoclasses, 
                                                  task_id, device, True, args.num_classes, args.save_dir)
        seen_episodes_so_far = task_id*args.num_episodes_per_sleep + WS_trainer.sleep_episode_counter
        writer.add_scalar('Val_seen_NMI', val_seen_stats['NMI'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_AMI', val_seen_stats['AMI'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_ARI', val_seen_stats['ARI'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_F', val_seen_stats['F'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_Top1_ACC', val_seen_stats['ACC'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_Top5_ACC', val_seen_stats['ACC-Top5'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_Purity', val_seen_stats['Purity'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_Homogeneity', val_seen_stats['Homogeneity'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_Completeness', val_seen_stats['Completeness'], seen_episodes_so_far)
        writer.add_scalar('Val_seen_Class-Fragmentation', val_seen_stats['Class-Fragmentation'], seen_episodes_so_far)
        writer.add_scalar('Task_id_val_tag', task_id, seen_episodes_so_far)
        print(f"Val Seen Task {task_id} -- NMI: {val_seen_stats['NMI']:.4f} -- AMI: {val_seen_stats['AMI']:.4f} -- " + 
              f"ARI: {val_seen_stats['ARI']:.4f} -- F: {val_seen_stats['F']:.4f} -- Top1 ACC: {val_seen_stats['ACC']:.4f} -- " +
              f"Top5 ACC: {val_seen_stats['ACC-Top5']:.4f} -- Purity: {val_seen_stats['Purity']:.4f} -- " +
              f"Homogeneity: {val_seen_stats['Homogeneity']:.4f} -- Completeness: {val_seen_stats['Completeness']:.4f} -- " + 
              f"Class-Fragmentation: {val_seen_stats['Class-Fragmentation']:.4f}")
        
        val_seen_metrics[f'Seen Tasks IDs {list(range(task_id+1))} performance'] = val_seen_stats
        with open(os.path.join(args.save_dir, 'val_seen_metrics.json'), 'w') as f:
            json.dump(val_seen_metrics, f, indent=2)

        ### Validation step on each Val task
        val_each_task_metrics[f'Seen Tasks IDs {list(range(task_id+1))}'] = {}
        for task_id_aux in range(task_id+1):
            task_id_val = torch.utils.data.DataLoader(val_tasks[task_id_aux], batch_size = 128, shuffle = False, num_workers = args.workers, pin_memory = True)
            task_id_val_stats = evaluate_semantic_memory(task_id_val, 
                                                         view_encoder, 
                                                         projector_rep,
                                                         predictor_rep,
                                                         semantic_memory, 
                                                         args.tau_s, args.num_pseudoclasses, None, device, False, args.num_classes, None)
            val_each_task_metrics[f'Seen Tasks IDs {list(range(task_id+1))}'][f'Task ID {task_id_aux} performance'] = task_id_val_stats
        # Save Validation Each task metrics
        with open(os.path.join(args.save_dir, 'val_each_task_metrics.json'), 'w') as f:
            json.dump(val_each_task_metrics, f, indent=2)

        
        ### Save view encoder model
        view_encoder_state_dict = view_encoder.module.state_dict()
        torch.save(view_encoder_state_dict, os.path.join(args.save_dir, f'view_encoder_taskid_{task_id}.pth'))

        ### Save projector representation model
        projector_rep_state_dict = projector_rep.module.state_dict()
        torch.save(projector_rep_state_dict, os.path.join(args.save_dir, f'projector_rep_taskid_{task_id}.pth'))

        ### Save predictor representation model
        predictor_rep_state_dict = predictor_rep.module.state_dict()
        torch.save(predictor_rep_state_dict, os.path.join(args.save_dir, f'predictor_rep_taskid_{task_id}.pth'))

        ### Save semantic memory model
        semantic_memory_state_dict = semantic_memory.module.state_dict()
        torch.save(semantic_memory_state_dict, os.path.join(args.save_dir, f'semantic_memory_taskid_{task_id}.pth'))

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