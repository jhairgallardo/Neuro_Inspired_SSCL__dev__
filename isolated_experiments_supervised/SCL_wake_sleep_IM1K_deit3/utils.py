import torch
import torch.distributed as dist
import torchvision
import numpy as np
from PIL import Image
import os

def file_broadcast_tensor(tensor: torch.Tensor, tmp_path: str, rank: int):
    """
    Rank 0  → torch.save(tensor, tmp_path)
    All ranks → torch.load(tmp_path)
    Barrier before/after to synchronize.
    Returns the loaded tensor on each rank (CPU).
    """
    if rank == 0:
        # overwrites any existing file
        torch.save(tensor, tmp_path)
    dist.barrier()   # wait for rank 0 to finish saving

    # everyone loads from the same file path
    out = torch.load(tmp_path, map_location='cpu')
    dist.barrier()   # wait for all loads to complete

    # (optionally) have rank 0 delete the file afterwards
    if rank == 0 and os.path.exists(tmp_path):
        os.remove(tmp_path)

    return out


def file_broadcast_list(pylist: list, tmp_path: str, rank: int):
    """
    Same pattern for Python lists: pickle via torch.save / torch.load.
    """
    if rank == 0:
        torch.save(pylist, tmp_path)
    dist.barrier()
    out = torch.load(tmp_path)
    dist.barrier()
    if rank == 0 and os.path.exists(tmp_path):
        os.remove(tmp_path)
    return out

def reduce_tensor(tensor: torch.Tensor, mean: bool = False) -> torch.Tensor:
    """
    All‐reduce a 1‐element tensor across all ranks. If `mean=True`, divide by world_size.

    Args:
        tensor:  A 1‐element float/int tensor (already on `device`).
        mean:    If True, divide the summed tensor by world_size to get the average.

    Returns:
        The reduced tensor (on the same device), either summed (if mean=False)
        or averaged (if mean=True).
    """
    # In‐place all‐reduce sum:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if mean:
        world_size = dist.get_world_size()
        tensor /= world_size
    return tensor

class MetricLogger(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def time_duration_print(seconds: float) -> str:
    """Return H:MM:SS (with H unbounded) from a duration in seconds."""
    total_seconds = int(seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def plot_generated_images_hold_set(view_encoder, cond_generator, episodes_plot_dict, task_id, mean, std, num_views, save_dir, device):
    view_encoder.eval()
    cond_generator.eval()

    for task_id_plot, episodes_plot in episodes_plot_dict.items():
        B, V, C, H, W = episodes_plot[0].shape
        episodes_imgs = episodes_plot[0].to(device) # (B, V, C, H, W)
        episodes_actions = episodes_plot[1] # (B, V)
        with torch.no_grad():
            first_view_feats = view_encoder(episodes_imgs[:, 0])[:, 1:, :] # (B, T-1, D)
            first_view_feats = first_view_feats.unsqueeze(1).expand(-1, V, -1, -1) # (B, V, T-1, D)
            first_view_feats = first_view_feats.reshape(B * V, first_view_feats.shape[2], first_view_feats.shape[3])  # (B*V, T-1, D)
            flat_actions = [episodes_actions[b][v] for b in range(B) for v in range(V)]  # list length B*V
            flat_gen_imgs, _ = cond_generator(first_view_feats, flat_actions) # input tokens without CLS token. (B*V, C, H, W)
            episodes_gen_imgs = flat_gen_imgs.reshape(B, V, C, H, W)  # Reshape back to (B, V, C, H, W).
        # Detach and move to CPU
        episodes_imgs = episodes_imgs.detach().cpu()
        episodes_gen_imgs = episodes_gen_imgs.detach().cpu()

        # Plot for each episode
        for i in range(len(episodes_imgs)):
            episode_i_imgs = episodes_imgs[i]
            episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_i_imgs]
            episode_i_imgs = torch.stack(episode_i_imgs, dim=0)

            episode_i_gen_imgs = episodes_gen_imgs[i]
            episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(mean, std)], [1/s for s in std]) for img in episode_i_gen_imgs]
            episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0)
            episode_i_gen_imgs = torch.clamp(episode_i_gen_imgs, 0, 1) # Clip values to [0, 1]

            grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs], dim=0), nrow=num_views)
            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            grid = Image.fromarray(grid)
            save_plot_dir = os.path.join(save_dir, 'generated_images_hold_set', f'Learned_taskid_{task_id}')
            image_name = f'{task_id_plot}_episode{i}.png'
            # create folder if it doesn't exist
            if not os.path.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            grid.save(os.path.join(save_plot_dir, image_name))
            
    return None