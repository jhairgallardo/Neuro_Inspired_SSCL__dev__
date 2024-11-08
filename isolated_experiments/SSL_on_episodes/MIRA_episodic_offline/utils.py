import torch
import numpy as np
import random
import os
from scipy.spatial import distance_matrix

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

@torch.no_grad()
def statistics(prob, eps=1e-10):
    # prob = concat_all_gather(prob) if dist.is_available() and dist.is_initialized() else prob
    entropy = - (prob * torch.log(prob + eps)).sum(dim=1).mean()
    m_prob = prob.mean(dim=0)   # marginal probability
    m_entropy = - (m_prob * torch.log(m_prob + eps)).sum()   # marginal entropy
    mi = m_entropy - entropy
    return entropy, m_entropy, mi

        

    
