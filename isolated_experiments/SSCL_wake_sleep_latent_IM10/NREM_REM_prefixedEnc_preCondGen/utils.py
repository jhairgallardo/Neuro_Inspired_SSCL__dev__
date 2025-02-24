import torch
import numpy as np
from scipy.spatial import distance_matrix

@torch.no_grad()
def statistics(prob, eps=1e-10):
    # prob = concat_all_gather(prob) if dist.is_available() and dist.is_initialized() else prob
    entropy = - (prob * torch.log(prob + eps)).sum(dim=1).mean()
    m_prob = prob.mean(dim=0)   # marginal probability
    m_entropy = - (m_prob * torch.log(m_prob + eps)).sum()   # marginal entropy
    mi = m_entropy - entropy
    return entropy, m_entropy, mi


        

    
