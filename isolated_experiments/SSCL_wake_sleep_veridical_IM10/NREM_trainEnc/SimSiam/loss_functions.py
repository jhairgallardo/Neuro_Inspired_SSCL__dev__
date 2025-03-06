import torch
import torch.nn.functional as F    

def simsiam_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Computes SimSiam's loss given batch of predicted features p from view 1 and
    a batch of projected features z from view 2.

    Args:
        p (torch.Tensor): Tensor containing predicted features from view 1.
        z (torch.Tensor): Tensor containing projected features from view 2.
        simplified (bool): faster computation, but with same result.

    Returns:
        torch.Tensor: SimSiam loss.
    """

    if simplified:
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    return -(p * z.detach()).sum(dim=1).mean()

class SimSiamLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(SimSiamLossViewExpanded, self).__init__()
        self.N = num_views

    def forward(self, predictions, targets):
        loss = 0
        for t in range(self.N-1):
            loss0 = simsiam_loss_func(predictions[:,0], targets[:,t+1])
            loss1 = simsiam_loss_func(predictions[:,t+1], targets[:,0])
            temp = ( loss0 + loss1 ) / 2.
            loss = loss + temp
        loss = loss / (self.N-1)
        return loss
    
class KoLeoLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(KoLeoLossViewExpanded, self).__init__()
        """Koleo loss for multiple views"""
        
        self.num_views = num_views
        self.KoLeo = KoLeoLoss()

    def forward(self, episodes_vectors):
        loss = 0
        for t in range(self.num_views): # t is the view index
            loss += self.KoLeo(episodes_vectors[:,t])
        loss = loss / self.num_views

        return loss

class KoLeoLoss(torch.nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = torch.nn.PairwiseDistance(2, eps=1e-12)

    def forward(self, student_output, eps=1e-12):
        """
        Args:
            student_output (BxD): backbone output of student
        """

        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        
        return loss
    
    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I
    
class RedundancyReductionViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(RedundancyReductionViewExpanded, self).__init__()
        """Redundancy reduction loss for multiple views"""
        
        self.num_views = num_views
        self.RedundancyReduction = RedundancyReduction()

    def forward(self, episodes_vectors):
        loss = 0
        for t in range(self.num_views-1): # t is the view index
            loss += self.RedundancyReduction(episodes_vectors[:,0], episodes_vectors[:,t+1])
        loss = loss / (self.num_views-1)
        return loss
    
class RedundancyReduction(torch.nn.Module):
    """Redundancy reduction loss from Barlow Twins"""
    
    def __init__(self):
        super().__init__()

    def forward(self, z_a, z_b):
        """
        Args:
            z_a (BxD): projection of view A
            z_b (BxD): projection of view B
        """
        # batch norm with affine=False
        B, D = z_a.shape
        bn = torch.nn.BatchNorm1d(D, affine=False).to(z_a.device)
        z_a = bn(z_a)
        z_b = bn(z_b)

        # empirical cross-correlation matrix
        c = z_a.T @ z_b  # DxD
        c.div_(B)

        # loss
        off_diag = off_diagonal(c).pow(2).sum()
        return off_diag

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    