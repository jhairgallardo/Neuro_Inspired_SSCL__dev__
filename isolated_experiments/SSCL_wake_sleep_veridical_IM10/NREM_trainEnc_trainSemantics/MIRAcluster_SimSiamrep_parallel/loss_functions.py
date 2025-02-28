import torch
import torch.nn.functional as F

class SwapLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(SwapLossViewExpanded, self).__init__()
        self.N = num_views

    def forward(self, logits, targets):
        loss = 0
        for t in range(self.N-1):
            loss0 = - (targets[:,t+1] * F.log_softmax(logits[:,0], dim=1)).sum(dim=1).mean()
            loss1 = - (targets[:,0] * F.log_softmax(logits[:,t+1], dim=1)).sum(dim=1).mean()
            temp = ( loss0 + loss1 ) / 2.
            loss = loss + temp
        loss = loss / (self.N-1)
        return loss
    

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
    


    