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