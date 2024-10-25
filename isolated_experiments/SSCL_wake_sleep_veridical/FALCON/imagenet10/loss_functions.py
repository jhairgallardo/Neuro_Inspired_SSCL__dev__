import torch
import torch.nn.functional as F

class KLLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(KLLossViewExpanded, self).__init__()
        self.N = num_views
        self.crossentropy = torch.nn.CrossEntropyLoss()
    
    def KL(self, probs1, probs2, eps = 1e-8): # probs are 1D tensors
        kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum()
        return kl

    def forward(self, logits, c):

        # Push all views mean probs to prior
        # loss = 0
        # for t in range(self.N):
        #     probs_mean = F.softmax(logits[:,t], dim=1).mean(dim=0)
        #     probs_uniform = 1/c * torch.ones_like(probs_mean)
        #     loss += self.KL(probs_mean, probs_uniform)
        # loss = loss / self.N

        # Only push first view mean probs to prior
        probs_mean = F.softmax(logits[:,0], dim=1).mean(dim=0)
        probs_uniform = 1/c * torch.ones_like(probs_mean)
        loss = self.KL(probs_mean, probs_uniform)
        
        return loss  
    
class CrossEntropyLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(CrossEntropyLossViewExpanded, self).__init__()
        self.N = num_views
        self.crossentropy = torch.nn.CrossEntropyLoss()

    def forward(self, logits):
        loss = 0
        for t in range(self.N-1):
            target_probs = F.softmax(logits[:,t+1], dim=1)
            loss += self.crossentropy(logits[:,0], target_probs)
        loss = loss / (self.N-1)
        return loss

    


    