import torch
import torch.nn.functional as F

class EntropyRegularizerExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(EntropyRegularizerExpanded, self).__init__()
        self.N = num_views

    def forward(self, episodes_logits, entropy_threshold):
        episodes_probs = F.softmax(episodes_logits, dim=-1)
        entropy_reg_loss = 0
        entropy_val_mean = 0

        for t in range(self.N):
            mean_across_episodes = episodes_probs[:,t].mean(dim=0)
            entropy_val = self.entropy(mean_across_episodes, dim=0)
            entropy_reg_loss += torch.abs(entropy_val - entropy_threshold)
            entropy_val_mean += entropy_val
        entropy_reg_loss = entropy_reg_loss / self.N # Mean across number of views
        entropy_val_mean = entropy_val_mean / self.N # Mean across number of views

        return entropy_reg_loss, entropy_val_mean

    def entropy(self, probs, eps = 1e-5, dim=1):
        H = - (probs * (probs + eps).log()).sum(dim=dim)
        return H


class KoLeoLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(KoLeoLossViewExpanded, self).__init__()
        """Koleo loss for multiple views"""
        
        self.num_views = num_views
        self.KoLeo = KoLeoLoss()

    def forward(self, episodes_logits):
        loss = 0
        for t in range(self.num_views): # t is the view index
            loss += self.KoLeo(episodes_logits[:,t])
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


class ConsistLossCARLViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(ConsistLossCARLViewExpanded, self).__init__()
        self.N = num_views

    def forward(self, episodes_logits):
        episodes_probs = F.softmax(episodes_logits, dim=-1)

        loss = 0
        for t in range(self.N-1): # t is the view index
            loss += self.cluster_loss(episodes_probs[:,0], episodes_probs[:,t+1])
        loss = loss / (self.N-1) # I was using self.N on the oracle runs. I think it is wrong.
        return loss

    def cluster_loss(self, P1, P2, EPS=1e-12): # -log(dot(P1,P2))
        assert P1.shape == P2.shape
        dot_products = torch.einsum("nc,nc->n", [P1, P2])
        cluster_loss = -torch.log(dot_products + EPS).mean()
        return cluster_loss


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

    


    