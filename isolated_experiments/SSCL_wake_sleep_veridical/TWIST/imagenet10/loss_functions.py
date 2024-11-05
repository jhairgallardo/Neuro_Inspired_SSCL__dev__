import torch
import torch.nn.functional as F
import einops

class TwistLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4, tau=1):
        super(TwistLossViewExpanded, self).__init__()
        self.tau = tau
        self.N = num_views

    def forward(self, episodes_logits):
        # episodes_probs = F.softmax(episodes_logits, dim=1)
        # episodes_probs = einops.rearrange(episodes_probs, '(b v) c -> b v c', v=self.N).contiguous()
        episodes_sharp_probs = F.softmax(episodes_logits/self.tau, dim=1)
        episodes_sharp_probs = einops.rearrange(episodes_sharp_probs, '(b v) c -> b v c', v=self.N).contiguous()

        consis_loss = 0
        sharp_loss = 0
        div_loss = 0

        for t in range(self.N):
            if t < self.N-1:
                SKL = 0.5 * (self.KL(episodes_sharp_probs[:,0], episodes_sharp_probs[:,t+1]) + self.KL(episodes_sharp_probs[:,t+1], episodes_sharp_probs[:,0])) # Simetrized KL anchor based
                consis_loss += SKL
            sharp_loss += self.entropy(episodes_sharp_probs[:,t]).mean() #### Sharpening loss
            mean_across_episodes = episodes_sharp_probs[:,t].mean(dim=0)
            div_entropy_val = self.entropy(mean_across_episodes, dim=0)
            div_loss += div_entropy_val #### Diversity loss
        consis_loss = consis_loss / (self.N-1) # mean over views
        consis_loss = consis_loss.mean() # mean over episodes
        sharp_loss = sharp_loss / self.N
        div_loss = div_loss / self.N

        return consis_loss, sharp_loss, div_loss

    def KL(self, probs1, probs2, eps = 1e-5):
        kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum(dim=1)
        return kl

    def entropy(self, probs, eps = 1e-5, dim=1):
        H = - (probs * (probs + eps).log()).sum(dim=dim)
        return H


class ConsistLossCARLViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4, tau=1):
        super(ConsistLossCARLViewExpanded, self).__init__()
        self.tau = tau
        self.N = num_views

    def forward(self, episodes_logits):
        episodes_sharp_probs = F.softmax(episodes_logits/self.tau, dim=-1)
        episodes_sharp_probs = einops.rearrange(episodes_sharp_probs, '(b v) c -> b v c', v=self.N).contiguous()

        loss = 0
        for t in range(self.N-1): # t is the view index
            loss += self.cluster_loss(episodes_sharp_probs[:,0], episodes_sharp_probs[:,t+1])
        loss = loss / self.N
        return loss

    def cluster_loss(self, P1, P2, EPS=1e-12): # -log(dot(P1,P2))
        assert P1.shape == P2.shape
        dot_products = torch.einsum("nc,nc->n", [P1, P2])
        cluster_loss = -torch.log(dot_products + EPS).mean()
        return cluster_loss