import torch
import torch.nn.functional as F
import einops

class TwistPriorLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4, tau=1):
        super(TwistPriorLossViewExpanded, self).__init__()
        self.tau = tau
        self.N = num_views

    def forward(self, episodes_logits):
        episodes_sharp_probs = F.softmax(episodes_logits/self.tau, dim=1)
        episodes_sharp_probs = einops.rearrange(episodes_sharp_probs, '(b v) c -> b v c', v=self.N).contiguous()

        consis_loss = 0
        sharp_loss = 0
        prior_loss = 0

        for t in range(self.N):

            if t < self.N-1:
                SKL = 0.5 * (self.KL(episodes_sharp_probs[:,0], episodes_sharp_probs[:,t+1]) + self.KL(episodes_sharp_probs[:,t+1], episodes_sharp_probs[:,0])) # Simetrized KL anchor based
                consis_loss += SKL

            sharp_loss += self.entropy(episodes_sharp_probs[:,t]).mean() #### Sharpening loss

            mean_across_episodes = episodes_sharp_probs[:,t].mean(dim=0)
            # prior_dist = _uniform_distribution(mean_across_episodes.size(0), mean_across_episodes.device)
            prior_dist = _power_law_distribution(mean_across_episodes.size(0), 0.25, mean_across_episodes.device)
            prior_loss += self.KL(prior_dist, mean_across_episodes, dim=0) #### Diversity loss

        consis_loss = consis_loss / (self.N-1) # mean over views
        consis_loss = consis_loss.mean() # mean over episodes
        sharp_loss = sharp_loss / self.N
        prior_loss = prior_loss / self.N

        return consis_loss, sharp_loss, prior_loss

    def KL(self, probs1, probs2, eps = 1e-5, dim=1):
        kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum(dim=dim)
        return kl

    def entropy(self, probs, eps = 1e-5, dim=1):
        H = - (probs * (probs + eps).log()).sum(dim=dim)
        return H
    
def _power_law_distribution(size: int, exponent: float, device: torch.device):
    """Returns a power law distribution summing up to 1.

    Args:
        size:
            The size of the distribution.
        exponent:
            The exponent for the power law distribution.
        device:
            The device to create tensor on.

    Returns:
        A power law distribution tensor summing up to 1.
    """
    k = torch.arange(1, size + 1, device=device)
    power_dist = k ** (-exponent)
    power_dist = power_dist / power_dist.sum()
    return power_dist

def _uniform_distribution(size: int, device: torch.device):
    """Returns a uniform distribution summing up to 1.

    Args:
        size:
            The size of the distribution.
        device:
            The device to create tensor on.

    Returns:
        A uniform distribution tensor summing up to 1.
    """
    return torch.ones(size, device=device) / size