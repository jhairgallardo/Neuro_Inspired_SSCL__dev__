import torch
import torch.nn.functional as F
import einops

class TwistLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4, tau=1):
        super(TwistLossViewExpanded, self).__init__()
        self.tau = tau
        self.N = num_views

    def forward(self, episodes_logits):
        episodes_probs = F.softmax(episodes_logits, dim=1)
        episodes_probs = einops.rearrange(episodes_probs, '(b v) c -> b v c', v=self.N).contiguous()
        episodes_sharp_probs = F.softmax(episodes_logits/self.tau, dim=1)
        episodes_sharp_probs = einops.rearrange(episodes_sharp_probs, '(b v) c -> b v c', v=self.N).contiguous()

        consis_loss = 0
        sharp_loss = 0
        div_loss = 0

        for t in range(self.N):
            if t < self.N-1:
                SKL = 0.5 * (self.KL(episodes_probs[:,0], episodes_probs[:,t+1]) + self.KL(episodes_probs[:,t+1], episodes_probs[:,0])) # Simetrized KL anchor based
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


class KoLeoLossViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(KoLeoLossViewExpanded, self).__init__()
        self.N = num_views
        self.KoLeo = KoLeoLoss()

    def forward(self, episodes_logits):
        episodes_logits_expanded = einops.rearrange(episodes_logits, '(b v) c -> b v c', v=self.N).contiguous()

        loss = 0
        for t in range(self.N): # t is the view index
            loss += self.KoLeo(episodes_logits_expanded[:,t])
        loss = loss / self.N

        return loss


class KoLeoLoss(torch.nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = torch.nn.PairwiseDistance(2, eps=1e-8)

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

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        #with torch.cuda.amp.autocast(enabled=False):
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        
        return loss