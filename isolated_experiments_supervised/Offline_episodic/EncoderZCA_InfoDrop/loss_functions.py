import torch
import torch.nn.functional as F

class CrossEntropyViewExpanded(torch.nn.Module):
    def __init__(self, num_views=4):
        super(CrossEntropyViewExpanded, self).__init__()
        """Cross entropy loss for multiple views"""
        
        self.num_views = num_views
        self.crossentropy = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, episodes_vectors, episodes_labels):
        loss = 0
        for t in range(self.num_views): # t is the view index
            loss += self.crossentropy(episodes_vectors[:,t], episodes_labels[:,t])
        loss = loss / self.num_views
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
    


    