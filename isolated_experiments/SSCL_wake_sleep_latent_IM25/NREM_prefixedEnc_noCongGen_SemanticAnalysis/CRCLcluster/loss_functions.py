import torch
import torch.nn as nn
import math

class CRCL_PCLoss(nn.Module):
    def __init__(self, batch_size, alpha, device):
        super(CRCL_PCLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        # loss to avoid cluster collapse
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        # smooth probabilities
        C = c_i.shape[1]
        c_i = (1.0 - 0.01) * c_i + 0.01 * (1.0 / C)
        c_j = (1.0 - 0.01) * c_j + 0.01 * (1.0 / C)

        # loss to pull positive samples together and negative samples apart
        N = 2 * self.batch_size
        c = torch.cat((c_i, c_j), dim=0)

        sim = torch.matmul(c, c.T).log()
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + self.alpha*ne_loss
    