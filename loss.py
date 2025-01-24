
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=0.2, learnable_temperature=True,
                  regularization=1e-4):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.regularization = regularization

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        else:
            self.temperature = temperature

    def forward(self, anchor, aug_anchor, hard_negatives):
        batch_size, k, embedding_dim = hard_negatives.size()
        positive_similarity = F.cosine_similarity(anchor, aug_anchor, dim=-1)

        anchor_expanded = anchor.unsqueeze(1).expand(-1, k, -1)  # (batch_size, k, embedding_dim)
        negative_similarity = F.cosine_similarity(anchor_expanded, hard_negatives, dim=-1)

        mean_negative_similarity = negative_similarity.mean(dim=1)


        loss = F.relu(positive_similarity - mean_negative_similarity + self.margin) / self.temperature

        if isinstance(self.temperature, nn.Parameter):
            loss = loss.mean() + self.regularization * self.temperature.abs()
        else:
            loss = loss.mean()

        return loss
