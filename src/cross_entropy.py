import torch
import torch.nn as nn

from constants import SIMILARITY_WEIGHT

class CustomEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        soft_maxed = torch.nn.functional.log_softmax(logits, dim=-1)

        # for each token, zero out the wrong answers
        mask = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1])
        target_probabilities = (soft_maxed * mask).sum(dim=-1)

        # get a long list of tokens (size: sequence length)
        predictions = torch.softmax(logits, dim=-1)
        entropy = -(predictions * torch.log(predictions + 0.00001)).sum(dim=-1)

        return -target_probabilities.mean() + (entropy.mean() * SIMILARITY_WEIGHT)
