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
        flattened_predictions = predictions.view(-1, predictions.shape[-1])

        # multiply the matrix by its transpose, two tokens that are the same will raise the similarity score
        similarity = torch.matmul(flattened_predictions, flattened_predictions.T)

        # need to only use the off diagonals, zero the diagonals
        identity_matrix = torch.eye(similarity.shape[0]).to(logits.device)
        repeat_loss = (similarity * (1 - identity_matrix)).mean()

        return -target_probabilities.mean() + (repeat_loss * SIMILARITY_WEIGHT)
