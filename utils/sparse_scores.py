import torch

def sparse_scores(
        anchor_activations: torch.Tensor,
        positive_activations: torch.Tensor,
):
    positive_scores = torch.sum(anchor_activations * positive_activations, dim=1)
    return {
        "positive_scores": positive_scores,
    }
