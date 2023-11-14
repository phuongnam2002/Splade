import torch
import torch.nn as nn


class FlopsScheduler:
    def __init__(self, weight: float = 3e-5, steps: int = 10000):
        self._weight = weight
        self.weight = 0
        self._steps = 0
        self.steps = steps

    def step(self) -> None:
        if self._steps >= self.steps:
            return
        self._steps += 1
        self.weight = self._weight * (self._steps / self.steps) ** 2

    def get(self):
        return self.weight


class Flops(nn.Module):
    """
        Flops loss, act as regularization loss over sparse activations.
        Example:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            from sparsembed import model, utils, losses
            from pprint import pprint as print
            import torch

            model = model.Splade(
                model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
                tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
            )

             anchor_activations = model(["Sports", "Music"])

             positive_activations = model(["Sports", "Music"])

             losses.Flops()(
                    anchor_activations=anchor_activations["sparse_activations"],
                    positive_activations=positive_activations["sparse_activations"],
             )
             tensor(value)
    """

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            anchor_activations: torch.Tensor,
            positive_activations: torch.Tensor,
            threshold: float = 10.0
    ) -> torch.Tensor:
        activations = torch.cat([anchor_activations, positive_activations], dim=0)

        return torch.abs(threshold - torch.sum(torch.mean(torch.abs(activations), dim=0) ** 2, dim=0))
