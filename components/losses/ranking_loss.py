import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    """
        Ranking loss.
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

             scores = utils.sparse_scores(
                anchor_activations=queries_activations["sparse_activations"],
                positive_activations=positive_activations["sparse_activations"],
            )
            losses.Ranking()(**scores)
            tensor(value)
        """

    def __init__(self):
        super().__init__()
        self.activation = nn.LogSoftmax(dim=1)

    def __call__(
            self,
            positive_scores: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.stack([positive_scores], dim=1)

        loss = torch.index_select(
            input=self.activation(scores),
            dim=1,
            index=torch.zeros(1, dtype=torch.int64, device=scores.device)
        ).mean()

        return torch.clip(loss, min=0.0, max=1.0)
