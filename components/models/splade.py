import torch
import torch.nn as nn
from typing import Dict, Tuple
from transformers import (
    RobertaForMaskedLM,
    PreTrainedTokenizer,
    PretrainedConfig,
    RobertaPreTrainedModel,
)


class Splade(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
    ]

    def __init__(self, config: PretrainedConfig, tokenizer: PreTrainedTokenizer, args):
        super().__init__(config)

        self.config = config
        self.args = args

        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM(config)

        self.activation = nn.GELU().to(self.args.device)

        self.model.config.output_hidden_states = True

    def encode(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_ids, attention_mask=attention_mask)
        return output.logits, output.hidden_states[0]

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        logits, _ = self.encode(input_ids=input_ids, attention_mask=attention_mask)

        activations = self._get_activation(logits=logits)

        if self.args.k_tokens is not None:
            activations = self._update_activations(
                **activations,
                k_tokens=self.args.k_tokens
            )

        return {"sparse_activations": activations["sparse_activations"]}

    def _get_activation(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"sparse_activations": torch.amax(torch.log1p(self.activation(logits)), dim=1)}

    def _filter_activations(
            self,
            sparse_activations: torch.Tensor,
            k_tokens: int = 256
    ):
        """
            Among the set of activations, select the ones with a score > 0.
        """
        scores, activations = torch.topk(input=sparse_activations, k=k_tokens, dim=-1)
        return [
            torch.index_select(
                activation, dim=-1, index=torch.nonzero(score, as_tuple=True)[0]
            )
            for score, activation in zip(scores, activations)
        ]

    def _update_activations(
            self,
            sparse_activations: torch.Tensor,
            k_tokens: int = 256
    ):
        activations = torch.topk(input=sparse_activations, k=k_tokens, dim=1).indices

        # Set value of max sparse_activations which are not in top_k to 0.
        sparse_activations = sparse_activations * torch.zeros(
            (sparse_activations.shape[0], sparse_activations.shape[1]), dtype=int
        ).to(self.args.device).scatter_(dim=1, index=activations.long(), value=1)

        return {
            "activations": activations,
            "sparse_activations": sparse_activations
        }
