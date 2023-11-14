import os
import torch
from typing import List

from components.models.splade import Splade
from components.loaders.utils import convert_text_to_features


class SpladeRetriever:
    def __init__(
            self,
            key: str,
            on: str,
            model: Splade,
            args,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        self.args = args
        self.key = key
        self.on = [on] if isinstance(on, str) else on

        self.model = model
        self.vocab_size = len(model.tokenizer.get_vocab())

        # Mapping between sparse matrix index and document keys
        self.sparse_matrix = None
        self.document_keys = {}

        self.documents_embeddings, self.documents_activations = [], []

    def add(
            self,
            documents: List,
            bath_size: int = 32,
            **kwargs
    ):
        """
            Add new documents to the retriever
            Computes documents embeddings and activations and update the sparse matrix.
        """
        for X in self._to_batch(documents, batch_size=bath_size):
            sparse_matrix = self._build_index(
                X=[" ".join(document[field] for field in self.on) for document in X],
                **kwargs
            )

            self.sparse_matrix = (
                sparse_matrix.T
                if self.sparse_matrix is None
                else torch.cat([self.sparse_matrix, sparse_matrix.T], dim=1)
            )

            self.document_keys = {
                **self.document_keys,
                **{
                    len(self.document_keys) + index: document[self.key]['id']
                    for index, document in enumerate(X)
                }
            }

        return self

    def __call__(
            self,
            q: List[str],
            topk: int = 100,
            batch_size: int = 32,
            **kwargs
    ):
        """
            Retrial Documents
        """
        q = [q] if isinstance(q, str) else q
        ranked = []

        for X in self._to_batch(q, batch_size=batch_size):
            sparse_matrix = self._build_index(
                X=X,
                **kwargs
            )

            sparse_scores = (sparse_matrix @ self.sparse_matrix).to_dense()

            ranked += self._rank(
                sparse_scores=sparse_scores,
                topk=topk
            )

        return ranked

    def _rank(
            self,
            sparse_scores: torch.Tensor,
            topk: int = 100
    ):
        sparse_scores, sparse_matchs = torch.topk(
            input=sparse_scores, k=min(topk, len(self.document_keys)), dim=-1
        )

        sparse_scores = sparse_scores.tolist()
        sparse_matchs = sparse_matchs.tolist()

        return [
            [
                {
                    'id': self.document_keys[document],
                    "similarity": score,
                }
                for score, document in zip(query_scores, query_matchs)
            ]
            for query_scores, query_matchs in zip(sparse_scores, sparse_matchs)
        ]

    def _build_index(
            self,
            X: List[str],
            **kwargs
    ):
        inputs_ids = []
        attention_masks = []

        for x in X:
            input_ids, attention_mask = convert_text_to_features(
                text=x,
                max_seq_len=self.args.max_seq_len_document,
                tokenizer=self.model.tokenizer
            )

            inputs_ids.append(input_ids)
            attention_masks.append(attention_mask)

        inputs_ids = torch.tensor(inputs_ids, dtype=torch.long, device=self.args.device)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long, device=self.args.device)

        inputs = {
            "input_ids": inputs_ids,
            "attention_mask": attention_masks
        }

        with torch.no_grad():
            batch_embeddings = self.model(**inputs)
        return batch_embeddings['sparse_activations'].to_sparse_coo()

    @staticmethod
    def _to_batch(X: List, batch_size: int) -> List:
        """
            Convert input list to batch
        """
        for X in [X[pos: pos + batch_size] for pos in range(0, len(X), batch_size)]:
            yield X
