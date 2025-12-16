"""
Frozen BERT sentence embeddings for classic classifiers.

Supports multiple pooling strategies without any fine-tuning:
- cls: [CLS] from last layer
- mean: masked token mean on last layer
- last4_mean: mean-pool the average of last 4 layers
- last4_cls_concat: concatenate [CLS] from the last 4 layers (4*hidden)
"""
from typing import Iterable, List, Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


EmbedStrategy = Literal["cls", "mean", "last4_mean", "last4_cls_concat"]


def _masked_mean(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings with attention mask."""
    mask = attention_mask.unsqueeze(-1)  # (b, seq, 1)
    masked = token_embeddings * mask
    summed = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    return summed / lengths


def _pool(
    last_hidden: torch.Tensor,
    hidden_states: List[torch.Tensor],
    attention_mask: torch.Tensor,
    strategy: EmbedStrategy,
) -> torch.Tensor:
    if strategy == "cls":
        return last_hidden[:, 0]
    if strategy == "mean":
        return _masked_mean(last_hidden, attention_mask)
    if strategy == "last4_mean":
        stacked = torch.stack(hidden_states[-4:], dim=0)  # (4, b, seq, h)
        merged = stacked.mean(dim=0)  # (b, seq, h)
        return _masked_mean(merged, attention_mask)
    if strategy == "last4_cls_concat":
        cls_list = [hs[:, 0] for hs in hidden_states[-4:]]  # 4 x (b, h)
        return torch.cat(cls_list, dim=-1)
    raise ValueError(f"Unknown strategy: {strategy}")


def embedding_dim(hidden_size: int, strategy: EmbedStrategy) -> int:
    if strategy == "last4_cls_concat":
        return hidden_size * 4
    return hidden_size


class BERTEmbedder:
    """
    Frozen BERT encoder that turns text into sentence embeddings.

    Example:
        encoder = BERTEmbedder(model_name="bert-base-uncased", max_length=64)
        vecs = encoder.encode(["hello world"], strategy="last4_mean")
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 64, device: str | None = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        # ensure gradients stay off
        for p in self.model.parameters():
            p.requires_grad = False

    def encode(
        self,
        texts: Iterable[str],
        strategy: EmbedStrategy = "cls",
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode a list/iterable of texts into a numpy matrix."""
        all_vecs: List[np.ndarray] = []
        texts_list: List[str] = list(texts)

        with torch.no_grad():
            for i in range(0, len(texts_list), batch_size):
                batch = texts_list[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                outputs = self.model(**enc)
                last_hidden = outputs.last_hidden_state  # (b, seq, h)
                hidden_states = list(outputs.hidden_states)  # len L+1

                pooled = _pool(last_hidden, hidden_states, enc["attention_mask"], strategy)
                all_vecs.append(pooled.cpu().float().numpy())

        return np.vstack(all_vecs) if all_vecs else np.empty((0, embedding_dim(self.model.config.hidden_size, strategy)))


__all__ = ["BERTEmbedder", "EmbedStrategy", "embedding_dim"]
