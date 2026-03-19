"""
forked from https://github.com/datawhalechina/torch-rechub.git
"""

from typing import Optional, cast

import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.utilities


class RandomNormal:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim, padding_idx=0):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed


class RandomUniform:
    def __init__(self, minval=0.0, maxval=1.0):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, vocab_size, embed_dim, padding_idx=0):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.uniform_(embed.weight, self.minval, self.maxval)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed


class XavierNormal:
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim, padding_idx=0):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.xavier_normal_(embed.weight, self.gain)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed


class XavierUniform:
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim, padding_idx=0):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.xavier_uniform_(embed.weight, self.gain)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed


class BaseFeature:
    def __init__(self, name):
        self.name = name


class DenseFeature(BaseFeature):
    """Dense float-vector feature projected to model space by downstream model layers."""

    def __init__(self, name: str, input_dim: int, embed_dim: int):
        self.name = name
        self.input_dim = input_dim
        self.embed_dim = embed_dim

    def __repr__(self):
        return f"<DenseFeature {self.name} with shape ({self.input_dim} -> {self.embed_dim})>"


class SequenceFeature(BaseFeature):
    """The Feature Class for Sequence feature or multi-hot feature.
    In recommendation, there are many user behaviour features which we want to take the sequence model
    and tag featurs (multi hot) which we want to pooling. Note that if you use this feature, you must padding
    the feature value before training.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        pooling (str): pooling method, support `["mean", "sum", "concat"]` (default=`"mean"`)
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(
        self,
        name,
        vocab_size,
        embed_dim,
        pooling="mean",
        shared_with=None,
        padding_idx=0,
        initializer=RandomNormal(0, 0.0001),
    ):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            raise ValueError(f"embed_dim must be specified for SequenceFeature {self.name}.")
        else:
            self.embed_dim = embed_dim
        self.pooling = pooling
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f"<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim}; shared_with={self.shared_with})>"

    def get_embedding_layer(self):
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class SparseFeature(BaseFeature):
    """The Feature Class for Sparse feature.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(
        self,
        name,
        vocab_size,
        embed_dim,
        shared_with=None,
        padding_idx=0,
        initializer=RandomNormal(0, 0.0001),
    ):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            raise ValueError(f"embed_dim must be specified for SparseFeature {self.name}.")
        else:
            self.embed_dim = embed_dim
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f"<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim}; shared_with={self.shared_with})>"

    def get_embedding_layer(self):
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class WeightedMultiHotFeature(BaseFeature):
    def __init__(
        self,
        name,
        sparse_vocab_size: int,
        embed_dim: int,
        pooling="weighted_concat",  # or "weighted_sum"
        shared_with=None,
        padding_idx=0,
        initializer=RandomNormal(0, 0.0001),
    ):
        self.name = name
        self.sparse_vocab_size = sparse_vocab_size
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f"<WeightedMultiHotFeature {self.name} with Embedding shape ({self.sparse_vocab_size}, {self.embed_dim}; shared_with={self.shared_with})>"

    def get_embedding_layer(self):
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.sparse_vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class EmbeddingLayer(nn.Module):
    def __init__(self, features: list[BaseFeature]):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        for f in features:
            if (
                (isinstance(f, SequenceFeature) and f.shared_with is None)
                or (isinstance(f, SparseFeature) and f.shared_with is None)
                or (isinstance(f, WeightedMultiHotFeature) and f.shared_with is None)
            ):
                self.embed_dict[f.name] = f.get_embedding_layer()

    def forward(self, x: dict, features: Optional[list[BaseFeature]], squeeze_dim=False):
        embs = []
        if not features:
            features = self.features
        if isinstance(features, BaseFeature):
            features = [features]

        def _pool_sequence(embedded, ids, feature: SequenceFeature):
            mask = (ids != feature.padding_idx).float().unsqueeze(-1)
            if feature.pooling == "sum":
                return (embedded * mask).sum(dim=1)
            if feature.pooling == "mean":
                denom = mask.sum(dim=1).clamp_min(1.0)
                return (embedded * mask).sum(dim=1) / denom
            if feature.pooling == "concat":
                return (embedded * mask).flatten(start_dim=1)
            raise ValueError(f"Sequence pooling supports ['sum', 'mean', 'concat'], got {feature.pooling}.")

        def _pool_weighted(embedded, ids, weights, feature: WeightedMultiHotFeature):
            mask = (ids != feature.padding_idx).float().unsqueeze(-1)
            if weights is None:
                weights = torch.ones_like(ids, dtype=embedded.dtype)
            else:
                weights = weights.to(dtype=embedded.dtype)
            weight_term = weights.unsqueeze(-1) * mask

            if feature.pooling == "weighted_sum":
                return (embedded * weight_term).sum(dim=1)
            if feature.pooling == "weighted_concat":
                return (embedded * weight_term).flatten(start_dim=1)
            raise ValueError(
                f"Weighted multi-hot pooling supports ['weighted_sum', 'weighted_concat'], got {feature.pooling}."
            )

        for f in features:
            shared_with = getattr(f, "shared_with", None)
            emb_name = f.name if shared_with is None else shared_with
            if emb_name not in self.embed_dict and not isinstance(f, DenseFeature):
                raise KeyError(f"Embedding table for feature '{emb_name}' not found.")

            if isinstance(f, SparseFeature):
                ids = x[f.name].long()
                try:
                    feat_emb = self.embed_dict[emb_name](ids)
                except Exception as e:
                    print(f"ids: {ids}")
                    print(f"Embedding layer: {self.embed_dict[emb_name]}")
                    print(f"type: {type(self.embed_dict[emb_name])}")
                    raise ValueError(f"Error parsing input for SparseFeature '{f.name}': {e}")

                if feat_emb.dim() == 3:
                    feat_emb = feat_emb.squeeze(1)
                embs.append(feat_emb.unsqueeze(1))  # [B, 1, D]

            elif isinstance(f, SequenceFeature):
                ids = x[f.name].long()
                seq_emb = self.embed_dict[emb_name](ids)
                pooled = _pool_sequence(seq_emb, ids, f)
                embs.append(pooled.unsqueeze(1))  # [B, 1, D_pooling] after pooling

            elif isinstance(f, WeightedMultiHotFeature):
                raw_value = x[f.name]
                try:
                    ids, weights = raw_value.get("ids"), raw_value.get("weights")
                except Exception as e:
                    raise ValueError(f"Error parsing input for WeightedMultiHotFeature '{f.name}': {e}")
                ids = ids.long()
                wm_emb = self.embed_dict[emb_name](ids)
                pooled = _pool_weighted(wm_emb, ids, weights, f)
                embs.append(pooled.unsqueeze(1))  # [B, 1, D_pooling] after pooling
                
            elif isinstance(f, DenseFeature):
                dense_input = x[f.name].float()
                if dense_input.dim() == 2 and dense_input.size(1) == f.input_dim:
                    embs.append(dense_input.unsqueeze(1))  # [B, 1, D_in], return original dense vector
                else:
                    raise ValueError(
                        f"DenseFeature '{f.name}' expects input shape [B, {f.input_dim}], got {dense_input.shape}."
                    )

        if len(embs) == 0:
            return torch.empty(
                x[next(iter(x))].size(0), 0, device=x[next(iter(x))].device
            )  # Return empty tensor if no features

        if squeeze_dim:
            return torch.cat([e.flatten(start_dim=1) for e in embs], dim=1)  # [B, sum(D_i)]

        embed_dims = {e.size(-1) for e in embs}
        if len(embed_dims) != 1:
            raise ValueError(
                "Feature embedding dimensions are inconsistent; use squeeze_dim=True for flattened output."
            )
        return torch.cat(embs, dim=1)  # [B, N, D]


class CTRMetrics(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        _ = kwargs
        # store detached tensors to avoid holding graph
        cast(list[torch.Tensor], self.preds).append(preds.detach())
        cast(list[torch.Tensor], self.targets).append(targets.detach())

    def compute(self):
        preds = torchmetrics.utilities.dim_zero_cat(cast(list[torch.Tensor], self.preds))
        targets = torchmetrics.utilities.dim_zero_cat(cast(list[torch.Tensor], self.targets))

        preds = preds.float().view(-1)
        targets = targets.float().view(-1)

        # For metrics that expect probabilities in [0,1]
        probs = preds.clamp(0.0, 1.0)
        pred_label = (probs >= 0.5).to(torch.int32)
        true_label = targets.to(torch.int32)

        out: dict[str, torch.Tensor] = {}

        # torchmetrics.functional.* return type annotations may include None in some stubs;
        # at runtime they return tensors for valid inputs.
        out["auc"] = cast(torch.Tensor, torchmetrics.functional.auroc(probs, true_label, task="binary"))
        out["acc"] = cast(torch.Tensor, torchmetrics.functional.accuracy(pred_label, true_label, task="binary"))
        out["precision"] = cast(torch.Tensor, torchmetrics.functional.precision(pred_label, true_label, task="binary"))
        out["recall"] = cast(torch.Tensor, torchmetrics.functional.recall(pred_label, true_label, task="binary"))
        out["f1"] = cast(torch.Tensor, torchmetrics.functional.f1_score(pred_label, true_label, task="binary"))

        # logloss / BCE
        eps = 1e-8
        out["logloss"] = -(targets * torch.log(probs + eps) + (1.0 - targets) * torch.log(1.0 - probs + eps)).mean()
        return out
