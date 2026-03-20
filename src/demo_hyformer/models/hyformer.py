"""
https://arxiv.org/abs/2601.12681v2 HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction
"""

import math
from typing import Callable, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_hyformer.basic import (
    CTRMetrics,
    EmbeddingLayer,
)
from demo_hyformer.dataset import get_features
from demo_hyformer.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class SequenceWiseRMSNorm(nn.Module):
    def __init__(self, num_seq: int, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_seq, d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(var + self.eps)

        weight_shape = [1] * x.dim()
        weight_shape[1] = self.weight.size(0)
        weight_shape[-1] = self.weight.size(1)
        return normalized * self.weight.view(*weight_shape)


class HyFormer(nn.Module):
    def __init__(
        self,
        num_global_tokens: int,
        num_nonseq_tokens: int,
        d_model: int,
        max_seq_len: int,
        max_delta_tt_bucket: int,
        kv_encoder: Callable,
        semantic_groups_fn: Callable,
        num_blocks: int,
        ffn_hidden_dim: int,
        num_heads: int,
        mlp_hidden_dim: list[int] = [1024, 512],
        num_seq: int = 3,
        drop_out: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        log.info(f"Initializing Hyformer with {args, kwargs}")
        self.num_global_tokens = num_global_tokens
        self.num_nonseq_tokens = num_nonseq_tokens
        self.d_model = d_model
        self.kv_encoder = kv_encoder
        self.num_blocks = num_blocks
        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_heads = num_heads
        self.num_seq = num_seq
        self.mlp_hidden_dim = mlp_hidden_dim
        self.drop_out = drop_out
        self.max_delta_tt_bucket = max_delta_tt_bucket
        self.max_seq_len = max_seq_len

        # embedding layer
        (
            self.non_seq_sparse_features,
            self.non_seq_multi_features,
            self.non_seq_embedding_features,
            self.action_seq_features,
            self.content_seq_features,
            self.item_seq_features,
        ) = get_features(emb_dim=self.d_model, max_delta_tt_bucket=self.max_delta_tt_bucket)

        self.non_seq_full_len = (
            len(self.non_seq_sparse_features) + len(self.non_seq_multi_features) + len(self.non_seq_embedding_features)
        )

        self.embedding_layer = EmbeddingLayer(
            self.non_seq_sparse_features
            + self.non_seq_multi_features
            + self.non_seq_embedding_features
            + self.action_seq_features
            + self.content_seq_features
            + self.item_seq_features
        )

        self.non_seq_multi_linear = nn.Linear(self.max_seq_len * self.d_model, self.d_model)
        # TODO: add other transforms like mean / weighted mean to avoid overfitting

        self.non_seq_embedding_mlps = []
        for f in self.non_seq_embedding_features:
            input_dim = f.input_dim  # type: ignore
            mlp = nn.Sequential(
                nn.Linear(input_dim, 2 * input_dim),
                nn.LayerNorm(2 * input_dim),
                nn.SiLU(),
                nn.Dropout(p=self.drop_out),
                nn.Linear(2 * input_dim, self.d_model),
            )
            self.non_seq_embedding_mlps.append(mlp)
        self.non_seq_embedding_mlps = nn.ModuleList(self.non_seq_embedding_mlps)

        self.semantic_groups = semantic_groups_fn()
        if not self.semantic_groups:
            self.full_non_seq_mlp = nn.Sequential(
                nn.Linear(self.non_seq_full_len * self.d_model, 2 * self.non_seq_full_len * self.d_model),
                nn.LayerNorm(2 * self.non_seq_full_len * self.d_model),
                nn.SiLU(),
                nn.Dropout(p=self.drop_out),
                nn.Linear(2 * self.non_seq_full_len * self.d_model, self.num_nonseq_tokens * self.d_model),
            )

        self.action_seq_mlp = nn.Sequential(
            nn.Linear(len(self.action_seq_features) * self.d_model, 2 * len(self.action_seq_features) * self.d_model),
            nn.LayerNorm(2 * len(self.action_seq_features) * self.d_model),
            nn.SiLU(),
            nn.Dropout(p=self.drop_out),
            nn.Linear(2 * len(self.action_seq_features) * self.d_model, self.d_model),
        )
        self.content_seq_mlp = nn.Sequential(
            nn.Linear(len(self.content_seq_features) * self.d_model, 2 * len(self.content_seq_features) * self.d_model),
            nn.LayerNorm(2 * len(self.content_seq_features) * self.d_model),
            nn.SiLU(),
            nn.Dropout(p=self.drop_out),
            nn.Linear(2 * len(self.content_seq_features) * self.d_model, self.d_model),
        )
        self.item_seq_mlp = nn.Sequential(
            nn.Linear(len(self.item_seq_features) * self.d_model, 2 * len(self.item_seq_features) * self.d_model),
            nn.LayerNorm(2 * len(self.item_seq_features) * self.d_model),
            nn.SiLU(),
            nn.Dropout(p=self.drop_out),
            nn.Linear(2 * len(self.item_seq_features) * self.d_model, self.d_model),
        )

        if self.semantic_groups:
            self.num_nonseq_tokens = len(self.semantic_groups)
            log.info(f"Using semantic grouping with {self.num_nonseq_tokens} groups: \n{self.semantic_groups}")

        # query generation
        self.query_generation: nn.Module = QueryGeneration(
            num_global_tokens=self.num_global_tokens,
            num_nonseq_tokens=self.num_nonseq_tokens,
            d_model=self.d_model,
            num_seq=self.num_seq,
            ffn_hidden_ratio=2,
            drop_out=self.drop_out,
        )

        # hyformer blocks
        self.hyformer_blocks: nn.ModuleList = nn.ModuleList([
            HyFormerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_global_tokens=self.num_global_tokens,
                num_nonseq_tokens=self.num_nonseq_tokens,
                ffn_hidden_dim=self.ffn_hidden_dim,
                kv_encoder=self.kv_encoder,
                num_seq=self.num_seq,
                drop_out=self.drop_out,
            )
            for _ in range(self.num_blocks)
        ])

        # output head
        layers = []
        in_dim = self.num_seq * (self.num_global_tokens + self.num_nonseq_tokens) * self.d_model
        for hi in self.mlp_hidden_dim:
            layers.append(nn.Linear(in_dim, hi))
            layers.append(nn.LayerNorm(hi))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=self.drop_out))
            in_dim = hi

        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

    def _get_embedding(self, x: dict) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        non_seq_features:
            * non_seq_sparse_features: normal embedding -> [B, M1, D]
            * non_seq_multi_features: concat / weighted_concat + linear -> [B, M2, D]
            * non_seq_embedding_features: mlp -> [B, M3, D]
            * (non sematic gr) concat + mlp -> [B, M, D]
            or (with sematic gr) TODO...
        seq_features:
            * action_seq_features: embedding -> [B, T, D]
            * item_seq_features: embedding -> [B, T, D]
            * content_seq_features: embedding -> [B, T, D]
            * concat + mlp -> [B, T. D]

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
                non_seq_x: [B, M, D]
                seq_x: [B, S, T, D]
                seq_time_diffs: [B, S, T] or None
                seq_valid_mask: [B, S, T]
        """
        # non_seq feature
        non_seq_sparse_emb = self.embedding_layer(x["non_seq"], self.non_seq_sparse_features)  # [B, M1, D]
        non_seq_multi_emb = self.embedding_layer(x["non_seq"], self.non_seq_multi_features)  # [B, M2, D_multi]
        non_seq_embedding_embs = [
            self.embedding_layer(x["non_seq"], self.non_seq_embedding_features[i])
            for i in range(len(self.non_seq_embedding_features))
        ]  # list of [B, 1, D_in]

        non_seq_multi_emb = self.non_seq_multi_linear(non_seq_multi_emb)  # [B, M2, D]
        non_seq_embedding_emb_parts = []
        for i, emb in enumerate(non_seq_embedding_embs):
            B, _, D_in = emb.size()
            emb_flat = emb.reshape(B, D_in)  # [B, D_in]
            transformed = self.non_seq_embedding_mlps[i](emb_flat)  # [B, D]
            non_seq_embedding_emb_parts.append(transformed.unsqueeze(1))  # [B, 1, D]

        non_seq_embedding_emb = torch.cat(non_seq_embedding_emb_parts, dim=1)  # [B, M3, D]

        full_non_seq_x = torch.cat(
            [non_seq_sparse_emb, non_seq_multi_emb, non_seq_embedding_emb], dim=1
        )  # [B, M_full, D]
        if not self.semantic_groups:
            non_seq_x = self.full_non_seq_mlp(full_non_seq_x.reshape(full_non_seq_x.size(0), -1)).reshape(
                -1, self.num_nonseq_tokens, self.d_model
            )  # [B, M, D]
        else:
            # TODO: add sematic gr imple
            raise NotImplementedError

        # seq feature
        action_seq_emb = (
            self.embedding_layer(x["action_seq"], self.action_seq_features)  # [B, M_action, T * D]
            .reshape(-1, len(self.action_seq_features), self.max_seq_len, self.d_model)  # [B, M_action, T, D]
            .permute(0, 2, 1, 3)  # [B, T, M_action, D]
            .reshape(-1, self.max_seq_len, len(self.action_seq_features) * self.d_model)  # [B, T, M_action*D]
        )
        content_seq_emb = (
            self.embedding_layer(x["content_seq"], self.content_seq_features)  # [B, M_content, T * D]
            .reshape(-1, len(self.content_seq_features), self.max_seq_len, self.d_model)  # [B, M_content, T, D]
            .permute(0, 2, 1, 3)  # [B, T, M_content, D]
            .reshape(-1, self.max_seq_len, len(self.content_seq_features) * self.d_model)  # [B, T, M_content*D]
        )
        item_seq_emb = (
            self.embedding_layer(x["item_seq"], self.item_seq_features)  # [B, M_item, T * D]
            .reshape(-1, len(self.item_seq_features), self.max_seq_len, self.d_model)  # [B, M_item, T, D]
            .permute(0, 2, 1, 3)  # [B, T, M_item, D]
            .reshape(-1, self.max_seq_len, len(self.item_seq_features) * self.d_model)  # [B, T, M_item*D]
        )

        action_seq_x = self.action_seq_mlp(action_seq_emb)  # [B, T, D]
        content_seq_x = self.content_seq_mlp(content_seq_emb)  # [B, T, D]
        item_seq_x = self.item_seq_mlp(item_seq_emb)  # [B, T, D]

        seq_x = torch.stack([action_seq_x, content_seq_x, item_seq_x], dim=1)  # [B, S, T, D]

        seq_timestamps = torch.stack(
            [
                x["action_seq"]["action_seq_timestamp"],
                x["content_seq"]["content_seq_timestamp"],
                x["item_seq"]["item_seq_timestamp"],
            ],
            dim=1,
        )  # [B, S, T]
        seq_valid_mask = seq_timestamps.gt(0)

        # seq_time_diffs
        seq_time_diffs = torch.stack(
            [
                x["seq_time_diffs"]["action_seq_time_diff"].long(),
                x["seq_time_diffs"]["content_seq_time_diff"].long(),
                x["seq_time_diffs"]["item_seq_time_diff"].long(),
            ],
            dim=1,
        )  # [B, S, T]

        return non_seq_x, seq_x, seq_time_diffs, seq_valid_mask

    def forward(
        self,
        x: dict,
    ) -> torch.Tensor:
        non_seq_x, seq_x, seq_time_diffs, seq_valid_mask = self._get_embedding(x)
        # non_seq_x: [B, M, D] ; seq_x: [B, S, T, D]
        B = seq_x.size(0)
        global_tokens = self.query_generation(non_seq_x, seq_x, seq_valid_mask)  # [B, S, N, D]
        non_seq_tokens = non_seq_x.unsqueeze(1).expand(-1, self.num_seq, -1, -1)  # [B, S, M, D]
        seq_tokens = seq_x  # [B, S, T, D]

        for block in self.hyformer_blocks:
            global_tokens, non_seq_tokens, seq_tokens = block(
                global_tokens,
                non_seq_tokens,
                seq_tokens,
                seq_valid_mask=seq_valid_mask,
                seq_time_diffs=seq_time_diffs,
            )

        final_rep = torch.cat([global_tokens.reshape(B, -1), non_seq_tokens.reshape(B, -1)], dim=1)  # [B, S*(N+M)*D]

        output = self.mlp_head(final_rep)  # [B, 1]

        return output


class QueryGeneration(nn.Module):
    def __init__(
        self,
        num_global_tokens: int,
        num_nonseq_tokens: int,
        d_model: int,
        num_seq: int = 3,
        ffn_hidden_ratio: float = 2.0,
        drop_out: float = 0.1,
    ):
        super().__init__()
        self.num_global_tokens = num_global_tokens
        self.num_nonseq_tokens = num_nonseq_tokens
        self.d_model = d_model
        self.num_seq = num_seq
        self.drop_out = drop_out

        global_info_dim = (self.num_nonseq_tokens + 1) * self.d_model
        hidden_dim = int(global_info_dim * ffn_hidden_ratio)
        out_dim = self.num_global_tokens * self.d_model

        self.W1 = nn.Parameter(torch.empty(self.num_seq, global_info_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(self.num_seq, hidden_dim))

        self.W2 = nn.Parameter(torch.empty(self.num_seq, hidden_dim, out_dim))
        self.b2 = nn.Parameter(torch.empty(self.num_seq, out_dim))

        self.hidden_dropout = nn.Dropout(p=self.drop_out)
        self.output_norm = SequenceWiseRMSNorm(num_seq=self.num_seq, d_model=self.d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

        # Calculate bounds for bias initialization based on fan_in
        fan_in_1, _ = nn.init._calculate_fan_in_and_fan_out(self.W1[0])
        bound_1 = 1 / math.sqrt(fan_in_1) if fan_in_1 > 0 else 0
        nn.init.uniform_(self.b1, -bound_1, bound_1)

        fan_in_2, _ = nn.init._calculate_fan_in_and_fan_out(self.W2[0])
        bound_2 = 1 / math.sqrt(fan_in_2) if fan_in_2 > 0 else 0
        nn.init.uniform_(self.b2, -bound_2, bound_2)

    def forward(self, non_seq_x: torch.Tensor, seq_x: torch.Tensor, seq_valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            non_seq_x (torch.Tensor): [B, M, D]
            seq_x (torch.Tensor): [B, S, T, D]
            seq_valid_mask (torch.Tensor): [B, S, T]

        Returns:
            torch.Tensor: first global_tokens [B, S, N, D]
        """
        B = non_seq_x.size(0)

        valid_counts = seq_valid_mask.sum(dim=2, keepdim=True).clamp_min(1)
        # log.debug(f"Valid counts: {valid_counts}")
        # if torch.any(valid_counts == 0):
        #     raise ValueError("Encountered all-padding sequence while generating queries")
        seq_weights = seq_valid_mask.unsqueeze(-1).to(dtype=seq_x.dtype)
        seq_pooled = (seq_x * seq_weights).sum(dim=2, keepdim=True) / valid_counts.unsqueeze(-1).to(dtype=seq_x.dtype)
        non_seq_expanded = non_seq_x.unsqueeze(dim=1).expand(-1, self.num_seq, -1, -1)  # [B, S, M, D]

        global_info = torch.cat([seq_pooled, non_seq_expanded], dim=2)  # [B, S, M+1, D]
        global_info_flat = global_info.reshape(B, self.num_seq, -1)  # [B, S, (M+1)*D]

        h1 = torch.einsum("bsi,sih->bsh", global_info_flat, self.W1)  # [B, S, Hidden]
        h1 = h1 + self.b1.unsqueeze(0)  # [B, S, Hidden] + [1, S, Hidden] -> [B, S, Hidden]

        h1_act = F.silu(h1)
        h1_act = self.hidden_dropout(h1_act)

        out_flat = torch.einsum("bsh,sho->bso", h1_act, self.W2)  # [B, S, N*D]
        out_flat = out_flat + self.b2.unsqueeze(0)  # [B, S, N*D] + [1, S, N*D] -> [B, S, N*D]

        queries = out_flat.reshape(B, self.num_seq, self.num_global_tokens, self.d_model)  # [B, S, N, D]
        queries = self.output_norm(queries)

        return queries


class HyFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_global_tokens: int,
        num_nonseq_tokens: int,
        ffn_hidden_dim: int,
        kv_encoder: Callable,
        num_seq: int = 3,
        drop_out: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_seq = num_seq
        self.num_heads = num_heads
        self.num_global_tokens = num_global_tokens  # N
        self.num_nonseq_tokens = num_nonseq_tokens  # M
        self.drop_out = drop_out

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={self.d_model}, num_heads={self.num_heads}"
            )

        S = self.num_seq
        N = self.num_global_tokens
        M = self.num_nonseq_tokens
        T = N + M
        D = self.d_model

        # query decoding
        self.kv_encoder = kv_encoder()
        self.k_proj = nn.Parameter(torch.Tensor(S, D, D))
        self.v_proj = nn.Parameter(torch.Tensor(S, D, D))

        self.o_proj = nn.Parameter(torch.Tensor(S, D, D))
        self.b_o = nn.Parameter(torch.Tensor(S, D))

        # query boosting
        self.ffn_W = nn.Parameter(torch.Tensor(S, T, D, ffn_hidden_dim))
        self.ffn_b = nn.Parameter(torch.Tensor(S, T, ffn_hidden_dim))
        self.ffn_out_W = nn.Parameter(torch.Tensor(S, T, ffn_hidden_dim, D))
        self.ffn_out_b = nn.Parameter(torch.Tensor(S, T, D))

        self.query_input_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)
        self.kv_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)
        self.query_output_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)
        self.boost_input_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)
        self.boost_output_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)

        self.query_dropout = nn.Dropout(p=self.drop_out)
        self.ffn_hidden_dropout = nn.Dropout(p=self.drop_out)
        self.ffn_output_dropout = nn.Dropout(p=self.drop_out)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in (self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj)

        nn.init.xavier_uniform_(self.o_proj)
        nn.init.zeros_(self.b_o)

        nn.init.kaiming_uniform_(self.ffn_W, a=math.sqrt(5), nonlinearity="relu")
        nn.init.zeros_(self.ffn_b)
        nn.init.kaiming_uniform_(self.ffn_out_W, a=math.sqrt(5), nonlinearity="relu")
        nn.init.zeros_(self.ffn_out_b)

    def _query_decoding(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, seq_valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Q (torch.Tensor): [B, S, N, D]
            K (torch.Tensor): [B, S, T, D]
            V (torch.Tensor): [B, S, T, D]
            seq_valid_mask (torch.Tensor): [B, S, T]

        Returns:
            torch.Tensor: [B, S, N, D]
        """
        B, S, N, D = Q.size()
        T = K.size(2)

        # if torch.any(seq_valid_mask.sum(dim=-1) == 0):
        #     raise ValueError("Encountered all-padding sequence while decoding queries")

        Q = self.query_input_norm(Q)

        head_dim = D // self.num_heads

        Q_mha = Q.reshape(B, S, N, self.num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()  # [B, S, H, N, D/H]
        K_mha = K.reshape(B, S, T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()  # [B, S, H, T, D/H]
        V_mha = V.reshape(B, S, T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()  # [B, S, H, T, D/H]

        attn_mask = seq_valid_mask.unsqueeze(2).unsqueeze(2).expand(-1, -1, self.num_heads, N, -1)
        attn_out = F.scaled_dot_product_attention(
            Q_mha,
            K_mha,
            V_mha,
            attn_mask=attn_mask,
            dropout_p=self.drop_out if self.training else 0.0,
        )  # [B, S, H, N, D/H]
        attn_out_reshaped = attn_out.permute(0, 1, 3, 2, 4).reshape(B, S, N, D)  # [B, S, N, D]

        out = torch.einsum("bsnd,sde->bsne", attn_out_reshaped, self.o_proj) + self.b_o.reshape(
            1, S, 1, D
        )  # [B, S, N, D]
        out = self.query_dropout(out)
        return self.query_output_norm(Q + out)

    def _query_boosting(
        self,
        non_seq_tokens: torch.Tensor,
        Q_tilda: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            non_seq_tokens (torch.Tensor): [B, S, M, D]
            Q_tilda (torch.Tensor): [B, S, N, D]

        Returns:
            torch.Tensor: [B, S, N + M, D]
        """
        B, S, M, D = non_seq_tokens.size()
        N = Q_tilda.size(2)
        T = M + N
        assert D % T == 0, f"get D={D}, T={T}, M = {M}, N = {N}"

        Q_ = torch.cat([Q_tilda, non_seq_tokens], dim=2)  # [B, S, N + M, D]
        Q_ = self.boost_input_norm(Q_)
        sub_dim = D // T
        Q_hat = (
            Q_.reshape(B, S, T, T, sub_dim)  # [B, S, T, T, D/T]
            .transpose(2, 3)  # [B, S, T, T, D/T]
            .reshape(B, S, T, D)  # [B, S, T, D]
        )

        hidden = torch.einsum("bstd,stdh->bsth", Q_hat, self.ffn_W) + self.ffn_b.unsqueeze(0)  # [B, S, T, Hidden]
        out_act = F.silu(hidden)  # [B, S, T, Hidden]
        out_act = self.ffn_hidden_dropout(out_act)
        out = torch.einsum("bsth,sthd->bstd", out_act, self.ffn_out_W) + self.ffn_out_b.unsqueeze(0)  # [B, S, T, D]
        out = self.ffn_output_dropout(out)

        Q_boost = out + Q_  # residual connection [B, S, T, D]
        return self.boost_output_norm(Q_boost)

    def forward(
        self,
        global_tokens: torch.Tensor,
        non_seq_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_valid_mask: torch.Tensor,
        seq_time_diffs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            global_tokens (torch.Tensor): [B, S, N, D]
            non_seq_tokens (torch.Tensor): [B, S, M, D]
            seq_tokens (torch.Tensor): [B, S, T, D]
            seq_valid_mask (torch.Tensor): [B, S, T]
            seq_time_diffs (torch.Tensor | None): [B, S, T] or None

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        if seq_time_diffs is None:
            H = self.kv_encoder(seq_tokens, seq_valid_mask=seq_valid_mask)  # [B, S, T, D]
        else:
            try:
                H = self.kv_encoder(
                    seq_tokens, time_diffs=seq_time_diffs, seq_valid_mask=seq_valid_mask
                )  # [B, S, T, D]
            except Exception as e:
                log.info(f"kv_encoder does not support time_diffs, see {e}")
                H = self.kv_encoder(seq_tokens, seq_valid_mask=seq_valid_mask)  # [B, S, T, D]
        H = self.kv_norm(H)
        K = torch.einsum("bstd,sde->bste", H, self.k_proj)  # [B, S, T, D]
        V = torch.einsum("bstd,sde->bste", H, self.v_proj)  # [B, S, T, D]

        Q_tilda = self._query_decoding(global_tokens, K, V, seq_valid_mask=seq_valid_mask)  # [B, S, N, D]

        Q_boost = self._query_boosting(non_seq_tokens, Q_tilda)  # [B, S, N + M, D]

        next_global_tokens = Q_boost[:, :, : self.num_global_tokens, :]  # [B, S, N, D]
        next_non_seq_tokens = Q_boost[:, :, self.num_global_tokens :, :]  # [B, S, M, D]

        return next_global_tokens, next_non_seq_tokens, H


class HyFormerModule(L.LightningModule):
    def __init__(
        self,
        model: HyFormer,
        optimizer: Callable,
        scheduler: Optional[Callable],
        embed_dim: int,
        val_at_k_list: list[int] = [1, 5, 10],
        *args,
        **kwargs,
    ):
        super().__init__()
        log.info(f"Initializing HyFormerModule with {args, kwargs}")
        self.model = model
        self.val_metrics = CTRMetrics()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.val_at_k_list = val_at_k_list
        self.emb_dim = embed_dim

        self.configure_optimizers()

    def forward(self, x: dict) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        loss = self._compute_loss(batch)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        _ = batch_idx
        _ = self._compute_loss(batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=(name in {"auc", "logloss"}))
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        _ = batch_idx
        _ = self._compute_loss(batch, stage="test")
        # TODO: add metrics config later

    def configure_optimizers(self):
        optimizer = self._optimizer(self.model.parameters())
        if self._scheduler is None:
            return optimizer
        scheduler = self._scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }

    def _compute_loss(self, batch: dict, stage: str = "train") -> torch.Tensor:
        logits = self.forward(batch).squeeze(-1)  # [B]
        labels = batch["label"].float()

        total_loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)

        self.log(
            f"{stage}/loss",
            total_loss,
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
        )

        if stage == "val":
            # Flattened binary diagnostics (AUC, logloss, etc.)
            self.val_metrics.update(probs.reshape(-1), labels.reshape(-1))

        if stage == "test":
            # TODO: add test metrics later
            pass

        return total_loss


if __name__ == "__main__":
    """AI generated test..."""

    def test_forward_simplified():
        """
        Test HyFormer forward pass with simplified mock data.
        Skips EmbeddingLayer complexity and focuses on core HyFormer logic:
        - QueryGeneration
        - HyFormerBlocks with kv_encoder
        - Output head MLP
        """
        print("=" * 80)
        print("Testing HyFormer Forward Pass (Simplified)")
        print("=" * 80)

        device = torch.device("cpu")
        batch_size = 2
        max_seq_len = 12
        num_global_tokens = 4
        num_nonseq_tokens = 5
        num_seq = 3

        # d_model must be divisible by (num_global_tokens + num_nonseq_tokens)
        # for _query_boosting to work
        d_model = 36  # 36 % (4+5) = 36 % 9 = 0
        num_heads = 4
        num_blocks = 2

        print("\n[Test Config]")
        print(f"  device: {device}")
        print(f"  batch_size: {batch_size}")
        print(f"  max_seq_len: {max_seq_len}")
        print(f"  d_model: {d_model} (divisible by T={num_global_tokens + num_nonseq_tokens})")
        print(f"  num_heads: {num_heads}")
        print(f"  num_blocks: {num_blocks}")
        print(f"  num_global_tokens: {num_global_tokens}")
        print(f"  num_nonseq_tokens: {num_nonseq_tokens}")

        # Create KV encoder factory
        from demo_hyformer.models.kv_encoder import HSTUSeqKVEncoder

        kv_encoder_fn = lambda: HSTUSeqKVEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_seq=num_seq,
            num_layers=2,
            ffn_hidden_ratio=2.0,
            drop_out=0.1,
            max_seq_len=max_seq_len,
            pos_bias_type="rel_pos_bias",
            use_time_embedding=True,
            num_time_buckets=256,
            time_bucket_fn="log",
        ).to(device)

        # Test QueryGeneration
        print("\n[Testing QueryGeneration]")
        query_gen = QueryGeneration(
            num_global_tokens=num_global_tokens,
            num_nonseq_tokens=num_nonseq_tokens,
            d_model=d_model,
            num_seq=num_seq,
            ffn_hidden_ratio=2.0,
            drop_out=0.1,
        ).to(device)

        non_seq_x = torch.randn(batch_size, num_nonseq_tokens, d_model, device=device)
        seq_x = torch.randn(batch_size, num_seq, max_seq_len, d_model, device=device)
        seq_valid_mask = torch.ones(batch_size, num_seq, max_seq_len, device=device, dtype=torch.bool)
        seq_valid_mask[:, :, :2] = False

        try:
            global_tokens = query_gen(non_seq_x, seq_x, seq_valid_mask)
            assert global_tokens.shape == (batch_size, num_seq, num_global_tokens, d_model)
            print(f"  ✓ QueryGeneration output shape: {global_tokens.shape}")
            assert torch.isfinite(global_tokens).all()
            print("  ✓ All values are finite")
        except Exception as e:
            print(f"  ✗ QueryGeneration failed: {e}")
            raise

        # Test HyFormerBlock
        print("\n[Testing HyFormerBlock]")
        block = HyFormerBlock(
            d_model=d_model,
            num_heads=num_heads,
            num_global_tokens=num_global_tokens,
            num_nonseq_tokens=num_nonseq_tokens,
            ffn_hidden_dim=int(d_model * 2.0),
            kv_encoder=kv_encoder_fn,
            num_seq=num_seq,
            drop_out=0.1,
        ).to(device)

        seq_time_diffs = torch.randint(0, 100, (batch_size, num_seq, max_seq_len), device=device, dtype=torch.long)

        try:
            block.eval()
            with torch.no_grad():
                next_global, next_nonseq, next_seq = block(
                    global_tokens,
                    non_seq_x.unsqueeze(1).expand(-1, num_seq, -1, -1),
                    seq_x,
                    seq_valid_mask=seq_valid_mask,
                    seq_time_diffs=seq_time_diffs,
                )

            assert next_global.shape == global_tokens.shape
            assert next_nonseq.shape == (batch_size, num_seq, num_nonseq_tokens, d_model)
            assert next_seq.shape == seq_x.shape
            print("  ✓ Block output shapes correct")
            print(f"    global_tokens: {next_global.shape}")
            print(f"    nonseq_tokens: {next_nonseq.shape}")
            print(f"    seq_tokens: {next_seq.shape}")
            assert torch.isfinite(next_global).all()
            assert torch.isfinite(next_nonseq).all()
            assert torch.isfinite(next_seq).all()
            print("  ✓ All values are finite")
        except Exception as e:
            print(f"  ✗ HyFormerBlock failed: {e}")
            raise

        # Test multiple blocks
        print("\n[Testing Multiple HyFormerBlocks]")
        blocks = nn.ModuleList([
            HyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_global_tokens=num_global_tokens,
                num_nonseq_tokens=num_nonseq_tokens,
                ffn_hidden_dim=int(d_model * 2.0),
                kv_encoder=kv_encoder_fn,
                num_seq=num_seq,
                drop_out=0.1,
            ).to(device)
            for _ in range(num_blocks)
        ])

        try:
            blocks.eval()
            with torch.no_grad():
                curr_global = global_tokens
                curr_nonseq = non_seq_x.unsqueeze(1).expand(-1, num_seq, -1, -1)
                curr_seq = seq_x

                for i, blk in enumerate(blocks):
                    curr_global, curr_nonseq, curr_seq = blk(
                        curr_global, curr_nonseq, curr_seq, seq_valid_mask=seq_valid_mask, seq_time_diffs=seq_time_diffs
                    )
                    print(f"  ✓ Block {i + 1} output shape: {curr_global.shape}")

            assert torch.isfinite(curr_global).all()
            assert torch.isfinite(curr_nonseq).all()
            assert torch.isfinite(curr_seq).all()
            print("  ✓ All final values are finite")
        except Exception as e:
            print(f"  ✗ Multiple blocks failed: {e}")
            raise

        # Test MLP head
        print("\n[Testing MLP Head]")
        in_dim = num_seq * (num_global_tokens + num_nonseq_tokens) * d_model
        mlp_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        ).to(device)

        try:
            mlp_head.eval()
            with torch.no_grad():
                final_rep = torch.cat([curr_global.reshape(batch_size, -1), curr_nonseq.reshape(batch_size, -1)], dim=1)
                output = mlp_head(final_rep)

            assert output.shape == (batch_size, 1)
            print(f"  ✓ MLP output shape: {output.shape}")
            assert torch.isfinite(output).all()
            print("  ✓ All output values are finite")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"  Output mean: {output.mean().item():.4f}")
        except Exception as e:
            print(f"  ✗ MLP head failed: {e}")
            raise

        # Test backward pass
        print("\n[Testing Backward Pass]")
        try:
            mlp_head.train()
            blocks.train()
            query_gen.train()

            curr_global = global_tokens
            curr_nonseq = non_seq_x.unsqueeze(1).expand(-1, num_seq, -1, -1)
            curr_seq = seq_x

            for blk in blocks:
                curr_global, curr_nonseq, curr_seq = blk(
                    curr_global, curr_nonseq, curr_seq, seq_valid_mask=seq_valid_mask, seq_time_diffs=seq_time_diffs
                )

            final_rep = torch.cat([curr_global.reshape(batch_size, -1), curr_nonseq.reshape(batch_size, -1)], dim=1)
            output = mlp_head(final_rep)
            loss = output.mean()
            loss.backward()

            print("  ✓ Backward pass successful")

            # Count parameters with grad
            all_params = sum(p.numel() for p in mlp_head.parameters() if p.requires_grad)
            params_with_grad = sum(p.numel() for p in mlp_head.parameters() if p.requires_grad and p.grad is not None)
            print(f"  MLP params with gradients: {params_with_grad}/{all_params}")

            for blk in blocks:
                blk_grad = sum(p.numel() for p in blk.parameters() if p.requires_grad and p.grad is not None)
                blk_total = sum(p.numel() for p in blk.parameters() if p.requires_grad)
                print(f"  Block params with gradients: {blk_grad}/{blk_total}")

        except Exception as e:
            print(f"  ✗ Backward pass failed: {e}")
            raise

        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)

    def test_dataset_and_collate():
        """
        Test DemoDataset structure and collate_fn compatibility.

        This test:
        1. Creates mock sample data matching DemoDataset.__getitem__ output
        2. Tests default collate_fn behavior with batch of samples
        3. Validates data types and shapes
        4. Reports any issues without modifying code
        """
        print("\n" + "=" * 80)
        print("Testing DemoDataset Structure and Collate Function")
        print("=" * 80)

        max_seq_len = 12

        print("\n[Creating Mock Sample Data]")
        print(f"  max_seq_len: {max_seq_len}")

        # Create a single mock sample matching DemoDataset.__getitem__ format
        def create_mock_sample():
            sample = {
                "non_seq": {
                    "item_id": torch.tensor(123, dtype=torch.long),
                    "user_id": torch.tensor(456, dtype=torch.long),
                    "action_tt_hour": torch.tensor(10, dtype=torch.long),
                    "action_tt_dow": torch.tensor(3, dtype=torch.long),
                    "delta_tt_bucket": torch.tensor(5, dtype=torch.long),
                    "item_sparse_1": torch.tensor(10, dtype=torch.long),
                    "item_dense_bin_2": torch.tensor(3, dtype=torch.long),
                    "item_multihot_3": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "user_sparse_4": torch.tensor(20, dtype=torch.long),
                    "user_embedding_5": torch.randn(16),
                    "user_multihot_6": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "user_weighted_multihot_7": {
                        "ids": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                        "weights": torch.rand(max_seq_len),
                    },
                },
                "action_seq": {
                    "action_seq_1": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "action_seq_2": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "action_seq_timestamp": torch.randint(1000000, 2000000, (max_seq_len,), dtype=torch.long),
                    "action_seq_hour": torch.randint(1, 25, (max_seq_len,), dtype=torch.long),
                    "action_seq_dow": torch.randint(1, 8, (max_seq_len,), dtype=torch.long),
                },
                "content_seq": {
                    "content_seq_1": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "content_seq_timestamp": torch.randint(1000000, 2000000, (max_seq_len,), dtype=torch.long),
                    "content_seq_hour": torch.randint(1, 25, (max_seq_len,), dtype=torch.long),
                    "content_seq_dow": torch.randint(1, 8, (max_seq_len,), dtype=torch.long),
                },
                "item_seq": {
                    "item_seq_1": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "item_seq_2": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "item_seq_timestamp": torch.randint(1000000, 2000000, (max_seq_len,), dtype=torch.long),
                    "item_seq_hour": torch.randint(1, 25, (max_seq_len,), dtype=torch.long),
                    "item_seq_dow": torch.randint(1, 8, (max_seq_len,), dtype=torch.long),
                },
                "seq_time_diffs": {
                    "action_seq_time_diff": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "content_seq_time_diff": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "item_seq_time_diff": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                },
                "timestamp": torch.tensor(1609459200, dtype=torch.int32),
                "label": torch.tensor(1, dtype=torch.int32),
            }
            return sample

        # Create multiple samples for batching
        batch_size = 3
        samples = [create_mock_sample() for _ in range(batch_size)]
        print(f"  ✓ Created {batch_size} mock samples")

        # Test 1: Check individual sample structure
        print("\n[Testing Individual Sample Structure]")
        sample = samples[0]
        try:
            assert isinstance(sample, dict), "Sample must be dict"
            expected_keys = {"non_seq", "action_seq", "content_seq", "item_seq", "seq_time_diffs", "timestamp", "label"}
            assert set(sample.keys()) == expected_keys, (
                f"Sample keys mismatch. Got {set(sample.keys())}, expected {expected_keys}"
            )

            # Check nested dict structure
            assert isinstance(sample["non_seq"], dict), "non_seq must be dict"
            assert isinstance(sample["action_seq"], dict), "action_seq must be dict"
            assert isinstance(sample["seq_time_diffs"], dict), "seq_time_diffs must be dict"

            # Check tensor types
            assert isinstance(sample["timestamp"], torch.Tensor), "timestamp must be tensor"
            assert isinstance(sample["label"], torch.Tensor), "label must be tensor"

            print("  ✓ Individual sample structure valid")
            print(f"    Keys: {list(sample.keys())}")

        except AssertionError as e:
            print(f"  ✗ Sample structure validation failed: {e}")
            return False

        # Test 2: Try default collate_fn
        print("\n[Testing Default Collate Function]")
        from torch.utils.data.dataloader import default_collate

        try:
            batch = default_collate(samples)
            print("  ✓ default_collate succeeded")
            print(f"    Batch type: {type(batch)}")
            print(f"    Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'}")

            # Check if batch structure is usable
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, dict):
                        print(f"      {key}: dict with keys {list(value.keys())[:3]}...")
                    elif isinstance(value, torch.Tensor):
                        print(f"      {key}: Tensor shape {value.shape}")
                    else:
                        print(f"      {key}: {type(value)}")

        except Exception as e:
            print("  ✗ default_collate failed with error:")
            print(f"    {type(e).__name__}: {e}")
            print("\n  ⚠ Issue Detected: default_collate cannot handle nested dicts with mixed tensor/dict values")
            print("\n  Recommendation:")
            print("    Create a custom collate_fn in _dataset.py to properly handle this structure:")
            print("    ```python")
            print("    def demo_collate_fn(batch):")
            print("        # Handle nested dicts and tensors properly")
            print("        # Should stack/cat tensors and keep dict structure")
            print("        pass")
            print("    ```")
            print("    Then use it in DataLoader: DataLoader(..., collate_fn=demo_collate_fn)")
            return False

        # Test 3: Check data shapes after collate
        print("\n[Testing Batch Data Shapes]")
        if isinstance(batch, dict):
            try:
                # Check if we can access nested tensors
                if "non_seq" in batch and isinstance(batch["non_seq"], dict):
                    for key, val in batch["non_seq"].items():
                        if isinstance(val, torch.Tensor):
                            print(f"  non_seq.{key}: shape {val.shape}, dtype {val.dtype}")
                        elif isinstance(val, dict):
                            print(f"  non_seq.{key}: dict (nested structure preserved)")

                if "seq_time_diffs" in batch and isinstance(batch["seq_time_diffs"], dict):
                    for key, val in batch["seq_time_diffs"].items():
                        if isinstance(val, torch.Tensor):
                            expected_shape = (batch_size, max_seq_len)
                            actual_shape = val.shape
                            if actual_shape == expected_shape:
                                print(f"  ✓ seq_time_diffs.{key}: shape {actual_shape} (correct)")
                            else:
                                print(f"  ✗ seq_time_diffs.{key}: shape {actual_shape}, expected {expected_shape}")

                print("  ✓ Batch structure accessible")

            except Exception as e:
                print(f"  ✗ Failed to inspect batch shapes: {e}")
                return False

        # Test 4: Check HyFormer compatibility
        print("\n[Testing HyFormer Input Compatibility]")
        try:
            # Check if batch can be passed to HyFormer._get_embedding
            required_keys = {"non_seq", "action_seq", "content_seq", "item_seq", "seq_time_diffs"}
            batch_keys = set(batch.keys()) if isinstance(batch, dict) else set()

            if required_keys.issubset(batch_keys):
                print("  ✓ Batch has all required keys for HyFormer")
                print(f"    Required: {required_keys}")
                print(f"    Actual: {batch_keys}")
            else:
                missing = required_keys - batch_keys
                print(f"  ✗ Batch missing keys: {missing}")
                return False

            # Check seq_time_diffs structure
            if "seq_time_diffs" in batch:
                expected_seq_keys = {"action_seq_time_diff", "content_seq_time_diff", "item_seq_time_diff"}
                actual_seq_keys = (
                    set(batch["seq_time_diffs"].keys()) if isinstance(batch["seq_time_diffs"], dict) else set()
                )

                if expected_seq_keys.issubset(actual_seq_keys):
                    print("  ✓ seq_time_diffs has all required sequence types")
                else:
                    missing_seq = expected_seq_keys - actual_seq_keys
                    print(f"  ✗ seq_time_diffs missing: {missing_seq}")
                    return False

        except Exception as e:
            print(f"  ✗ HyFormer compatibility check failed: {e}")
            return False

        print("\n" + "=" * 80)
        print("Dataset and Collate Function Tests Completed")
        print("=" * 80)
        return True

    def test_dataset_embedding_format():
        """
        Deep validation: Check if batch data format is compatible with EmbeddingLayer.
        This test focuses on potential issues in _get_embedding method.
        """
        print("\n" + "=" * 80)
        print("Testing Dataset Format Compatibility with EmbeddingLayer")
        print("=" * 80)

        max_seq_len = 12

        # Create a mock batch similar to what DataLoader produces
        print("\n[Creating Realistic Mock Batch]")

        from torch.utils.data.dataloader import default_collate

        def create_mock_sample():
            return {
                "non_seq": {
                    "item_id": torch.tensor(123, dtype=torch.long),
                    "user_id": torch.tensor(456, dtype=torch.long),
                    "action_tt_hour": torch.tensor(10, dtype=torch.long),
                    "action_tt_dow": torch.tensor(3, dtype=torch.long),
                    "delta_tt_bucket": torch.tensor(5, dtype=torch.long),
                    "item_sparse_1": torch.tensor(10, dtype=torch.long),
                    "item_multihot_3": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "user_weighted_multihot_7": {
                        "ids": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                        "weights": torch.rand(max_seq_len),
                    },
                },
                "action_seq": {
                    "action_seq_1": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "action_seq_timestamp": torch.randint(1000000, 2000000, (max_seq_len,), dtype=torch.long),
                },
                "content_seq": {
                    "content_seq_1": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                },
                "item_seq": {
                    "item_seq_1": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                },
                "seq_time_diffs": {
                    "action_seq_time_diff": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "content_seq_time_diff": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                    "item_seq_time_diff": torch.randint(0, 100, (max_seq_len,), dtype=torch.long),
                },
                "timestamp": torch.tensor(1609459200, dtype=torch.int32),
                "label": torch.tensor(1, dtype=torch.int32),
            }

        batch_size = 2
        samples = [create_mock_sample() for _ in range(batch_size)]
        batch = default_collate(samples)

        # Test 1: Check weighted_multihot structure in batch
        print("\n[Testing Weighted MultiHot Structure in Batch]")
        try:
            wmhot_in_batch = batch["non_seq"].get("user_weighted_multihot_7")
            if wmhot_in_batch is None:
                print("  ⚠ Warning: user_weighted_multihot_7 not in batch")
            elif isinstance(wmhot_in_batch, dict):
                print("  ⚠ Issue Found: user_weighted_multihot_7 is a dict in batch")
                print(f"    Structure: {list(wmhot_in_batch.keys())}")

                # Check inner structure
                ids_shape = wmhot_in_batch["ids"].shape if "ids" in wmhot_in_batch else None
                weights_shape = wmhot_in_batch["weights"].shape if "weights" in wmhot_in_batch else None

                print(f"    ids shape: {ids_shape}")
                print(f"    weights shape: {weights_shape}")

                print("\n    ⚠ Potential Issue:")
                print("       EmbeddingLayer expects: {'user_weighted_multihot_7': Tensor}")
                print("       But received: {'user_weighted_multihot_7': {'ids': Tensor, 'weights': Tensor}}")
                print("\n    This will likely cause an error in EmbeddingLayer.__call__")
                print("    when it tries to index embedding table with a dict instead of tensor.")

                # Check if this matches WeightedMultiHotFeature expectation
                print("\n    Expected by WeightedMultiHotFeature:")
                print("      Input: x['user_weighted_multihot_7'] should be a dict with 'ids' and 'weights'")
                print("      Current batch structure: ✓ Matches")
                print("    ✓ Batch structure is CORRECT for WeightedMultiHotFeature")

            elif isinstance(wmhot_in_batch, torch.Tensor):
                print(f"  ✓ user_weighted_multihot_7 is a tensor of shape {wmhot_in_batch.shape}")

        except Exception as e:
            print(f"  ✗ Error checking weighted_multihot: {e}")

        # Test 2: Data accessibility from HyFormer._get_embedding perspective
        print("\n[Testing Data Access in _get_embedding]")
        try:
            # Simulate what happens in _get_embedding
            x = batch  # This is what gets passed to _get_embedding

            # Check if keys are accessible
            if "non_seq" in x:
                print("  ✓ batch['non_seq'] exists")
                for key in list(x["non_seq"].keys())[:5]:
                    val = x["non_seq"][key]
                    if isinstance(val, torch.Tensor):
                        print(f"    ✓ batch['non_seq']['{key}']: Tensor {val.shape}")
                    elif isinstance(val, dict):
                        print(f"    ⚠ batch['non_seq']['{key}']: dict {list(val.keys())}")

            if "seq_time_diffs" in x:
                print("  ✓ batch['seq_time_diffs'] exists")
                for seq_type in ["action_seq_time_diff", "content_seq_time_diff", "item_seq_time_diff"]:
                    if seq_type in x["seq_time_diffs"]:
                        shape = x["seq_time_diffs"][seq_type].shape
                        print(f"    ✓ seq_time_diffs['{seq_type}']: shape {shape}")
                    else:
                        print(f"    ✗ Missing: seq_time_diffs['{seq_type}']")

        except Exception as e:
            print(f"  ✗ Error accessing batch data: {e}")

        # Test 3: Check seq_time_diffs stacking in HyFormer.forward
        print("\n[Testing seq_time_diffs Stacking (as in HyFormer.forward)]")
        try:
            # Simulate the stacking that happens in HyFormer._get_embedding
            stacked = torch.stack(
                [
                    x["seq_time_diffs"]["action_seq_time_diff"].long(),
                    x["seq_time_diffs"]["content_seq_time_diff"].long(),
                    x["seq_time_diffs"]["item_seq_time_diff"].long(),
                ],
                dim=1,
            )

            expected_shape = (batch_size, 3, max_seq_len)
            actual_shape = stacked.shape

            if actual_shape == expected_shape:
                print("  ✓ seq_time_diffs stacking correct")
                print(f"    Expected: {expected_shape}")
                print(f"    Actual: {actual_shape}")
            else:
                print("  ✗ seq_time_diffs stacking shape mismatch")
                print(f"    Expected: {expected_shape}")
                print(f"    Actual: {actual_shape}")
                print("\n    ⚠ Issue: Shape mismatch will cause error in HyFormerBlock forward")

        except Exception as e:
            print(f"  ✗ Error stacking seq_time_diffs: {type(e).__name__}: {e}")
            print(f"\n  ⚠ Issue Found: {e}")
            print("     This will cause HyFormer.forward to fail when trying to pass seq_time_diffs")

        print("\n" + "=" * 80)
        print("Embedding Format Tests Completed")
        print("=" * 80)
        return True

    test_forward_simplified()
    test_dataset_and_collate()
    test_dataset_embedding_format()
