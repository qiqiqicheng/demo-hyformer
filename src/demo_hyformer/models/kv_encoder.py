import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class RelPosBias(nn.Module):
    def __init__(self, n_heads: int, max_seq_len: int, num_buckets: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.rel_pos_bias_table = nn.Parameter(torch.empty(num_buckets, n_heads))
        nn.init.normal_(self.rel_pos_bias_table, mean=0.0, std=0.02)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = max(self.max_seq_len, 2)

        n = torch.abs(relative_position)
        max_exact = max(1, num_buckets // 2)
        is_small = n < max_exact

        n_float = n.float().clamp(min=1.0)
        log_scale = math.log(max_distance / max_exact) if max_distance > max_exact else 1.0
        log_scale = max(log_scale, 1e-6)

        val_if_large = max_exact + (torch.log(n_float / max_exact) / log_scale * (num_buckets - max_exact)).long()
        val_if_large = torch.clamp(val_if_large, min=max_exact, max=num_buckets - 1)

        return torch.where(is_small, n.long(), val_if_large)

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.long, device=self.rel_pos_bias_table.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self._relative_position_bucket(relative_positions)
        bias = self.rel_pos_bias_table[buckets]
        return bias.permute(2, 0, 1).unsqueeze(0)


class RoPEPositionEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) as drop-in alternative to RelPosBias.

    Applies rotary embeddings directly to Q and K tensors.
    Compatible with HSTUKVLayer via the ``rope`` argument.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to Q and K.

        Args:
            q: [B, S, H, T, head_dim]
            k: [B, S, H, T, head_dim]

        Returns:
            q_rot, k_rot: same shapes as inputs
        """
        T = q.shape[-2]
        t = torch.arange(T, device=q.device, dtype=self.inv_freq.dtype)  # type: ignore
        freqs = torch.outer(t, self.inv_freq)  # type:ignore [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1).to(dtype=q.dtype)  # [T, head_dim]
        cos = emb.cos().view(1, 1, 1, T, -1)
        sin = emb.sin().view(1, 1, 1, T, -1)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class HSTUKVLayer(nn.Module):
    """https://arxiv.org/abs/2402.17152 Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_seq: int,
        ffn_hidden_dim: int,
        drop_out: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_seq = num_seq
        self.ffn_hidden_dim = ffn_hidden_dim
        self.drop_out = drop_out

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={self.d_model}, num_heads={self.num_heads}"
            )

        S = self.num_seq
        D = self.d_model
        H = self.ffn_hidden_dim

        self.qkv_w = nn.Parameter(torch.empty(S, D, 3 * D))
        self.qkv_b = nn.Parameter(torch.empty(S, 3 * D))
        self.o_w = nn.Parameter(torch.empty(S, D, D))
        self.o_b = nn.Parameter(torch.empty(S, D))

        self.ffn_w1 = nn.Parameter(torch.empty(S, D, H))
        self.ffn_b1 = nn.Parameter(torch.empty(S, H))
        self.ffn_w2 = nn.Parameter(torch.empty(S, H, D))
        self.ffn_b2 = nn.Parameter(torch.empty(S, D))

        self.attn_in_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)
        self.attn_out_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)
        self.ffn_in_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)
        self.ffn_out_norm = SequenceWiseRMSNorm(num_seq=S, d_model=D)

        self.attn_dropout = nn.Dropout(p=self.drop_out)
        self.ffn_hidden_dropout = nn.Dropout(p=self.drop_out)
        self.ffn_out_dropout = nn.Dropout(p=self.drop_out)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_w)
        nn.init.zeros_(self.qkv_b)

        nn.init.zeros_(self.o_w)
        nn.init.zeros_(self.o_b)

        nn.init.kaiming_uniform_(self.ffn_w1, a=math.sqrt(5), nonlinearity="relu")
        nn.init.zeros_(self.ffn_b1)

        nn.init.zeros_(self.ffn_w2)
        nn.init.zeros_(self.ffn_b2)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: torch.Tensor | None = None,
        time_attn_bias: torch.Tensor | None = None,
        rope: "RoPEPositionEncoding | None" = None,
    ) -> torch.Tensor:
        B, S, T, D = x.shape
        head_dim = D // self.num_heads

        attn_input = self.attn_in_norm(x)
        qkv = torch.einsum("bstd,sdf->bstf", attn_input, self.qkv_w) + self.qkv_b.unsqueeze(0).unsqueeze(2)
        q, k, v = torch.split(qkv, D, dim=-1)

        q = q.view(B, S, T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)
        k = k.view(B, S, T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)
        v = v.view(B, S, T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)

        if rope is not None:
            q, k = rope(q, k)

        attn_bias = None
        if rel_pos_bias is not None:
            attn_bias = rel_pos_bias.unsqueeze(1)
            attn_bias = attn_bias.expand(B, S, -1, -1, -1)
        if time_attn_bias is not None:
            attn_bias = time_attn_bias if attn_bias is None else (attn_bias + time_attn_bias)
        if attn_bias is not None:
            attn_bias = attn_bias.to(dtype=q.dtype)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.drop_out if self.training else 0.0,
        )
        attn_out = attn_out.permute(0, 1, 3, 2, 4).contiguous().view(B, S, T, D)

        attn_out = torch.einsum("bstd,sde->bste", attn_out, self.o_w) + self.o_b.unsqueeze(0).unsqueeze(2)
        attn_out = self.attn_dropout(attn_out)
        x = self.attn_out_norm(x + attn_out)

        ffn_input = self.ffn_in_norm(x)
        hidden = torch.einsum("bstd,sdh->bsth", ffn_input, self.ffn_w1) + self.ffn_b1.unsqueeze(0).unsqueeze(2)
        hidden = F.silu(hidden)
        hidden = self.ffn_hidden_dropout(hidden)

        ffn_out = torch.einsum("bsth,shd->bstd", hidden, self.ffn_w2) + self.ffn_b2.unsqueeze(0).unsqueeze(2)
        ffn_out = self.ffn_out_dropout(ffn_out)

        return self.ffn_out_norm(x + ffn_out)


class HSTUSeqKVEncoder(nn.Module):
    """Lightweight HSTU-style KV encoder for HyFormer.

    Input:  seq_tokens [B, S, T, D]
    Output: encoded tokens [B, S, T, D]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_seq: int = 3,
        num_layers: int = 2,
        ffn_hidden_ratio: float = 2.0,
        drop_out: float = 0.1,
        max_seq_len: int = 256,
        pos_bias_type: str = "rel_pos_bias",  # or rope
        rel_pos_buckets: int = 32,
        use_time_embedding: bool = True,
        num_time_buckets: int = 32,
        time_bucket_fn: str = "log",
        *args,
        **kwargs,
    ):
        super().__init__()
        log.info(f"Initializing HSTUSeqKVEncoder with args {args} and kwargs {kwargs}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_seq = num_seq
        self.num_layers = num_layers
        self.drop_out = drop_out
        self.max_seq_len = max_seq_len

        if pos_bias_type not in ("rel_pos_bias", "rope", "none"):
            raise ValueError(f"pos_bias_type must be 'rel_pos_bias', 'rope', or 'none', got '{pos_bias_type}'")
        self.pos_bias_type = pos_bias_type
        self.use_time_embedding = use_time_embedding
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn

        hidden_dim = int(self.d_model * ffn_hidden_ratio)

        self.input_norm = SequenceWiseRMSNorm(num_seq=self.num_seq, d_model=self.d_model)
        self.input_dropout = nn.Dropout(p=self.drop_out)

        self.layers = nn.ModuleList([
            HSTUKVLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_seq=self.num_seq,
                ffn_hidden_dim=hidden_dim,
                drop_out=self.drop_out,
            )
            for _ in range(self.num_layers)
        ])

        if self.pos_bias_type == "rel_pos_bias":
            self.pos_encoder = RelPosBias(
                n_heads=self.num_heads,
                max_seq_len=self.max_seq_len,
                num_buckets=rel_pos_buckets,
            )
        elif self.pos_bias_type == "rope":
            head_dim = self.d_model // self.num_heads
            self.pos_encoder = RoPEPositionEncoding(
                head_dim=head_dim,
                max_seq_len=self.max_seq_len,
            )

        if self.use_time_embedding:
            self.time_emb = nn.Embedding(self.num_time_buckets + 1, self.d_model, padding_idx=0)
            self.time_attn_bias = nn.Embedding(self.num_time_buckets + 1, self.num_heads, padding_idx=0)
            nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.time_attn_bias.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.time_emb.weight[0].zero_()
                self.time_attn_bias.weight[0].zero_()

    def _time_to_bucket(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x.float(), min=0.0)
        x_minutes = x / 60.0

        if self.time_bucket_fn == "sqrt":
            b = torch.sqrt(x_minutes)
        elif self.time_bucket_fn == "log":
            b = torch.log1p(x_minutes)
        else:
            raise ValueError(f"Unsupported time_bucket_fn: {self.time_bucket_fn}")

        b = b.long() + 1
        b = torch.where(x > 0, b, torch.zeros_like(b))
        return torch.clamp(b, min=0, max=self.num_time_buckets)

    def _prepare_time_features(
        self,
        time_diffs: torch.Tensor | None,
        batch_size: int,
        seq_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Args:
            time_diffs (torch.Tensor | None): [B, S, T] or None
            batch_size (int): _description_
            seq_size (int): _description_
            seq_len (int): _description_
            device (torch.device): _description_
            dtype (torch.dtype): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor] | tuple[None, None]: [B, S, T, D], [B, S, H, T, T]
        """
        if not self.use_time_embedding:
            return None, None

        if time_diffs is None:
            delta = torch.zeros(batch_size, seq_size, seq_len, device=device, dtype=torch.float32)
        else:
            delta = time_diffs.to(device=device, dtype=torch.float32)

        token_bucket = self._time_to_bucket(delta)  # [B, S, T]
        token_time_emb = self.time_emb(token_bucket).to(dtype=dtype)  # [B, S, T, D]

        pseudo_ts = torch.cumsum(delta, dim=-1)  # [B, S, T]
        pair_delta = torch.abs(pseudo_ts.unsqueeze(-1) - pseudo_ts.unsqueeze(-2))  # [B, S, T, T]
        pair_bucket = self._time_to_bucket(pair_delta)  # [B, S, T, T]
        pair_time_bias = (
            self.time_attn_bias(pair_bucket)  # [B, S, T, T, H]
            .permute(0, 1, 4, 2, 3)  # [B, S, H, T, T]
            .to(dtype=dtype)
        )  # [B, S, H, T, T]

        return token_time_emb, pair_time_bias

    def forward(self, seq_tokens: torch.Tensor, time_diffs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            seq_tokens (torch.Tensor): [B, S, T, D]
            time_diffs (torch.Tensor | None, optional): [B, S, T]. Defaults to None.

        Returns:
            torch.Tensor: [B, S, T, D]
        """
        if seq_tokens.dim() != 4:
            raise ValueError(f"seq_tokens must be [B,S,T,D], got shape={tuple(seq_tokens.shape)}")

        B, S, T, D = seq_tokens.shape

        x = self.input_norm(seq_tokens)

        rel_pos_bias = None
        rope = None
        if self.pos_bias_type == "rel_pos_bias":
            rel_pos_bias = self.pos_encoder(T).to(device=seq_tokens.device, dtype=seq_tokens.dtype)
        elif self.pos_bias_type == "rope":
            rope = self.pos_encoder

        time_token_emb, time_attn_bias = self._prepare_time_features(  # [B, S, T, D], [B, S, H, T, T]
            time_diffs=time_diffs,
            batch_size=B,
            seq_size=S,
            seq_len=T,
            device=seq_tokens.device,
            dtype=seq_tokens.dtype,
        )
        if time_token_emb is not None:
            x = x + time_token_emb

        x = self.input_dropout(x)

        for layer in self.layers:
            x = layer(x, rel_pos_bias=rel_pos_bias, time_attn_bias=time_attn_bias, rope=rope)

        return x


if __name__ == "__main__":
    """AI generated test..."""
    torch.manual_seed(728)

    B, S, T, D = 4, 3, 12, 32
    H = 4

    seq_tokens = torch.randn(B, S, T, D)
    time_diffs = torch.randint(low=0, high=7 * 24 * 3600, size=(B, S, T), dtype=torch.long)
    time_diffs[:, :, 0] = 0

    cases = [
        {
            "name": "rel_pos_bias + time",
            "cfg": {
                "d_model": D,
                "num_heads": H,
                "num_seq": S,
                "num_layers": 2,
                "ffn_hidden_ratio": 2.0,
                "drop_out": 0.1,
                "max_seq_len": 256,
                "pos_bias_type": "rel_pos_bias",
                "rel_pos_buckets": 32,
                "use_time_embedding": True,
                "num_time_buckets": 256,
                "time_bucket_fn": "log",
            },
            "use_time": True,
            "train_mode": True,
        },
        {
            "name": "rope + time",
            "cfg": {
                "d_model": D,
                "num_heads": H,
                "num_seq": S,
                "num_layers": 2,
                "ffn_hidden_ratio": 2.0,
                "drop_out": 0.1,
                "max_seq_len": 256,
                "pos_bias_type": "rope",
                "rel_pos_buckets": 32,
                "use_time_embedding": True,
                "num_time_buckets": 256,
                "time_bucket_fn": "sqrt",
            },
            "use_time": True,
            "train_mode": True,
        },
        {
            "name": "none + no_time",
            "cfg": {
                "d_model": D,
                "num_heads": H,
                "num_seq": S,
                "num_layers": 2,
                "ffn_hidden_ratio": 2.0,
                "drop_out": 0.1,
                "max_seq_len": 256,
                "pos_bias_type": "none",
                "rel_pos_buckets": 32,
                "use_time_embedding": False,
                "num_time_buckets": 256,
                "time_bucket_fn": "log",
            },
            "use_time": False,
            "train_mode": False,
        },
    ]

    print("Running HSTUSeqKVEncoder forward tests...")
    for case in cases:
        encoder = HSTUSeqKVEncoder(**case["cfg"])
        if case["train_mode"]:
            encoder.train()
        else:
            encoder.eval()

        x = seq_tokens.clone().requires_grad_(True)
        td = time_diffs if case["use_time"] else None

        out = encoder(x, time_diffs=td)
        assert out.shape == (B, S, T, D), f"Unexpected output shape in case={case['name']}: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"Non-finite outputs found in case={case['name']}"

        # Backward pass sanity check to cover parameterized paths.
        loss = out.pow(2).mean()
        loss.backward()
        grad_finite = all((p.grad is None) or torch.isfinite(p.grad).all() for p in encoder.parameters())
        assert grad_finite, f"Non-finite gradients found in case={case['name']}"

        print(
            f"[PASS] {case['name']}: out_shape={tuple(out.shape)}, "
            f"mean={out.mean().item():.6f}, std={out.std(unbiased=False).item():.6f}"
        )

    print("All forward tests passed.")
