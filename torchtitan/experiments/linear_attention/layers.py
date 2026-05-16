# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    GQAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import (
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)
from torchtitan.protocols.module import Module


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}


def _init_raven_a_log(param: torch.Tensor) -> None:
    a = torch.empty_like(param, dtype=torch.float32).uniform_(0, 16)
    param.copy_(torch.log(a).to(param.dtype))


def _init_raven_dt_bias(param: torch.Tensor) -> None:
    dt_min = 0.001
    dt_max = 0.1
    dt_init_floor = 1e-4
    dt = torch.exp(
        torch.rand_like(param, dtype=torch.float32)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    param.copy_(inv_dt.to(param.dtype))


_GDN_STATE_INIT = {
    "A_log": nn.init.zeros_,
    "dt_bias": nn.init.zeros_,
}
_KDA_STATE_INIT = {
    "A_log": nn.init.zeros_,
    "dt_bias": nn.init.zeros_,
}
_MAMBA3_STATE_INIT = {
    "dt_bias": nn.init.zeros_,
    "B_bias": nn.init.ones_,
    "C_bias": nn.init.ones_,
    "D": nn.init.ones_,
}
_RAVEN_STATE_INIT = {
    "A_log": _init_raven_a_log,
    "dt_bias": _init_raven_dt_bias,
}
_SINK_GQA_INIT = {
    "sinks": partial(nn.init.trunc_normal_, std=0.02),
}


def _load_chunk_gated_delta_rule():
    try:
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
    except ImportError as e:
        raise ImportError(
            "GatedDeltaAttention requires the optional flash-linear-attention "
            "dependency. Install it with `uv sync --extra dev "
            "--extra linear-attention`."
        ) from e
    return chunk_gated_delta_rule


def _load_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as e:
        raise ImportError(
            "KimiDeltaAttention requires the optional flash-linear-attention "
            "dependency. Install it with `uv sync --extra dev "
            "--extra linear-attention`."
        ) from e
    return chunk_kda


def _load_chunk_raven():
    try:
        from fla.ops.gsa import chunk_gsa
    except ImportError as e:
        raise ImportError(
            "RavenAttention requires the optional flash-linear-attention "
            "dependency. Install it with `uv sync --extra dev "
            "--extra linear-attention`."
        ) from e
    return chunk_gsa


def _load_mamba3_siso_combined():
    try:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
            mamba3_siso_combined,
        )
    except ImportError as e:
        raise ImportError(
            "Mamba3Attention requires the optional mamba-ssm source build with "
            "Mamba3 support. Install it with "
            "`MAMBA_FORCE_BUILD=TRUE pip install "
            "git+https://github.com/state-spaces/mamba.git --no-build-isolation`."
        ) from e
    return mamba3_siso_combined


class GatedDeltaAttention(Module):
    """TorchTitan-owned wrapper around FLA's training Gated Delta kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int = 3
        head_dim: int = 64
        num_v_heads: int = 3
        head_v_dim: int = 128
        q_proj: Linear.Config | None = None
        k_proj: Linear.Config | None = None
        v_proj: Linear.Config | None = None
        a_proj: Linear.Config | None = None
        b_proj: Linear.Config | None = None
        g_proj: Linear.Config | None = None
        o_proj: Linear.Config | None = None
        o_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self._param_init = _GDN_STATE_INIT

        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_v_heads = config.num_v_heads
        self.head_v_dim = config.head_v_dim
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        if self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                "num_v_heads must be divisible by num_heads for FLA "
                "chunk_gated_delta_rule."
            )

        self.q_proj = (
            config.q_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.key_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.k_proj = (
            config.k_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.key_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.v_proj = (
            config.v_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.value_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.a_proj = (
            config.a_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_v_heads,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.b_proj = (
            config.b_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_v_heads,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.g_proj = (
            config.g_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.value_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.o_norm = (
            config.o_norm
            or RMSNorm.Config(
                normalized_shape=self.head_v_dim,
                eps=1e-6,
                param_init=_NORM_INIT,
            )
        ).build()
        self.o_proj = (
            config.o_proj
            or Linear.Config(
                in_features=self.value_dim,
                out_features=self.dim,
                param_init=_LINEAR_INIT,
            )
        ).build()

        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "GatedDeltaAttention does not support attention masks or packed "
                "document inputs in this experiment."
            )

        chunk_gated_delta_rule = _load_chunk_gated_delta_rule()

        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        g = self.a_proj(x)
        beta = self.b_proj(x).sigmoid()

        out, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        gate = self.g_proj(x).view(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        out = self.o_norm(out) * F.silu(gate)
        return self.o_proj(out.reshape(batch_size, seq_len, self.value_dim))


class KimiDeltaAttention(Module):
    """TorchTitan-owned wrapper around FLA's training Kimi Delta Attention kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int = 3
        head_dim: int = 64
        num_v_heads: int = 3
        head_v_dim: int = 128
        safe_gate: bool = False
        lower_bound: float | None = None
        q_proj: Linear.Config | None = None
        k_proj: Linear.Config | None = None
        v_proj: Linear.Config | None = None
        f1_proj: Linear.Config | None = None
        f2_proj: Linear.Config | None = None
        b_proj: Linear.Config | None = None
        g1_proj: Linear.Config | None = None
        g2_proj: Linear.Config | None = None
        o_proj: Linear.Config | None = None
        o_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self._param_init = _KDA_STATE_INIT

        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_v_heads = config.num_v_heads
        self.head_v_dim = config.head_v_dim
        self.safe_gate = config.safe_gate
        self.lower_bound = config.lower_bound
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gate_dim = self.num_v_heads * self.head_dim

        if self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                "num_v_heads must be divisible by num_heads for FLA chunk_kda."
            )
        if self.head_dim > 256:
            raise ValueError("head_dim must be <= 256 for FLA chunk_kda.")
        if self.safe_gate and self.lower_bound is None:
            raise ValueError("lower_bound must be set when safe_gate=True.")

        self.q_proj = (
            config.q_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.key_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.k_proj = (
            config.k_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.key_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.v_proj = (
            config.v_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.value_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.f1_proj = (
            config.f1_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.head_v_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.f2_proj = (
            config.f2_proj
            or Linear.Config(
                in_features=self.head_v_dim,
                out_features=self.gate_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.b_proj = (
            config.b_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_v_heads,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.g1_proj = (
            config.g1_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.head_v_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.g2_proj = (
            config.g2_proj
            or Linear.Config(
                in_features=self.head_v_dim,
                out_features=self.value_dim,
                bias=True,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.o_norm = (
            config.o_norm
            or RMSNorm.Config(
                normalized_shape=self.head_v_dim,
                eps=1e-6,
                param_init=_NORM_INIT,
            )
        ).build()
        self.o_proj = (
            config.o_proj
            or Linear.Config(
                in_features=self.value_dim,
                out_features=self.dim,
                param_init=_LINEAR_INIT,
            )
        ).build()

        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.gate_dim))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "KimiDeltaAttention does not support attention masks or packed "
                "document inputs in this experiment."
            )

        chunk_kda = _load_chunk_kda()

        batch_size, seq_len, _ = x.shape
        q = F.silu(self.q_proj(x)).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = F.silu(self.k_proj(x)).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = F.silu(self.v_proj(x)).view(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        g = self.f2_proj(self.f1_proj(x)).view(
            batch_size, seq_len, self.num_v_heads, self.head_dim
        )
        beta = self.b_proj(x).sigmoid()

        out, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
        )

        gate = self.g2_proj(self.g1_proj(x)).view(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        out = self.o_norm(out) * torch.sigmoid(gate)
        return self.o_proj(out.reshape(batch_size, seq_len, self.value_dim))


class RavenAttention(Module):
    """TorchTitan-owned Raven wrapper using FLA's GSA training kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int = 4
        num_kv_heads: int | None = None
        num_slots: int | None = None
        topk: int = 8
        feature_map: str = "swish"
        router_score: str = "sigmoid"
        add_gumbel_noise: bool = True
        scale: float = 1.0
        q_proj: Linear.Config | None = None
        k_proj: Linear.Config | None = None
        v_proj: Linear.Config | None = None
        r_proj: Linear.Config | None = None
        a_proj: Linear.Config | None = None
        o_proj: Linear.Config | None = None
        q_norm: RMSNorm.Config | None = None
        k_norm: RMSNorm.Config | None = None
        g_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self._param_init = _RAVEN_STATE_INIT

        self.dim = config.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = (
            config.num_heads if config.num_kv_heads is None else config.num_kv_heads
        )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.dim // self.num_heads
        self.value_dim = self.dim
        self.head_v_dim = self.value_dim // self.num_heads
        self.num_slots = self.head_dim if config.num_slots is None else config.num_slots
        self.topk = config.topk
        self.feature_map = config.feature_map
        self.router_score = config.router_score
        self.add_gumbel_noise = config.add_gumbel_noise
        self.scale = config.scale

        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads for RavenAttention.")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads for RavenAttention."
            )
        if self.num_slots != self.head_dim:
            raise ValueError(
                "RavenAttention v1 requires num_slots to match per-head q/k dim."
            )
        if self.topk > self.num_slots:
            raise ValueError("topk must be <= num_slots for RavenAttention.")
        if self.feature_map not in ("swish", "relu"):
            raise ValueError("feature_map must be 'swish' or 'relu'.")
        if self.router_score not in ("sigmoid", "softmax"):
            raise ValueError("router_score must be 'sigmoid' or 'softmax'.")

        self.q_proj = (
            config.q_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_heads * self.head_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.k_proj = (
            config.k_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_kv_heads * self.head_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.v_proj = (
            config.v_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_kv_heads * self.head_v_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.r_proj = (
            config.r_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_heads * self.num_slots,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.a_proj = (
            config.a_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.num_heads,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.q_norm = (
            config.q_norm
            or RMSNorm.Config(
                normalized_shape=self.head_dim,
                eps=1e-6,
                param_init=_NORM_INIT,
            )
        ).build()
        self.k_norm = (
            config.k_norm
            or RMSNorm.Config(
                normalized_shape=self.head_dim,
                eps=1e-6,
                param_init=_NORM_INIT,
            )
        ).build()
        self.g_norm = (
            config.g_norm
            or RMSNorm.Config(
                normalized_shape=self.value_dim,
                eps=1e-6,
                param_init=_NORM_INIT,
            )
        ).build()
        self.o_proj = (
            config.o_proj
            or Linear.Config(
                in_features=self.value_dim,
                out_features=self.dim,
                param_init=_LINEAR_INIT,
            )
        ).build()

        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "swish":
            return F.silu(x)
        if self.feature_map == "relu":
            return F.relu(x)
        raise AssertionError("unreachable")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "RavenAttention does not support attention masks or packed "
                "document inputs in this experiment."
            )

        chunk_raven = _load_chunk_raven()

        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_v_dim)
        router = self.r_proj(x).view(
            batch_size, seq_len, self.num_heads, self.num_slots
        )

        f = (
            -self.A_log.float().exp()
            * F.softplus(self.a_proj(x).float() + self.dt_bias)
        ).unsqueeze(-1)

        q = self.q_norm(self._feature_map(q))
        k = self.k_norm(self._feature_map(k))
        v = F.silu(v)

        if self.add_gumbel_noise and self.training:
            router = router - torch.empty_like(router).exponential_().log()

        if self.router_score == "sigmoid":
            orig_scores = torch.sigmoid(router)
        else:
            orig_scores = torch.softmax(router, dim=-1)

        route_idx = orig_scores.topk(self.topk, dim=-1).indices
        topk_weights = torch.gather(orig_scores, dim=-1, index=route_idx)
        if self.router_score == "sigmoid":
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-9
            )

        sparse_routes = torch.zeros_like(router).scatter_(
            -1, route_idx, topk_weights.to(router.dtype)
        )
        f = (f * sparse_routes).to(q.dtype)
        s = (1 - f.exp()).to(q.dtype)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)
            f = f.repeat_interleave(self.num_kv_groups, dim=2)
            s = s.repeat_interleave(self.num_kv_groups, dim=2)

        out, _ = chunk_raven(
            q=q,
            k=k,
            v=v,
            s=s,
            g=f,
            initial_state=None,
            output_final_state=False,
            scale=self.scale,
            cu_seqlens=None,
        )

        out = out.reshape(batch_size, seq_len, self.value_dim)
        out = self.g_norm(F.silu(out))
        return self.o_proj(out)


class SinkGQAttention(GQAttention):
    """GQA with GPT-OSS attention sink rescaling for sliding-window layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        pass

    def __init__(self, config: Config):
        if not isinstance(config.inner_attention, FlexAttention.Config):
            raise ValueError("SinkGQAttention requires FlexAttention.")
        super().__init__(config)
        self._param_init = _SINK_GQA_INIT
        self.sinks = nn.Parameter(torch.empty(config.n_heads))

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.qkv_linear(x)

        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            if self.rope_backend == "cos_sin":
                xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        if isinstance(attention_masks, dict):
            mask_key = "rope" if self.use_rope else "nope"
            attention_masks = attention_masks[mask_key]

        output, lse = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            return_lse=True,
        )
        sink_scale = torch.sigmoid(lse - self.sinks.view(1, 1, -1)).unsqueeze(-1)
        output = output * sink_scale.to(output.dtype)
        return self.wo(output.contiguous().view(bs, seqlen, -1))


class Mamba3Attention(Module):
    """TorchTitan-owned wrapper around the official Mamba-3 SISO training kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        d_state: int = 128
        expand: int = 2
        head_dim: int = 64
        num_bc_heads: int = 1
        rope_fraction: float = 0.5
        chunk_size: int = 64
        a_floor: float = 1e-4
        in_proj: Linear.Config | None = None
        out_proj: Linear.Config | None = None
        b_norm: RMSNorm.Config | None = None
        c_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self._param_init = _MAMBA3_STATE_INIT

        self.dim = config.dim
        self.d_state = config.d_state
        self.expand = config.expand
        self.head_dim = config.head_dim
        self.num_bc_heads = config.num_bc_heads
        self.rope_fraction = config.rope_fraction
        self.chunk_size = config.chunk_size
        self.a_floor = config.a_floor

        if self.rope_fraction not in (0.5, 1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0 for Mamba3.")

        self.d_inner = self.expand * self.dim
        if self.d_inner % self.head_dim != 0:
            raise ValueError(
                f"expand * dim ({self.d_inner}) must be divisible by "
                f"head_dim ({self.head_dim})."
            )
        self.num_heads = self.d_inner // self.head_dim

        split_tensor_size = int(self.d_state * self.rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError("Mamba3 requires at least one RoPE angle.")
        self.rotary_dim_divisor = int(2 / self.rope_fraction)

        self.in_proj_dim = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads
            + 3 * self.num_heads
            + self.num_rope_angles
        )

        self.in_proj = (
            config.in_proj
            or Linear.Config(
                in_features=self.dim,
                out_features=self.in_proj_dim,
                param_init=_LINEAR_INIT,
            )
        ).build()
        self.b_norm = (
            config.b_norm
            or RMSNorm.Config(
                normalized_shape=self.d_state,
                eps=1e-5,
                param_init=_NORM_INIT,
            )
        ).build()
        self.c_norm = (
            config.c_norm
            or RMSNorm.Config(
                normalized_shape=self.d_state,
                eps=1e-5,
                param_init=_NORM_INIT,
            )
        ).build()
        self.out_proj = (
            config.out_proj
            or Linear.Config(
                in_features=self.d_inner,
                out_features=self.dim,
                param_init=_LINEAR_INIT,
            )
        ).build()

        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.B_bias = nn.Parameter(torch.empty(self.num_heads, 1, self.d_state))
        self.C_bias = nn.Parameter(torch.empty(self.num_heads, 1, self.d_state))
        self.D = nn.Parameter(torch.empty(self.num_heads))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "Mamba3Attention does not support attention masks or packed "
                "document inputs in this experiment."
            )

        mamba3_siso_combined = _load_mamba3_siso_combined()

        batch_size, seq_len, _ = x.shape
        projected = self.in_proj(x)
        z, v, b, c, dd_dt, dd_a, trap, angles = torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads,
                self.d_state * self.num_bc_heads,
                self.num_heads,
                self.num_heads,
                self.num_heads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        z = z.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        b = b.view(batch_size, seq_len, self.num_bc_heads, self.d_state)
        c = c.view(batch_size, seq_len, self.num_bc_heads, self.d_state)
        b = self.b_norm(b)
        c = self.c_norm(c)

        a = -F.softplus(dd_a.float())
        a = torch.clamp(a, max=-self.a_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        adt = (a * dt).transpose(1, 2).contiguous()
        dt = dt.transpose(1, 2).contiguous()
        trap = trap.transpose(1, 2).contiguous()
        angles = angles.unsqueeze(-2).expand(-1, -1, self.num_heads, -1).float()

        out = mamba3_siso_combined(
            Q=c,
            K=b,
            V=v,
            ADT=adt,
            DT=dt,
            Trap=trap,
            Q_bias=self.C_bias.squeeze(1),
            K_bias=self.B_bias.squeeze(1),
            Angles=angles,
            D=self.D,
            Z=z,
            chunk_size=self.chunk_size,
            Input_States=None,
            return_final_states=False,
            cu_seqlens=None,
        )
        out = out.reshape(batch_size, seq_len, self.d_inner)
        return self.out_proj(out.to(x.dtype))
