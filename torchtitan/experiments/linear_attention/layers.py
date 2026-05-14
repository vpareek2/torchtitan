# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_GDN_STATE_INIT = {
    "A_log": nn.init.zeros_,
    "dt_bias": nn.init.zeros_,
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
