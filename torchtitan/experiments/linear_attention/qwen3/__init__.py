# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.models.common import Embedding, Linear, RoPE
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan.experiments.linear_attention.layers import GatedDeltaAttention

from .model import HybridQwen3Model, HybridQwen3TransformerBlock
from .parallelize import parallelize_hybrid_qwen3

__all__ = [
    "GatedDeltaAttention",
    "HybridQwen3Model",
    "HybridQwen3TransformerBlock",
    "model_registry",
    "parallelize_hybrid_qwen3",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_GDN_STATE_INIT = {
    "A_log": nn.init.zeros_,
    "dt_bias": nn.init.zeros_,
}
_EPS = 1e-6


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _qwen3_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _build_gated_delta_config(dim: int, layer_id: int) -> GatedDeltaAttention.Config:
    value_dim = 3 * 128
    return GatedDeltaAttention.Config(
        dim=dim,
        num_heads=3,
        head_dim=64,
        num_v_heads=3,
        head_v_dim=128,
        q_proj=Linear.Config(
            in_features=dim,
            out_features=3 * 64,
            param_init=_LINEAR_INIT,
        ),
        k_proj=Linear.Config(
            in_features=dim,
            out_features=3 * 64,
            param_init=_LINEAR_INIT,
        ),
        v_proj=Linear.Config(
            in_features=dim,
            out_features=value_dim,
            param_init=_LINEAR_INIT,
        ),
        a_proj=Linear.Config(
            in_features=dim,
            out_features=3,
            param_init=_LINEAR_INIT,
        ),
        b_proj=Linear.Config(
            in_features=dim,
            out_features=3,
            param_init=_LINEAR_INIT,
        ),
        g_proj=Linear.Config(
            in_features=dim,
            out_features=value_dim,
            param_init=_LINEAR_INIT,
        ),
        o_norm=RMSNorm.Config(
            normalized_shape=128,
            eps=_EPS,
            param_init=_NORM_INIT,
        ),
        o_proj=Linear.Config(
            in_features=value_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        param_init=_GDN_STATE_INIT,
    )


def _build_hybrid_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    gated_delta_layers: tuple[int, ...],
    attn_backend: str,
):
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        if layer_id in gated_delta_layers:
            attention = _build_gated_delta_config(dim, layer_id)
        else:
            attention = make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                mask_type=mask_type,
                rope_backend="cos_sin",
                qk_norm=_qwen3_norm(head_dim),
            )

        layers.append(
            HybridQwen3TransformerBlock.Config(
                attention_norm=_qwen3_norm(dim),
                ffn_norm=_qwen3_norm(dim),
                attention=attention,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _10m_gdn_interleaved(attn_backend: str) -> HybridQwen3Model.Config:
    dim = 256
    head_dim = 128
    n_layers = 8
    vocab_size = 2048
    gated_delta_layers = (1, 3, 5, 7)
    return HybridQwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        gated_delta_layers=gated_delta_layers,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_hybrid_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            hidden_dim=1024,
            gated_delta_layers=gated_delta_layers,
            attn_backend=attn_backend,
        ),
    )


hybrid_qwen3_configs = {
    "10M-gdn-interleaved": _10m_gdn_interleaved,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> ModelSpec:
    config = hybrid_qwen3_configs[flavor](attn_backend=attn_backend)
    return ModelSpec(
        name="linear_attention.qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_hybrid_qwen3,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
