# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.experiments.linear_attention.layers import (
    GatedDeltaAttention,
    KimiDeltaAttention,
    Mamba3Attention,
    RavenAttention,
    SinkGQAttention,
)

from torchtitan.models.common import Embedding, Linear, RoPE
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model_spec import ModelSpec

from .model import HybridQwen3Model, HybridQwen3TransformerBlock
from .parallelize import parallelize_hybrid_qwen3

__all__ = [
    "GatedDeltaAttention",
    "HybridQwen3Model",
    "HybridQwen3TransformerBlock",
    "KimiDeltaAttention",
    "Mamba3Attention",
    "model_registry",
    "parallelize_hybrid_qwen3",
    "RavenAttention",
    "SinkGQAttention",
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
_EPS = 1e-6
_SLIDING_WINDOW_SIZE = 512


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


def _build_kimi_delta_config(dim: int, layer_id: int) -> KimiDeltaAttention.Config:
    value_dim = 3 * 128
    gate_dim = 3 * 64
    return KimiDeltaAttention.Config(
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
        f1_proj=Linear.Config(
            in_features=dim,
            out_features=128,
            param_init=_LINEAR_INIT,
        ),
        f2_proj=Linear.Config(
            in_features=128,
            out_features=gate_dim,
            param_init=_LINEAR_INIT,
        ),
        b_proj=Linear.Config(
            in_features=dim,
            out_features=3,
            param_init=_LINEAR_INIT,
        ),
        g1_proj=Linear.Config(
            in_features=dim,
            out_features=128,
            param_init=_LINEAR_INIT,
        ),
        g2_proj=Linear.Config(
            in_features=128,
            out_features=value_dim,
            bias=True,
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
        param_init=_KDA_STATE_INIT,
    )


def _build_mamba3_config(dim: int, layer_id: int) -> Mamba3Attention.Config:
    d_state = 128
    expand = 2
    head_dim = 64
    num_heads = expand * dim // head_dim
    num_rope_angles = d_state // 4
    in_proj_dim = 2 * expand * dim + 2 * d_state + 3 * num_heads + num_rope_angles
    return Mamba3Attention.Config(
        dim=dim,
        d_state=d_state,
        expand=expand,
        head_dim=head_dim,
        num_bc_heads=1,
        rope_fraction=0.5,
        chunk_size=64,
        in_proj=Linear.Config(
            in_features=dim,
            out_features=in_proj_dim,
            param_init=_LINEAR_INIT,
        ),
        b_norm=RMSNorm.Config(
            normalized_shape=d_state,
            eps=1e-5,
            param_init=_NORM_INIT,
        ),
        c_norm=RMSNorm.Config(
            normalized_shape=d_state,
            eps=1e-5,
            param_init=_NORM_INIT,
        ),
        out_proj=Linear.Config(
            in_features=expand * dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        param_init=_MAMBA3_STATE_INIT,
    )


def _build_raven_config(dim: int, layer_id: int) -> RavenAttention.Config:
    num_heads = 4
    num_slots = dim // num_heads
    return RavenAttention.Config(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        num_slots=num_slots,
        topk=16,
        feature_map="swish",
        router_score="sigmoid",
        add_gumbel_noise=True,
        q_proj=Linear.Config(
            in_features=dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
        ),
        k_proj=Linear.Config(
            in_features=dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
        ),
        v_proj=Linear.Config(
            in_features=dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
        ),
        r_proj=Linear.Config(
            in_features=dim,
            out_features=num_heads * num_slots,
            param_init=_LINEAR_INIT,
        ),
        a_proj=Linear.Config(
            in_features=dim,
            out_features=num_heads,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=num_slots,
            eps=_EPS,
            param_init=_NORM_INIT,
        ),
        k_norm=RMSNorm.Config(
            normalized_shape=num_slots,
            eps=_EPS,
            param_init=_NORM_INIT,
        ),
        g_norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=_EPS,
            param_init=_NORM_INIT,
        ),
        o_proj=Linear.Config(
            in_features=dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
    )


def _build_dense_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_id: int,
    attn_backend: str,
) -> tuple[object, str | None]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    attention_mask_key = "causal" if attn_backend in ("flex", "flex_flash") else None
    return (
        make_gqa_config(
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
        ),
        attention_mask_key,
    )


def _build_swa_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_id: int,
) -> tuple[object, str]:
    inner_attention, mask_type = get_attention_config("flex")
    return (
        SinkGQAttention.Config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qkv_linear=make_gqa_config(
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
            ).qkv_linear,
            wo=Linear.Config(
                in_features=n_heads * head_dim,
                out_features=dim,
                param_init=_depth_init(layer_id),
            ),
            inner_attention=inner_attention,
            mask_type=mask_type,
            rope_backend="cos_sin",
            qk_norm=_qwen3_norm(head_dim),
            param_init={
                "sinks": partial(
                    nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
                )
            },
        ),
        "sliding_window",
    )


def _build_attention_config(
    *,
    attention_kind: str,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_id: int,
    attn_backend: str,
) -> tuple[object, str | None]:
    if attention_kind == "dense":
        return _build_dense_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            attn_backend=attn_backend,
        )
    if attention_kind == "swa":
        return _build_swa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
        )
    if attention_kind == "gdn":
        return _build_gated_delta_config(dim, layer_id), None
    if attention_kind == "kda":
        return _build_kimi_delta_config(dim, layer_id), None
    if attention_kind == "mamba3":
        return _build_mamba3_config(dim, layer_id), None
    if attention_kind == "raven":
        return _build_raven_config(dim, layer_id), None
    raise ValueError(f"Unknown attention_kind: {attention_kind}")


def _build_hybrid_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    attention_pattern: tuple[str, ...],
    attn_backend: str,
):
    if len(attention_pattern) != n_layers:
        raise ValueError(
            f"attention_pattern must have {n_layers} entries, "
            f"got {len(attention_pattern)}."
        )

    layers = []
    for layer_id, attention_kind in enumerate(attention_pattern):
        attention, attention_mask_key = _build_attention_config(
            attention_kind=attention_kind,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            attn_backend=attn_backend,
        )

        layers.append(
            HybridQwen3TransformerBlock.Config(
                attention_norm=_qwen3_norm(dim),
                ffn_norm=_qwen3_norm(dim),
                attention=attention,
                attention_mask_key=attention_mask_key,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _make_10m_hybrid_config(
    *,
    attention_pattern: tuple[str, ...],
    attn_backend: str,
    vocab_size: int = 2048,
) -> HybridQwen3Model.Config:
    dim = 256
    head_dim = 128
    n_layers = 8
    linear_attention_layers = tuple(
        layer_id
        for layer_id, attention_kind in enumerate(attention_pattern)
        if attention_kind in ("gdn", "kda", "mamba3", "raven")
    )
    return HybridQwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        linear_attention_layers=linear_attention_layers,
        sliding_window_size=_SLIDING_WINDOW_SIZE,
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
            attention_pattern=attention_pattern,
            attn_backend=attn_backend,
        ),
    )


def _make_10m_climbmix_hybrid_config(
    *,
    attention_pattern: tuple[str, ...],
    attn_backend: str,
) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=attention_pattern,
        attn_backend=attn_backend,
        vocab_size=50257,
    )


def _10m_gdn_interleaved(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=(
            "dense",
            "gdn",
            "dense",
            "gdn",
            "dense",
            "gdn",
            "dense",
            "gdn",
        ),
        attn_backend=attn_backend,
    )


def _10m_kda_interleaved(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=(
            "dense",
            "kda",
            "dense",
            "kda",
            "dense",
            "kda",
            "dense",
            "kda",
        ),
        attn_backend=attn_backend,
    )


def _10m_mamba3_interleaved(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=(
            "dense",
            "mamba3",
            "dense",
            "mamba3",
            "dense",
            "mamba3",
            "dense",
            "mamba3",
        ),
        attn_backend=attn_backend,
    )


def _10m_raven_interleaved(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=(
            "dense",
            "raven",
            "dense",
            "raven",
            "dense",
            "raven",
            "dense",
            "raven",
        ),
        attn_backend=attn_backend,
    )


def _10m_gdn_dense_3to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=("dense", "gdn", "gdn", "gdn", "dense", "gdn", "gdn", "gdn"),
        attn_backend=attn_backend,
    )


def _10m_kda_dense_3to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=("dense", "kda", "kda", "kda", "dense", "kda", "kda", "kda"),
        attn_backend=attn_backend,
    )


def _10m_kda_dense_1to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=(
            "dense",
            "kda",
            "dense",
            "kda",
            "dense",
            "kda",
            "dense",
            "kda",
        ),
        attn_backend=attn_backend,
    )


def _10m_swa_dense_3to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=(
            "swa",
            "swa",
            "swa",
            "dense",
            "swa",
            "swa",
            "swa",
            "dense",
        ),
        attn_backend=attn_backend,
    )


def _10m_swa_gdn_1to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=("swa", "gdn", "swa", "gdn", "swa", "gdn", "swa", "gdn"),
        attn_backend=attn_backend,
    )


def _10m_swa_kda_1to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_hybrid_config(
        attention_pattern=("swa", "kda", "swa", "kda", "swa", "kda", "swa", "kda"),
        attn_backend=attn_backend,
    )


def _10m_climbmix_gdn_dense_3to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_climbmix_hybrid_config(
        attention_pattern=("dense", "gdn", "gdn", "gdn", "dense", "gdn", "gdn", "gdn"),
        attn_backend=attn_backend,
    )


def _10m_climbmix_kda_dense_3to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_climbmix_hybrid_config(
        attention_pattern=("dense", "kda", "kda", "kda", "dense", "kda", "kda", "kda"),
        attn_backend=attn_backend,
    )


def _10m_climbmix_kda_dense_1to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_climbmix_hybrid_config(
        attention_pattern=(
            "dense",
            "kda",
            "dense",
            "kda",
            "dense",
            "kda",
            "dense",
            "kda",
        ),
        attn_backend=attn_backend,
    )


def _10m_climbmix_swa_dense_3to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_climbmix_hybrid_config(
        attention_pattern=(
            "swa",
            "swa",
            "swa",
            "dense",
            "swa",
            "swa",
            "swa",
            "dense",
        ),
        attn_backend=attn_backend,
    )


def _10m_climbmix_swa_gdn_1to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_climbmix_hybrid_config(
        attention_pattern=("swa", "gdn", "swa", "gdn", "swa", "gdn", "swa", "gdn"),
        attn_backend=attn_backend,
    )


def _10m_climbmix_swa_kda_1to1(attn_backend: str) -> HybridQwen3Model.Config:
    return _make_10m_climbmix_hybrid_config(
        attention_pattern=("swa", "kda", "swa", "kda", "swa", "kda", "swa", "kda"),
        attn_backend=attn_backend,
    )


hybrid_qwen3_configs = {
    "10M-gdn-interleaved": _10m_gdn_interleaved,
    "10M-kda-interleaved": _10m_kda_interleaved,
    "10M-mamba3-interleaved": _10m_mamba3_interleaved,
    "10M-raven-interleaved": _10m_raven_interleaved,
    "10M-gdn-dense-3to1": _10m_gdn_dense_3to1,
    "10M-kda-dense-3to1": _10m_kda_dense_3to1,
    "10M-kda-dense-1to1": _10m_kda_dense_1to1,
    "10M-swa-dense-3to1": _10m_swa_dense_3to1,
    "10M-swa-gdn-1to1": _10m_swa_gdn_1to1,
    "10M-swa-kda-1to1": _10m_swa_kda_1to1,
    "10M-climbmix-gdn-dense-3to1": _10m_climbmix_gdn_dense_3to1,
    "10M-climbmix-kda-dense-3to1": _10m_climbmix_kda_dense_3to1,
    "10M-climbmix-kda-dense-1to1": _10m_climbmix_kda_dense_1to1,
    "10M-climbmix-swa-dense-3to1": _10m_climbmix_swa_dense_3to1,
    "10M-climbmix-swa-gdn-1to1": _10m_climbmix_swa_gdn_1to1,
    "10M-climbmix-swa-kda-1to1": _10m_climbmix_swa_kda_1to1,
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
