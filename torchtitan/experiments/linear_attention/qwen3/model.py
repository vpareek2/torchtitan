# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, and_masks

from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


class HybridQwen3TransformerBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Module.Config
        feed_forward: FeedForward.Config
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        attention_mask_key: str | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.attention_mask_key = config.attention_mask_key
        self.attention = config.attention.build()
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.moe_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(attention_masks, dict):
            attention_masks = (
                attention_masks[self.attention_mask_key]
                if self.attention_mask_key is not None
                else None
            )

        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        return x + self.feed_forward(self.ffn_norm(x))


class HybridQwen3Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 256
        vocab_size: int = 2048
        enable_weight_tying: bool = True
        linear_attention_layers: tuple[int, ...] = (1, 3, 5, 7)
        sliding_window_size: int | None = None
        flex_block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            parallelism = trainer_config.parallelism
            training = getattr(trainer_config, "training", None)

            if training is not None:
                seq_len = training.seq_len
                if seq_len > self.rope.max_seq_len:
                    logger.warning(
                        f"Sequence length {seq_len} exceeds original maximum "
                        f"{self.rope.max_seq_len}."
                    )
                self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            if parallelism.tensor_parallel_degree > 1:
                raise ValueError(
                    "Tensor Parallel is not supported for GatedDeltaAttention "
                    "in the linear_attention.qwen3 experiment."
                )
            if parallelism.context_parallel_degree > 1:
                raise ValueError(
                    "Context Parallel is not supported for GatedDeltaAttention "
                    "in the linear_attention.qwen3 experiment."
                )
            if parallelism.pipeline_parallel_degree > 1:
                raise ValueError(
                    "Pipeline Parallel is not supported for GatedDeltaAttention "
                    "in the linear_attention.qwen3 experiment."
                )
            if parallelism.expert_parallel_degree > 1:
                raise ValueError(
                    "Expert Parallel is not supported by the dense "
                    "linear_attention.qwen3 experiment."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            return nparams, 6 * nparams

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying
        self.sliding_window_size = config.sliding_window_size
        self.flex_block_size = config.flex_block_size

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        mask_keys = {
            layer.attention_mask_key
            for layer in self.layers.values()
            if layer.attention_mask_key is not None
        }
        if not mask_keys:
            return super().get_attention_masks(positions)

        seq_len = positions.shape[1]
        masks = {}
        if "causal" in mask_keys:
            masks["causal"] = create_attention_mask(
                and_masks(get_causal_mask_mod(), get_document_mask_mod(positions)),
                positions.shape[0],
                None,
                seq_len,
                seq_len,
                BLOCK_SIZE=self.flex_block_size,
            )

        if "sliding_window" in mask_keys:
            if self.sliding_window_size is None:
                raise ValueError(
                    "sliding_window_size must be set for sliding-window masks."
                )
            masks["sliding_window"] = create_attention_mask(
                and_masks(
                    get_document_mask_mod(positions),
                    get_sliding_window_mask_mod(self.sliding_window_size),
                ),
                positions.shape[0],
                None,
                seq_len,
                seq_len,
                BLOCK_SIZE=self.flex_block_size,
            )

        return masks

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            assert self.tok_embeddings is not None and self.lm_head is not None
            self.tok_embeddings.weight = self.lm_head.weight

        super().init_states(buffer_device=buffer_device)
