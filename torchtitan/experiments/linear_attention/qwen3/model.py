# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.experiments.linear_attention.layers import GatedDeltaAttention
from torchtitan.models.common.attention import AttentionMasksType
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

    def __init__(self, config: Config):
        super().__init__()
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
        gated_delta_layers: tuple[int, ...] = (1, 3, 5, 7)

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

        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            assert self.tok_embeddings is not None and self.lm_head is not None
            self.tok_embeddings.weight = self.lm_head.weight

        super().init_states(buffer_device=buffer_device)
