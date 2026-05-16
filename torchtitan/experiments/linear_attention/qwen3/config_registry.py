# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry

_CLIMBMIX_50M_PATH = "./data/climbmix_50m"


def qwen3_10m_gdn_interleaved() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("10M-gdn-interleaved", attn_backend="sdpa"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=1024,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def _qwen3_10m_climbmix_base(
    *,
    flavor: str,
) -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/gpt2",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry(flavor, attn_backend="sdpa"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="climbmix",
            dataset_path=_CLIMBMIX_50M_PATH,
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=122,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=1024,
            steps=6104,
        ),
        checkpoint=CheckpointManager.Config(
            interval=610,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        validator=Validator.Config(
            enable=True,
            freq=610,
            steps=128,
            dataloader=HuggingFaceTextDataLoader.Config(
                dataset="climbmix_validation",
                dataset_path=_CLIMBMIX_50M_PATH,
                infinite=False,
            ),
        ),
    )


def qwen3_10m_kda_interleaved() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-kda-interleaved", attn_backend="sdpa")
    return config


def qwen3_10m_mamba3_interleaved() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-mamba3-interleaved", attn_backend="sdpa")
    config.activation_checkpoint.mode = "none"
    return config


def qwen3_10m_raven_interleaved() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-raven-interleaved", attn_backend="sdpa")
    return config


def qwen3_10m_gdn_dense_3to1() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-gdn-dense-3to1", attn_backend="sdpa")
    return config


def qwen3_10m_kda_dense_3to1() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-kda-dense-3to1", attn_backend="sdpa")
    return config


def qwen3_10m_kda_dense_1to1() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-kda-dense-1to1", attn_backend="sdpa")
    return config


def qwen3_10m_swa_dense_3to1() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-swa-dense-3to1", attn_backend="sdpa")
    return config


def qwen3_10m_swa_gdn_1to1() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-swa-gdn-1to1", attn_backend="sdpa")
    return config


def qwen3_10m_swa_kda_1to1() -> Trainer.Config:
    config = qwen3_10m_gdn_interleaved()
    config.model_spec = model_registry("10M-swa-kda-1to1", attn_backend="sdpa")
    return config


def qwen3_10m_climbmix_gdn_dense_3to1() -> Trainer.Config:
    return _qwen3_10m_climbmix_base(flavor="10M-climbmix-gdn-dense-3to1")


def qwen3_10m_climbmix_kda_dense_3to1() -> Trainer.Config:
    return _qwen3_10m_climbmix_base(flavor="10M-climbmix-kda-dense-3to1")


def qwen3_10m_climbmix_kda_dense_1to1() -> Trainer.Config:
    return _qwen3_10m_climbmix_base(flavor="10M-climbmix-kda-dense-1to1")


def qwen3_10m_climbmix_swa_dense_3to1() -> Trainer.Config:
    return _qwen3_10m_climbmix_base(flavor="10M-climbmix-swa-dense-3to1")


def qwen3_10m_climbmix_swa_gdn_1to1() -> Trainer.Config:
    return _qwen3_10m_climbmix_base(flavor="10M-climbmix-swa-gdn-1to1")


def qwen3_10m_climbmix_swa_kda_1to1() -> Trainer.Config:
    return _qwen3_10m_climbmix_base(flavor="10M-climbmix-swa-kda-1to1")
