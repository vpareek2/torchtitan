# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util

import pytest
import torch

from torchtitan.config.manager import ConfigManager


def _load_config(config_name):
    return ConfigManager().parse_args(
        [
            "--module",
            "linear_attention.qwen3",
            "--config",
            config_name,
        ]
    )


def _load_qwen3_config(config_name):
    return ConfigManager().parse_args(
        [
            "--module",
            "qwen3",
            "--config",
            config_name,
        ]
    )


def _load_hybrid_config():
    return _load_config("qwen3_10m_gdn_interleaved")


def _load_kda_hybrid_config():
    return _load_config("qwen3_10m_kda_interleaved")


def _load_mamba3_hybrid_config():
    return _load_config("qwen3_10m_mamba3_interleaved")


def _load_raven_hybrid_config():
    return _load_config("qwen3_10m_raven_interleaved")


@pytest.mark.parametrize(
    ("load_config", "flavor", "activation_checkpoint_mode"),
    [
        (_load_hybrid_config, "10M-gdn-interleaved", "selective"),
        (_load_kda_hybrid_config, "10M-kda-interleaved", "selective"),
        (_load_mamba3_hybrid_config, "10M-mamba3-interleaved", "none"),
        (_load_raven_hybrid_config, "10M-raven-interleaved", "selective"),
        (
            lambda: _load_config("qwen3_10m_gdn_dense_3to1"),
            "10M-gdn-dense-3to1",
            "selective",
        ),
        (
            lambda: _load_config("qwen3_10m_kda_dense_3to1"),
            "10M-kda-dense-3to1",
            "selective",
        ),
        (
            lambda: _load_config("qwen3_10m_kda_dense_1to1"),
            "10M-kda-dense-1to1",
            "selective",
        ),
        (
            lambda: _load_config("qwen3_10m_swa_dense_3to1"),
            "10M-swa-dense-3to1",
            "selective",
        ),
        (
            lambda: _load_config("qwen3_10m_swa_gdn_1to1"),
            "10M-swa-gdn-1to1",
            "selective",
        ),
        (
            lambda: _load_config("qwen3_10m_swa_kda_1to1"),
            "10M-swa-kda-1to1",
            "selective",
        ),
    ],
)
def test_linear_attention_qwen3_config_resolves(
    load_config, flavor, activation_checkpoint_mode
):
    config = load_config()

    assert config.model_spec.name == "linear_attention.qwen3"
    assert config.model_spec.flavor == flavor
    assert config.hf_assets_path == "./tests/assets/tokenizer"
    assert config.training.local_batch_size == 8
    assert config.training.seq_len == 1024
    assert config.training.steps == 10
    assert config.activation_checkpoint.mode == activation_checkpoint_mode


@pytest.mark.parametrize(
    ("config_name", "flavor"),
    [
        ("qwen3_10m_climbmix_gdn_dense_3to1", "10M-climbmix-gdn-dense-3to1"),
        ("qwen3_10m_climbmix_kda_dense_3to1", "10M-climbmix-kda-dense-3to1"),
        ("qwen3_10m_climbmix_kda_dense_1to1", "10M-climbmix-kda-dense-1to1"),
        ("qwen3_10m_climbmix_swa_dense_3to1", "10M-climbmix-swa-dense-3to1"),
        ("qwen3_10m_climbmix_swa_gdn_1to1", "10M-climbmix-swa-gdn-1to1"),
        ("qwen3_10m_climbmix_swa_kda_1to1", "10M-climbmix-swa-kda-1to1"),
    ],
)
def test_linear_attention_qwen3_climbmix_config_resolves(config_name, flavor):
    config = _load_config(config_name)

    assert config.model_spec.name == "linear_attention.qwen3"
    assert config.model_spec.flavor == flavor
    assert config.model_spec.model.vocab_size == 50257
    assert config.hf_assets_path == "./assets/hf/gpt2"
    assert config.dataloader.dataset == "climbmix"
    assert config.dataloader.dataset_path == "./data/climbmix_50m"
    assert config.training.local_batch_size == 8
    assert config.training.seq_len == 1024
    assert config.training.steps == 6104
    assert config.lr_scheduler.warmup_steps == 122
    assert config.lr_scheduler.decay_type == "cosine"
    assert config.lr_scheduler.min_lr_factor == 0.1
    assert config.validator.enable
    assert config.validator.freq == 610
    assert config.validator.steps == 128
    assert config.validator.dataloader.dataset == "climbmix_validation"
    assert config.validator.dataloader.dataset_path == "./data/climbmix_50m"
    assert config.optimizer.param_groups == []


def test_qwen3_dense_climbmix_config_resolves():
    config = _load_qwen3_config("qwen3_10m_climbmix_dense")

    assert config.model_spec.name == "qwen3"
    assert config.model_spec.flavor == "10M-climbmix"
    assert config.model_spec.model.vocab_size == 50257
    assert config.hf_assets_path == "./assets/hf/gpt2"
    assert config.dataloader.dataset == "climbmix"
    assert config.dataloader.dataset_path == "./data/climbmix_50m"
    assert config.training.steps == 6104
    assert config.validator.enable
    assert config.validator.dataloader.dataset == "climbmix_validation"
    assert config.validator.dataloader.dataset_path == "./data/climbmix_50m"


@pytest.mark.parametrize(
    ("load_config", "linear_attention_cls_name"),
    [
        (_load_hybrid_config, "GatedDeltaAttention"),
        (_load_kda_hybrid_config, "KimiDeltaAttention"),
        (_load_mamba3_hybrid_config, "Mamba3Attention"),
        (_load_raven_hybrid_config, "RavenAttention"),
    ],
)
def test_linear_attention_qwen3_model_builds_and_alternates_layers(
    load_config, linear_attention_cls_name
):
    from torchtitan.experiments import linear_attention
    from torchtitan.models.common.attention import GQAttention

    config = load_config()
    model_config = config.model_spec.model
    model_config.update_from_config(trainer_config=config)

    with torch.device("meta"):
        model = model_config.build()

    model.verify_module_protocol()

    attention_types = [type(block.attention) for block in model.layers.values()]
    linear_attention_cls = getattr(linear_attention, linear_attention_cls_name)
    assert len(attention_types) == 8
    assert attention_types[0::2] == [GQAttention] * 4
    assert attention_types[1::2] == [linear_attention_cls] * 4


@pytest.mark.parametrize(
    ("config_name", "expected_types", "expected_mask_keys"),
    [
        (
            "qwen3_10m_gdn_dense_3to1",
            [
                "GQAttention",
                "GatedDeltaAttention",
                "GatedDeltaAttention",
                "GatedDeltaAttention",
                "GQAttention",
                "GatedDeltaAttention",
                "GatedDeltaAttention",
                "GatedDeltaAttention",
            ],
            [None, None, None, None, None, None, None, None],
        ),
        (
            "qwen3_10m_kda_dense_3to1",
            [
                "GQAttention",
                "KimiDeltaAttention",
                "KimiDeltaAttention",
                "KimiDeltaAttention",
                "GQAttention",
                "KimiDeltaAttention",
                "KimiDeltaAttention",
                "KimiDeltaAttention",
            ],
            [None, None, None, None, None, None, None, None],
        ),
        (
            "qwen3_10m_swa_dense_3to1",
            [
                "SinkGQAttention",
                "SinkGQAttention",
                "SinkGQAttention",
                "GQAttention",
                "SinkGQAttention",
                "SinkGQAttention",
                "SinkGQAttention",
                "GQAttention",
            ],
            [
                "sliding_window",
                "sliding_window",
                "sliding_window",
                None,
                "sliding_window",
                "sliding_window",
                "sliding_window",
                None,
            ],
        ),
        (
            "qwen3_10m_kda_dense_1to1",
            [
                "GQAttention",
                "KimiDeltaAttention",
                "GQAttention",
                "KimiDeltaAttention",
                "GQAttention",
                "KimiDeltaAttention",
                "GQAttention",
                "KimiDeltaAttention",
            ],
            [None, None, None, None, None, None, None, None],
        ),
        (
            "qwen3_10m_swa_gdn_1to1",
            [
                "SinkGQAttention",
                "GatedDeltaAttention",
                "SinkGQAttention",
                "GatedDeltaAttention",
                "SinkGQAttention",
                "GatedDeltaAttention",
                "SinkGQAttention",
                "GatedDeltaAttention",
            ],
            [
                "sliding_window",
                None,
                "sliding_window",
                None,
                "sliding_window",
                None,
                "sliding_window",
                None,
            ],
        ),
        (
            "qwen3_10m_swa_kda_1to1",
            [
                "SinkGQAttention",
                "KimiDeltaAttention",
                "SinkGQAttention",
                "KimiDeltaAttention",
                "SinkGQAttention",
                "KimiDeltaAttention",
                "SinkGQAttention",
                "KimiDeltaAttention",
            ],
            [
                "sliding_window",
                None,
                "sliding_window",
                None,
                "sliding_window",
                None,
                "sliding_window",
                None,
            ],
        ),
    ],
)
def test_linear_attention_qwen3_baseline_matrix_patterns(
    config_name, expected_types, expected_mask_keys
):
    config = _load_config(config_name)
    model_config = config.model_spec.model
    model_config.update_from_config(trainer_config=config)

    with torch.device("meta"):
        model = model_config.build()

    model.verify_module_protocol()

    assert [type(block.attention).__name__ for block in model.layers.values()] == (
        expected_types
    )
    assert [block.attention_mask_key for block in model.layers.values()] == (
        expected_mask_keys
    )


@pytest.mark.parametrize(
    ("field_name", "field_value", "match"),
    [
        ("tensor_parallel_degree", 2, "Tensor Parallel"),
        ("context_parallel_degree", 2, "Context Parallel"),
        ("pipeline_parallel_degree", 2, "Pipeline Parallel"),
        ("expert_parallel_degree", 2, "Expert Parallel"),
    ],
)
def test_linear_attention_qwen3_rejects_unsupported_parallelism(
    field_name, field_value, match
):
    config = _load_hybrid_config()
    setattr(config.parallelism, field_name, field_value)

    with pytest.raises(ValueError, match=match):
        config.model_spec.model.update_from_config(trainer_config=config)


def test_gated_delta_attention_rejects_attention_masks_without_fla():
    from torchtitan.experiments.linear_attention import GatedDeltaAttention

    module = GatedDeltaAttention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="attention masks"):
        module(x, None, torch.ones(2, 4, dtype=torch.bool))


def test_gated_delta_attention_lazy_import_error_without_fla():
    if importlib.util.find_spec("fla") is not None:
        pytest.skip("flash-linear-attention is installed")

    from torchtitan.experiments.linear_attention import GatedDeltaAttention

    module = GatedDeltaAttention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ImportError, match="flash-linear-attention"):
        module(x, None, None)


def test_kimi_delta_attention_rejects_attention_masks_without_fla():
    from torchtitan.experiments.linear_attention import KimiDeltaAttention

    module = KimiDeltaAttention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="attention masks"):
        module(x, None, torch.ones(2, 4, dtype=torch.bool))


def test_kimi_delta_attention_lazy_import_error_without_fla():
    if importlib.util.find_spec("fla") is not None:
        pytest.skip("flash-linear-attention is installed")

    from torchtitan.experiments.linear_attention import KimiDeltaAttention

    module = KimiDeltaAttention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ImportError, match="flash-linear-attention"):
        module(x, None, None)


def test_mamba3_attention_rejects_attention_masks_without_mamba_ssm():
    from torchtitan.experiments.linear_attention import Mamba3Attention

    module = Mamba3Attention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="attention masks"):
        module(x, None, torch.ones(2, 4, dtype=torch.bool))


def test_mamba3_attention_lazy_import_error_without_mamba_ssm():
    if importlib.util.find_spec("mamba_ssm") is not None:
        pytest.skip("mamba-ssm is installed")

    from torchtitan.experiments.linear_attention import Mamba3Attention

    module = Mamba3Attention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ImportError, match="mamba-ssm"):
        module(x, None, None)


def test_raven_attention_rejects_attention_masks_without_fla():
    from torchtitan.experiments.linear_attention import RavenAttention

    module = RavenAttention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="attention masks"):
        module(x, None, torch.ones(2, 4, dtype=torch.bool))


def test_raven_attention_lazy_import_error_without_fla():
    if importlib.util.find_spec("fla") is not None:
        pytest.skip("flash-linear-attention is installed")

    from torchtitan.experiments.linear_attention import RavenAttention

    module = RavenAttention.Config(dim=32).build()
    x = torch.randn(2, 4, 32)

    with pytest.raises(ImportError, match="flash-linear-attention"):
        module(x, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    importlib.util.find_spec("fla") is None,
    reason="flash-linear-attention is not installed",
)
def test_gated_delta_attention_cuda_forward_backward():
    from torchtitan.experiments.linear_attention import GatedDeltaAttention

    module = GatedDeltaAttention.Config(dim=256).build().cuda().bfloat16()
    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)

    out = module(x, None, None)
    assert out.shape == x.shape

    out.float().sum().backward()

    assert module.q_proj.weight.grad is not None
    assert module.k_proj.weight.grad is not None
    assert module.v_proj.weight.grad is not None
    assert module.a_proj.weight.grad is not None
    assert module.b_proj.weight.grad is not None
    assert module.g_proj.weight.grad is not None
    assert module.o_proj.weight.grad is not None
    assert module.A_log.grad is not None
    assert module.dt_bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None,
    reason="mamba-ssm with Mamba3 is not installed",
)
def test_mamba3_attention_cuda_forward_backward():
    from torchtitan.experiments.linear_attention import Mamba3Attention

    module = Mamba3Attention.Config(dim=256).build().cuda().bfloat16()
    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)

    out = module(x, None, None)
    assert out.shape == x.shape

    out.float().sum().backward()

    assert module.in_proj.weight.grad is not None
    assert module.out_proj.weight.grad is not None
    assert module.dt_bias.grad is not None
    assert module.B_bias.grad is not None
    assert module.C_bias.grad is not None
    assert module.D.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    importlib.util.find_spec("fla") is None,
    reason="flash-linear-attention is not installed",
)
def test_kimi_delta_attention_cuda_forward_backward():
    from torchtitan.experiments.linear_attention import KimiDeltaAttention

    module = KimiDeltaAttention.Config(dim=256).build().cuda().bfloat16()
    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)

    out = module(x, None, None)
    assert out.shape == x.shape

    out.float().sum().backward()

    assert module.q_proj.weight.grad is not None
    assert module.k_proj.weight.grad is not None
    assert module.v_proj.weight.grad is not None
    assert module.f1_proj.weight.grad is not None
    assert module.f2_proj.weight.grad is not None
    assert module.b_proj.weight.grad is not None
    assert module.g1_proj.weight.grad is not None
    assert module.g2_proj.weight.grad is not None
    assert module.o_proj.weight.grad is not None
    assert module.A_log.grad is not None
    assert module.dt_bias.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    importlib.util.find_spec("fla") is None,
    reason="flash-linear-attention is not installed",
)
def test_raven_attention_cuda_forward_backward():
    from torchtitan.experiments.linear_attention import RavenAttention

    module = RavenAttention.Config(dim=256).build().cuda().bfloat16()
    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)

    out = module(x, None, None)
    assert out.shape == x.shape

    out.float().sum().backward()

    assert module.q_proj.weight.grad is not None
    assert module.k_proj.weight.grad is not None
    assert module.v_proj.weight.grad is not None
    assert module.r_proj.weight.grad is not None
    assert module.a_proj.weight.grad is not None
    assert module.o_proj.weight.grad is not None
    assert module.A_log.grad is not None
    assert module.dt_bias.grad is not None
