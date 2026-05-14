# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util

import pytest
import torch

from torchtitan.config.manager import ConfigManager


def _load_hybrid_config():
    return ConfigManager().parse_args(
        [
            "--module",
            "linear_attention.qwen3",
            "--config",
            "qwen3_10m_gdn_interleaved",
        ]
    )


def test_linear_attention_qwen3_config_resolves():
    config = _load_hybrid_config()

    assert config.model_spec.name == "linear_attention.qwen3"
    assert config.model_spec.flavor == "10M-gdn-interleaved"
    assert config.hf_assets_path == "./tests/assets/tokenizer"
    assert config.training.local_batch_size == 8
    assert config.training.seq_len == 1024
    assert config.training.steps == 10


def test_linear_attention_qwen3_model_builds_and_alternates_layers():
    from torchtitan.experiments.linear_attention import GatedDeltaAttention
    from torchtitan.models.common.attention import GQAttention

    config = _load_hybrid_config()
    model_config = config.model_spec.model
    model_config.update_from_config(trainer_config=config)

    with torch.device("meta"):
        model = model_config.build()

    model.verify_module_protocol()

    attention_types = [type(block.attention) for block in model.layers.values()]
    assert len(attention_types) == 8
    assert attention_types[0::2] == [GQAttention] * 4
    assert attention_types[1::2] == [GatedDeltaAttention] * 4


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
