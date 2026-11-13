"""Fail-closed contracts for vLLM's rank-1 ProjectionCache path."""

from __future__ import annotations

import pytest
import torch

from abliterix.core.vllm_backend import ProjectionCache, VLLMGenerator
from abliterix.settings import AbliterixConfig, ModelConfig


@pytest.mark.parametrize(
    ("steering", "iterative", "reason"),
    [
        ({"n_directions": 2}, None, "n_directions=2"),
        (
            {"ablate_harmfulness_direction": True},
            None,
            "ablate_harmfulness_direction=true",
        ),
        (
            {"search_harmfulness_direction": True},
            None,
            "search_harmfulness_direction=true",
        ),
        ({"vector_method": "som"}, None, "vector_method='som'"),
        (
            {"vector_method": "sae", "sae_path": "/tmp/test-sae.pt"},
            None,
            "vector_method='sae'",
        ),
        (
            {"weight_normalization": "full"},
            None,
            "weight_normalization='full'",
        ),
        (
            {},
            {"enabled": True, "per_iteration_directions": 1},
            "iterative.enabled=true",
        ),
    ],
)
def test_vllm_rejects_recipes_that_produce_rank_k_vectors(steering, iterative, reason):
    kwargs = {
        "model": {"model_id": "test/model", "backend": "vllm"},
        "steering": steering,
    }
    if iterative is not None:
        kwargs["iterative"] = iterative

    with pytest.raises(ValueError, match=reason):
        AbliterixConfig(**kwargs)


def test_vllm_conv_span_needs_in_place_editing():
    """The LoRA path is rank 1; the in-place path applies the export's plan,
    conv span included."""
    with pytest.raises(
        ValueError, match="conv_span_projection=true without use_in_place_editing"
    ):
        AbliterixConfig(
            model={"model_id": "test/model", "backend": "vllm"},
            steering={"conv_span_projection": True},
        )
    config = AbliterixConfig(
        model={
            "model_id": "test/model",
            "backend": "vllm",
            "use_in_place_editing": True,
        },
        steering={"conv_span_projection": True},
    )
    assert config.steering.conv_span_projection


def test_hf_keeps_rank_k_recipes_available():
    config = AbliterixConfig(
        model={"model_id": "test/model", "backend": "hf"},
        steering={"n_directions": 3},
    )

    assert config.steering.n_directions == 3


@pytest.mark.parametrize("rank", [1, 8, 16, 32, 64, 128, 256, 320, 512])
def test_model_config_accepts_vllm_supported_lora_ranks(rank):
    config = ModelConfig(model_id="test/model", vllm_max_lora_rank=rank)

    assert config.vllm_max_lora_rank == rank


@pytest.mark.parametrize("rank", [0, 2, 3, 4, 7, 12, 1024])
def test_model_config_rejects_vllm_unsupported_lora_ranks(rank):
    with pytest.raises(ValueError, match="vllm_max_lora_rank"):
        ModelConfig(model_id="test/model", vllm_max_lora_rank=rank)


class _UnusedEngine:
    @property
    def transformer_layers(self):  # pragma: no cover - must fail first
        raise AssertionError(
            "ProjectionCache inspected the engine before rank validation"
        )


def test_projection_cache_build_rejects_3d_vectors_before_touching_engine():
    vectors = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match="2-D single-direction"):
        ProjectionCache.build(_UnusedEngine(), vectors)


def test_projection_cache_safetensors_build_rejects_3d_vectors_before_io():
    config = AbliterixConfig(model={"model_id": "does/not/exist"})
    vectors = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match="2-D single-direction"):
        ProjectionCache.build_from_safetensors(config, vectors)


def test_save_adapter_rejects_rank_above_engine_capacity(tmp_path):
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._lora_disabled = False
    generator._adapter_dir = str(tmp_path / "adapter")
    generator._lora_max_rank = 1
    weights = {"model.layers.0.o_proj": (torch.randn(2, 3), torch.randn(4, 2))}

    with pytest.raises(ValueError, match="rank 2.*max_lora_rank=1"):
        generator.save_adapter(weights, ["o_proj"], "test/model")
