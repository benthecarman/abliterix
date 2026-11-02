from __future__ import annotations

import torch
import torch.nn as nn

from abliterix.core.vllm_backend import ProjectionCache


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(3, 4, bias=False)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer()])


class _ToyEngine:
    def __init__(self) -> None:
        self.model = _ToyModel()
        self.transformer_layers = list(self.model.layers)

    def steerable_modules(self, layer_idx: int):
        layer = self.model.layers[layer_idx]
        return {"attn.o_proj": [layer.o_proj]}


def test_projection_cache_build_accepts_plain_linear_without_base_layer():
    engine = _ToyEngine()
    steering_vectors = torch.randn(2, 4)

    cache = ProjectionCache.build(engine, steering_vectors)

    info = cache.projections[0]["attn.o_proj"]
    assert info["module_path"] == "layers.0.o_proj"
    assert info["direction"] == "output"
    assert info["vW_all"].shape == (2, 3)
    assert cache.target_modules == ["o_proj"]
