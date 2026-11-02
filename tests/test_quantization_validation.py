from types import SimpleNamespace

import pytest
import torch
from torch import nn

from abliterix.core.quantization_validation import validate_quantization_state


def _model(state, *, weight_device="cpu"):
    model = nn.Module()
    model.projection = nn.Linear(2, 2, bias=False, device=weight_device)
    model.projection.weight.quant_state = state
    return model


def _state():
    return SimpleNamespace(
        code=torch.arange(16, dtype=torch.float32),
        absmax=torch.ones(2),
        offset=torch.tensor(0.0),
        state2=None,
    )


@pytest.mark.parametrize("field", ["code", "absmax", "offset"])
@pytest.mark.parametrize("nested", [False, True])
def test_rejects_meta_quantization_metadata_with_location(field, nested):
    state = _state()
    target = state
    if nested:
        state.state2 = target = _state()
    setattr(target, field, torch.empty(2, device="meta"))
    model = _model(state)
    path = f"projection.weight.quant_state.{('state2.' if nested else '')}{field}"

    with pytest.raises(RuntimeError) as exc:
        validate_quantization_state(model, stage="after adapter initialization")

    message = str(exc.value)
    assert path in message
    assert "after adapter initialization" in message
    assert "CUDA_VISIBLE_DEVICES" in message
    assert "model.max_memory" in message
    assert "hf_device_map" in message
    assert getattr(target, field).is_meta


def test_valid_cpu_quantization_state_is_unchanged():
    state = _state()
    state.state2 = _state()
    model = _model(state)
    original_weight = model.projection.weight
    original_code = state.code
    original_scales = state.state2.absmax

    validate_quantization_state(model, stage="after loading")

    assert model.projection.weight is original_weight
    assert state.code is original_code
    assert state.state2.absmax is original_scales


def test_ordinary_offloaded_meta_weights_are_not_rejected():
    model = nn.Linear(2, 2, device="meta")
    model.hf_device_map = {"": "disk"}

    validate_quantization_state(model, stage="after loading")

    assert model.weight.is_meta


def test_offloaded_weight_with_materialized_quantization_state_is_allowed():
    model = _model(_state(), weight_device="meta")
    model.hf_device_map = {"projection": "disk"}

    validate_quantization_state(model, stage="after loading")

    assert model.projection.weight.is_meta


def test_absent_lazy_quantization_state_is_allowed():
    validate_quantization_state(_model(None), stage="after loading")


def test_module_quantization_state_alias_is_checked():
    model = _model(None)
    state = _state()
    state.code = torch.empty(16, device="meta")
    model.projection.quant_state = state

    with pytest.raises(RuntimeError, match=r"projection\.quant_state\.code"):
        validate_quantization_state(model, stage="after loading")


def test_detects_state_lost_between_loading_and_adapter_initialization():
    state = _state()
    model = _model(state)
    validate_quantization_state(model, stage="after loading")

    # Reproduce the relevant PEFT/offload transition without optional kernels:
    # the weight still exists, but its codebook has become an empty placeholder.
    state.code = torch.empty_like(state.code, device="meta")

    with pytest.raises(RuntimeError, match="after adapter initialization"):
        validate_quantization_state(model, stage="after adapter initialization")


@pytest.mark.parametrize("corrupt_before_wrapping", [False, True])
def test_adapter_initialization_validates_both_sides_of_peft_wrapping(
    monkeypatch, abliterix_config, corrupt_before_wrapping
):
    from transformers import PretrainedConfig, PreTrainedModel

    from abliterix.core.engine import SteeringEngine

    class TinyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self):
            super().__init__(PretrainedConfig())
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.o_proj = nn.Linear(2, 2, bias=False)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])

    model = TinyModel()
    state = _state()
    model.model.layers[0].self_attn.o_proj.weight.quant_state = state
    engine = object.__new__(SteeringEngine)
    engine.model = model
    engine.config = abliterix_config
    wrapped = []

    def wrap(base_model, peft_config):
        wrapped.append(base_model)
        assert peft_config.target_modules == {"model.layers.0.self_attn.o_proj"}
        state.code = torch.empty_like(state.code, device="meta")
        return base_model

    monkeypatch.setattr("abliterix.core.engine.get_peft_model", wrap)
    stage = "after adapter initialization"
    if corrupt_before_wrapping:
        state.code = torch.empty_like(state.code, device="meta")
        stage = "before adapter initialization"

    with pytest.raises(RuntimeError, match=stage):
        engine._init_adapters()

    assert bool(wrapped) is not corrupt_before_wrapping
