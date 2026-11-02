from types import SimpleNamespace

import pytest
import torch
from torch import nn

from abliterix.core.bnb_utils import (
    dequantize_bnb_int8_weight,
    is_bnb_int8_weight,
    validate_bnb_int8_weight,
)


def _layer():
    layer = nn.Linear(3, 2, bias=False)
    layer.weight = nn.Parameter(
        torch.tensor([[127, -127, 0], [64, 0, -64]], dtype=torch.int8),
        requires_grad=False,
    )
    layer.weight.CB = None
    layer.weight.SCB = None
    layer.state = SimpleNamespace(CB=None, SCB=None, has_fp16_weights=False)
    return layer


@pytest.mark.parametrize("owner_name", ["weight", "state"])
@pytest.mark.parametrize("data_only", [False, True])
def test_parameter_and_forward_state_dequantize_to_same_matrix(owner_name, data_only):
    layer = _layer()
    owner = getattr(layer, owner_name)
    owner.CB = None if data_only else layer.weight.detach()
    owner.SCB = torch.tensor([2.0, 4.0])
    original_data = layer.weight.detach().clone()

    assert is_bnb_int8_weight(layer)
    validate_bnb_int8_weight(layer)
    result = dequantize_bnb_int8_weight(layer)

    expected = torch.tensor([[2.0, -2.0, 0.0], [256 / 127, 0.0, -256 / 127]])
    torch.testing.assert_close(result, expected)
    assert result.dtype == torch.float32
    torch.testing.assert_close(layer.weight, original_data)
    assert owner.CB is None if data_only else owner.CB is not None


def test_complete_state_pair_is_used_when_weight_lost_its_scales():
    layer = _layer()
    layer.state.CB = layer.weight.detach()
    layer.weight.CB = layer.state.CB
    layer.state.SCB = torch.tensor([2.0, 4.0])

    result = dequantize_bnb_int8_weight(layer)

    assert result[0, 0] == 2.0
    assert result[0, 1] == -2.0
    assert layer.weight.SCB is layer.state.SCB


def test_mismatched_partial_weight_codes_are_not_paired_with_state_scales():
    layer = _layer()
    layer.weight.CB = layer.weight.detach().clone()
    layer.state.CB = layer.weight.detach()
    layer.state.SCB = torch.tensor([2.0, 4.0])

    with pytest.raises(RuntimeError, match="does not share storage"):
        dequantize_bnb_int8_weight(layer)

    assert layer.weight.SCB is None


def test_complete_state_pair_precedes_stale_partial_weight_scales():
    layer = _layer()
    layer.weight.SCB = torch.tensor([20.0, 40.0])
    layer.state.CB = layer.weight.detach()
    layer.state.SCB = torch.tensor([2.0, 4.0])

    assert dequantize_bnb_int8_weight(layer)[0, 0] == 2.0


@pytest.mark.parametrize("owner_name", ["weight", "state"])
def test_raw_data_fallback_restores_live_runtime_metadata(owner_name):
    layer = _layer()
    scales = torch.tensor([2.0, 4.0])
    getattr(layer, owner_name).SCB = scales
    layer.init_8bit_state = lambda: None
    original = layer.weight.detach().clone()

    validate_bnb_int8_weight(layer)

    assert layer.weight.CB is None
    assert layer.weight.SCB is None
    assert layer.state.CB.data_ptr() == layer.weight.data_ptr()
    assert layer.state.SCB is scales
    torch.testing.assert_close(layer.weight, original)


def test_invalid_partial_scales_do_not_mutate_metadata():
    layer = _layer()
    layer.state.CB = layer.weight.detach()
    layer.weight.CB = layer.state.CB
    layer.state.SCB = torch.tensor([1.0, float("nan")])

    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        validate_bnb_int8_weight(layer)

    assert layer.weight.SCB is None


def test_transformed_storage_is_not_used_as_row_major_fallback():
    layer = _layer()
    layer.state.SCB = torch.ones(2)
    layer.state.formatB = "col32"

    with pytest.raises(RuntimeError, match="requires row-major"):
        dequantize_bnb_int8_weight(layer)


def test_complete_weight_pair_takes_precedence_over_stale_state():
    layer = _layer()
    layer.weight.CB = layer.weight.detach()
    layer.weight.SCB = torch.tensor([2.0, 4.0])
    layer.state.CB = torch.zeros_like(layer.weight)
    layer.state.SCB = torch.tensor([20.0, 40.0])

    result = dequantize_bnb_int8_weight(layer)

    assert result[0, 0] == 2.0


def test_raw_int8_without_any_metadata_never_falls_through_to_float_cast():
    layer = _layer()

    assert is_bnb_int8_weight(layer)
    with pytest.raises(RuntimeError, match="missing a coherent CB/SCB"):
        dequantize_bnb_int8_weight(layer)


def test_floating_master_weights_override_cached_quantization():
    layer = _layer()
    layer.weight = nn.Parameter(torch.full((2, 3), 3.0, dtype=torch.float16))
    layer.weight.has_fp16_weights = True
    layer.state.has_fp16_weights = True
    layer.state.CB = torch.ones((2, 3), dtype=torch.int8)
    layer.state.SCB = torch.ones(2)

    assert not is_bnb_int8_weight(layer)
    validate_bnb_int8_weight(layer)
    torch.testing.assert_close(
        dequantize_bnb_int8_weight(layer), torch.full((2, 3), 3.0)
    )


def test_plain_floating_layer_is_not_classified_as_int8():
    assert not is_bnb_int8_weight(nn.Linear(3, 2))


def test_nested_peft_wrappers_use_base_layer_state():
    layer = _layer()
    layer.state.CB = layer.weight.detach()
    layer.state.SCB = torch.tensor([2.0, 4.0])
    wrapped = SimpleNamespace(base_layer=SimpleNamespace(base_layer=layer))

    assert is_bnb_int8_weight(wrapped)
    torch.testing.assert_close(
        dequantize_bnb_int8_weight(wrapped), dequantize_bnb_int8_weight(layer)
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("CB", torch.zeros(2, 3), "2-D int8"),
        ("CB", torch.zeros(6, dtype=torch.int8), "2-D int8"),
        ("CB", torch.zeros(3, 2, dtype=torch.int8), "shape differs"),
        ("SCB", torch.ones(2, 1), "one scale per row"),
        ("SCB", torch.ones(3), "one scale per row"),
        ("SCB", torch.ones(2, dtype=torch.int32), "floating point"),
        ("SCB", torch.tensor([1.0, float("nan")]), "finite and nonnegative"),
        ("SCB", torch.tensor([1.0, float("inf")]), "finite and nonnegative"),
        ("SCB", torch.tensor([1.0, -1.0]), "finite and nonnegative"),
        (
            "SCB",
            torch.tensor([1.0, 1e100], dtype=torch.float64),
            "finite and nonnegative",
        ),
        ("CB", torch.empty(2, 3, dtype=torch.int8, device="meta"), "on meta"),
        ("SCB", torch.empty(2, device="meta"), "on meta"),
    ],
)
def test_invalid_quantization_state_fails_before_projection(field, value, message):
    layer = _layer()
    layer.state.CB = layer.weight.detach()
    layer.state.SCB = torch.ones(2)
    setattr(layer.state, field, value)

    with pytest.raises(RuntimeError, match=message):
        validate_bnb_int8_weight(layer)
    with pytest.raises(RuntimeError, match=message):
        dequantize_bnb_int8_weight(layer)


def test_zero_row_scale_produces_zero_row():
    layer = _layer()
    layer.state.SCB = torch.tensor([0.0, 4.0])

    assert torch.count_nonzero(dequantize_bnb_int8_weight(layer)[0]) == 0
