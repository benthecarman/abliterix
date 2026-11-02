"""Numerical and storage contracts for the full-vocabulary KL reducer."""

import math

import pytest
import torch
import torch.nn.functional as F

from abliterix.eval.kl import mean_token_kl


@pytest.mark.parametrize("shape", [(11,), (7, 11), (7, 3, 11)])
@pytest.mark.parametrize("chunk_elements", [1, 22, 100_000])
def test_matches_independent_probability_reference_with_uneven_chunks(
    shape, chunk_elements
):
    generator = torch.Generator().manual_seed(9)
    current = torch.randn(shape, generator=generator)
    baseline = torch.randn(shape, generator=generator)
    p = baseline.double().softmax(dim=-1)
    q = current.double().softmax(dim=-1)
    expected = (p * (p.log() - q.log())).sum(dim=-1).mean().item()

    actual = mean_token_kl(current, baseline, max_chunk_elements=chunk_elements)

    assert actual == pytest.approx(expected, abs=5e-7, rel=5e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mixed_input_dtypes_are_normalized_in_fp32(dtype):
    current = torch.tensor([[3.3, -0.7, 1.1]], dtype=dtype)
    baseline = torch.tensor([[1.2, -2.1, 0.3]], dtype=torch.float32)
    expected = F.kl_div(
        F.log_softmax(current.float(), dim=-1),
        F.log_softmax(baseline, dim=-1),
        log_target=True,
        reduction="sum",
    ).item()
    assert mean_token_kl(current, baseline) == pytest.approx(expected, abs=1e-7)


def test_orientation_is_baseline_relative_to_current():
    baseline = torch.tensor([0.8, 0.2]).log()
    current = torch.tensor([0.3, 0.7]).log()
    expected = 0.8 * math.log(0.8 / 0.3) + 0.2 * math.log(0.2 / 0.7)
    assert mean_token_kl(current, baseline) == pytest.approx(expected)
    assert mean_token_kl(baseline, current) != pytest.approx(expected)


def test_repeating_token_positions_does_not_multiply_mean_kl():
    current = torch.tensor([[2.0, 0.0], [-1.0, 1.0]])
    baseline = torch.tensor([[1.0, -1.0], [0.0, 0.0]])
    assert mean_token_kl(current, baseline) == pytest.approx(
        mean_token_kl(
            current[:, None, :].expand(-1, 5, -1),
            baseline[:, None, :].expand(-1, 5, -1),
            max_chunk_elements=6,
        )
    )


def test_transposed_inputs_are_reduced_in_bounded_chunks_without_mutation(monkeypatch):
    generator = torch.Generator().manual_seed(11)
    current = torch.randn(4, 7, 13, generator=generator).transpose(0, 1)
    baseline = torch.randn(7, 4, 13, generator=generator)
    current_before, baseline_before = current.clone(), baseline.clone()
    expected = mean_token_kl(current.contiguous(), baseline)
    chunk_sizes = []
    original = F.kl_div

    def track(input, target, **kwargs):
        chunk_sizes.append(input.numel())
        return original(input, target, **kwargs)

    monkeypatch.setattr(F, "kl_div", track)
    actual = mean_token_kl(current, baseline, max_chunk_elements=26)

    assert actual == pytest.approx(expected, abs=1e-7)
    assert len(chunk_sizes) > 1
    assert max(chunk_sizes) <= 26
    torch.testing.assert_close(current, current_before)
    torch.testing.assert_close(baseline, baseline_before)


def test_reduction_never_tracks_gradients(monkeypatch):
    original = F.log_softmax

    def track(*args, **kwargs):
        assert not torch.is_grad_enabled()
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "log_softmax", track)
    assert (
        mean_token_kl(
            torch.zeros(2, 3, requires_grad=True), torch.zeros(2, 3, requires_grad=True)
        )
        == 0
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["current", "baseline", "both"])
def test_invalid_numerical_outputs_cannot_appear_as_finite_damage(invalid, side):
    current, baseline = torch.zeros(2, 3), torch.zeros(2, 3)
    if side in {"current", "both"}:
        current[1, 0] = invalid
    if side in {"baseline", "both"}:
        baseline[1, 0] = invalid
    assert mean_token_kl(current, baseline, max_chunk_elements=3) == float("inf")


@pytest.mark.parametrize("side", ["current", "baseline"])
def test_all_zero_probability_row_is_invalid(side):
    current, baseline = torch.zeros(2, 3), torch.zeros(2, 3)
    (current if side == "current" else baseline)[1] = -float("inf")
    assert mean_token_kl(current, baseline) == float("inf")


def test_zero_probability_limits_are_not_replaced_with_artificial_mass():
    p = torch.tensor([1.0, 0.0]).log()
    assert mean_token_kl(p, p) == 0.0
    assert mean_token_kl(torch.tensor([0.5, 0.5]).log(), p) == pytest.approx(
        math.log(2)
    )
    assert mean_token_kl(p, torch.tensor([0.5, 0.5]).log()) == float("inf")


@pytest.mark.parametrize("shape", [(), (0, 3), (2, 0)])
def test_empty_or_scalar_inputs_are_not_valid_distributions(shape):
    with pytest.raises(ValueError, match="non-empty"):
        mean_token_kl(torch.empty(shape), torch.empty(shape))


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_chunk_budget_is_rejected(size):
    with pytest.raises(ValueError, match="positive integer"):
        mean_token_kl(torch.zeros(1, 2), torch.zeros(1, 2), max_chunk_elements=size)


def test_integer_inputs_are_rejected():
    with pytest.raises(ValueError, match="floating-point"):
        mean_token_kl(torch.zeros(2, dtype=torch.long), torch.zeros(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_device_inputs_match_cpu_reference():
    current = torch.randn(7, 13)
    baseline = torch.randn(7, 13)
    assert mean_token_kl(
        current.cuda(), baseline, max_chunk_elements=26
    ) == pytest.approx(mean_token_kl(current, baseline), abs=1e-6)
