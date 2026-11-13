"""The conv-span edit (``steering.conv_span_projection``) for writers whose
output passes through a per-channel short convolution (Inkling 2).

Synthetic tensors, no model: the basis, its exactness through the conv, and
that the engine, the recorded plan and its replay make the same weights.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from abliterix.core import fp4_repack as rp
from abliterix.core.steering import _apply_direct_steering, _apply_ega_steering
from abliterix.types import DecayKernel, SteeringProfile, WeightNorm
from abliterix.weight_transforms import apply_subspace_projection, conv_span_basis

H, K = 16, 4


def _taps(seed=0):
    g = torch.Generator().manual_seed(seed)
    mean = torch.tensor([0.01, 0.02, 0.13, 0.25])
    std = torch.tensor([0.2, 0.3, 0.6, 1.07])
    return mean + std * torch.randn(H, K, generator=g)


def _conv(x, taps):
    """Inkling 2's post-sublayer conv: (causal depthwise conv(x) + x) / sqrt(K + 1),
    taps oldest first. x is (seq, hidden)."""
    w = taps.unsqueeze(1)  # (H, 1, K)
    y = F.conv1d(x.T.unsqueeze(0), w, padding=K - 1, groups=H)[0, :, : x.shape[0]].T
    return (y + x) / math.sqrt(K + 1)


def test_basis_spans_the_direction_and_its_tap_images():
    r = torch.randn(H)
    taps = _taps()
    Q = conv_span_basis(r, taps)
    assert Q.shape == (H, K + 1)
    assert torch.allclose(Q.T @ Q, torch.eye(K + 1), atol=1e-5)
    for v in [r] + [taps[:, k] * r for k in range(K)]:
        residual = v - Q @ (Q.T @ v)
        assert residual.norm() < 1e-5 * v.norm()


def test_constant_taps_leave_rank_one():
    r = torch.randn(H)
    Q = conv_span_basis(r, torch.ones(H, K) * 0.3)
    assert Q.shape == (H, 1)


def test_direction_is_gone_after_the_conv_only_with_the_span():
    torch.manual_seed(0)
    r = torch.nn.functional.normalize(torch.randn(H), dim=0)
    taps = _taps()
    W = torch.randn(H, 32)
    x = torch.randn(10, 32)
    rank1 = W - torch.outer(r, r @ W)
    span = apply_subspace_projection(W, conv_span_basis(r, taps), strength=1.0, preserve_row_norm=False)
    base = (_conv(x @ W.T, taps) @ r).abs().mean()
    after1 = (_conv(x @ rank1.T, taps) @ r).abs().mean()
    afterk = (_conv(x @ span.T, taps) @ r).abs().mean()
    assert afterk < 1e-5 * base
    assert after1 > 0.1 * base  # rank 1 leaves much of it


def test_strength_scales_every_component_after_the_conv():
    torch.manual_seed(1)
    r = torch.nn.functional.normalize(torch.randn(H), dim=0)
    taps = _taps(1)
    W, x = torch.randn(H, 32), torch.randn(10, 32)
    edited = apply_subspace_projection(W, conv_span_basis(r, taps), strength=0.4, preserve_row_norm=False)
    before = _conv(x @ W.T, taps) @ r
    after = _conv(x @ edited.T, taps) @ r
    assert torch.allclose(after, 0.6 * before, atol=1e-4)


class _Conv(nn.Module):
    def __init__(self, taps):
        super().__init__()
        self.conv1d = nn.Conv1d(H, H, K, groups=H, bias=False)
        self.conv1d.weight.data.copy_(taps.unsqueeze(1))


class _Layer(nn.Module):
    def __init__(self, E, inter, seed):
        super().__init__()
        self.self_attn = SimpleNamespace()
        self.o_proj = nn.Linear(24, H, bias=False)
        self.down = nn.Parameter(torch.randn(E, H, inter), requires_grad=False)
        self.post_attention_convolution = _Conv(_taps(seed))
        self.post_mlp_convolution = _Conv(_taps(seed + 100))


class _Model(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(4, 8, i) for i in range(n)])


def _engine(n=3):
    model = _Model(n)
    layers = list(model.layers)

    def output_conv_taps(i, component):
        conv = {"attn.o_proj": layers[i].post_attention_convolution,
                "mlp.down_proj": layers[i].post_mlp_convolution}.get(component)
        return None if conv is None else conv.conv1d.weight[:, 0, :].detach()

    return SimpleNamespace(
        model=model,
        transformer_layers=layers,
        steerable_modules=lambda i: {"attn.o_proj": [layers[i].o_proj]},
        _locate_fused_weights=lambda layer: layer.down,
        has_expert_routing=lambda: True,
        _fused_down_proj_transposed=False,
        output_conv_taps=output_conv_taps,
    )


def _config(conv_span=True, norm=WeightNorm.PRE):
    return SimpleNamespace(steering=SimpleNamespace(
        decay_kernel=DecayKernel.GAUSSIAN, weight_normalization=norm, direct_transform="standard",
        direct_transform_preserve_row_norm=True, conv_span_projection=conv_span,
        discriminative_layer_selection=False))


PROFILES = {
    "attn.o_proj": SteeringProfile(2.0, 1.0, 0.5, 3.0),
    "mlp.down_proj": SteeringProfile(7.0, 1.0, 6.0, 3.0),
}


def test_engine_plan_and_replay_make_the_same_weights():
    torch.manual_seed(3)
    eng = _engine()
    svs = torch.randn(4, H)
    originals = {n: p.detach().clone() for n, p in eng.model.named_parameters() if "conv" not in n}
    config = _config()

    edits = rp.record_steering_plan(eng, svs, None, PROFILES, config, None)
    assert {e.kind for e in edits} == {"direct", "ega"}
    assert all(e.basis is not None and e.basis.shape == (H, K + 1) for e in edits)

    _apply_direct_steering(eng, svs, None, PROFILES, config, None)
    _apply_ega_steering(eng, svs, None, PROFILES, config, None)
    engine_weights = dict(eng.model.named_parameters())

    for name, group in rp.group_edits(edits).items():
        replayed = rp.apply_tensor_edits(originals[name], group)
        assert torch.equal(replayed, engine_weights[name].detach().to(torch.float32)), name


def test_without_the_option_the_edits_stay_rank_one():
    eng = _engine()
    edits = rp.record_steering_plan(eng, torch.randn(4, H), None, PROFILES, _config(conv_span=False), None)
    assert edits and all(e.basis is None for e in edits)


def test_plan_round_trip_keeps_the_basis(tmp_path):
    eng = _engine()
    edits = rp.record_steering_plan(eng, torch.randn(4, H), None, PROFILES, _config(), None)
    rp.save_plan(edits, tmp_path / "p.pt")
    loaded = rp.load_plan(tmp_path / "p.pt")
    assert all(torch.equal(a.basis, b.basis) for a, b in zip(edits, loaded))
    plain = rp.record_steering_plan(eng, torch.randn(4, H), None, PROFILES, _config(conv_span=False), None)
    rp.save_plan(plain, tmp_path / "q.pt")
    assert all("basis" not in e for e in torch.load(tmp_path / "q.pt", weights_only=False)["edits"])
