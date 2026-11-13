"""The vLLM in-place plan path (core/vllm_plan_editor.py) applies a trial's
steering plan to the workers' weights with the export's math.

No vLLM: fake workers expose vLLM's module layouts (gpt-oss, Gemma 4 with a
MoE runner, Inkling 2 with SkyRL's expert-parallel BF16 experts, a FusedMoE
with a round-robin expert map), split across two TP ranks run as threads with
an all-gather between them. What the ranks hold after the plan, put back
together, must equal ``apply_tensor_edits`` on the whole HF tensors, the
export's own replay.
"""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from abliterix.core import fp4_repack as rp
from abliterix.core.engine import SteeringEngine
from abliterix.core.fp4_repack import (
    TensorEdit,
    _ulps_apart,
    apply_tensor_edits,
    group_edits,
)
from abliterix.core.steering import _apply_direct_steering, _apply_ega_steering
from abliterix.core.vllm_plan_editor import (
    _worker_apply_plan,
    _worker_restore_plan,
    encode_plan,
    hf_shapes,
)
from abliterix.types import DecayKernel, SteeringProfile, WeightNorm
from abliterix.weight_transforms import conv_span_basis

H, A, KV, E, INTER, F, K = 16, 24, 8, 4, 8, 12, 4
WORLD = 2
BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Two TP ranks as threads
# ---------------------------------------------------------------------------


class _Hub:
    def __init__(self, world):
        self.slots = [None] * world
        self.barrier = threading.Barrier(world, timeout=20)
        self.gathers = 0


class _ThreadComm:
    def __init__(self, hub, rank, world):
        self.hub, self.rank, self.world = hub, rank, world

    def all_gather(self, t, dim):
        self.hub.slots[self.rank] = t.clone()
        self.hub.barrier.wait()
        out = torch.cat(self.hub.slots, dim)
        if self.rank == 0:
            self.hub.gathers += 1
        self.hub.barrier.wait()
        return out


def _workers(tops):
    hub = _Hub(len(tops))
    return [
        SimpleNamespace(
            model_runner=SimpleNamespace(model=top),
            _abliterix_comm=_ThreadComm(hub, r, len(tops)),
        )
        for r, top in enumerate(tops)
    ], hub


def _run(workers, fn, *args):
    results, errors = [None] * len(workers), []

    def go(r):
        try:
            results[r] = fn(workers[r], *args)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            workers[r]._abliterix_comm.hub.barrier.abort()

    threads = [threading.Thread(target=go, args=(r,)) for r in range(len(workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    if errors:
        raise errors[0]
    return results


# ---------------------------------------------------------------------------
# Fake vLLM models
# ---------------------------------------------------------------------------


def _mod(**params):
    m = nn.Module()
    for k, v in params.items():
        if isinstance(v, nn.Module):
            setattr(m, k, v)
        else:
            m.register_parameter(k, nn.Parameter(v.clone(), requires_grad=False))
    return m


def _top(layers):
    top = nn.Module()
    top.model = nn.Module()
    top.model.layers = nn.ModuleList(layers)
    return top


def _cols(t, r):
    n = t.shape[-1] // WORLD
    return t[..., r * n : (r + 1) * n]


def _rows(t, r):
    n = t.shape[0] // WORLD
    return t[r * n : (r + 1) * n]


class GptOss:
    """HF ``model.layers.L``: q/k/v/o_proj, ``mlp.router``, ``mlp.experts.down_proj``
    stored [E, INTER, H] (transposed). vLLM: fused qkv_proj split by heads, row-parallel
    o_proj, replicated router, ``w2_weight`` [E, H, INTER/TP]."""

    prefix = "model.layers"
    transposed = True

    def hf(self, g):
        return {
            "self_attn.q_proj.weight": torch.randn(A, H, generator=g),
            "self_attn.k_proj.weight": torch.randn(KV, H, generator=g),
            "self_attn.v_proj.weight": torch.randn(KV, H, generator=g),
            "self_attn.o_proj.weight": torch.randn(H, A, generator=g),
            "mlp.router.weight": torch.randn(E, H, generator=g),
            "mlp.experts.down_proj": torch.randn(E, INTER, H, generator=g),
        }

    def layer(self, w, r):
        attn = _mod(
            qkv_proj=_mod(
                weight=torch.cat(
                    [
                        _rows(w["self_attn.q_proj.weight"], r),
                        _rows(w["self_attn.k_proj.weight"], r),
                        _rows(w["self_attn.v_proj.weight"], r),
                    ]
                )
            ),
            o_proj=_mod(weight=_cols(w["self_attn.o_proj.weight"], r)),
        )
        attn.q_size, attn.kv_size = A // WORLD, KV // WORLD
        experts = _mod(w2_weight=_cols(w["mlp.experts.down_proj"].transpose(1, 2), r))
        return _mod(
            self_attn=attn,
            mlp=_mod(router=_mod(weight=w["mlp.router.weight"]), experts=experts),
        )

    def whole(self, layers, key):
        if key == "self_attn.o_proj.weight":
            return torch.cat([layer.self_attn.o_proj.weight for layer in layers], 1)
        if key.startswith("self_attn."):
            lo, hi = {
                "q": (0, A // WORLD),
                "k": (A // WORLD, (A + KV) // WORLD),
                "v": ((A + KV) // WORLD, (A + 2 * KV) // WORLD),
            }[key[10]]
            return torch.cat(
                [layer.self_attn.qkv_proj.weight[lo:hi] for layer in layers]
            )
        if key == "mlp.router.weight":
            return layers[0].mlp.router.weight
        return torch.cat(
            [layer.mlp.experts.w2_weight for layer in layers], 2
        ).transpose(1, 2)


class Gemma:
    """HF ``model.language_model.layers.L``: o_proj, a dense ``mlp.down_proj``,
    ``router.proj`` and ``experts.down_proj`` [E, H, INTER] beside the MLP. vLLM: the
    experts behind a MoE runner (``experts.routed_experts.w2_weight``)."""

    prefix = "model.language_model.layers"
    transposed = False

    def hf(self, g):
        return {
            "self_attn.o_proj.weight": torch.randn(H, A, generator=g),
            "mlp.down_proj.weight": torch.randn(H, F, generator=g),
            "router.proj.weight": torch.randn(E, H, generator=g),
            "experts.down_proj": torch.randn(E, H, INTER, generator=g),
        }

    def experts(self, w, r):
        routed = _mod(w2_weight=_cols(w["experts.down_proj"], r))
        routed.global_num_experts = E
        return _mod(routed_experts=routed)

    def layer(self, w, r):
        return _mod(
            self_attn=_mod(o_proj=_mod(weight=_cols(w["self_attn.o_proj.weight"], r))),
            mlp=_mod(down_proj=_mod(weight=_cols(w["mlp.down_proj.weight"], r))),
            router=_mod(proj=_mod(weight=w["router.proj.weight"])),
            experts=self.experts(w, r),
        )

    def whole_experts(self, layers):
        return torch.cat(
            [layer.experts.routed_experts.w2_weight for layer in layers], 2
        )

    def whole(self, layers, key):
        if key == "experts.down_proj":
            return self.whole_experts(layers)
        if key == "router.proj.weight":
            return layers[0].router.proj.weight
        outer, inner = (
            ("self_attn", "o_proj")
            if key.startswith("self_attn")
            else ("mlp", "down_proj")
        )
        return torch.cat(
            [getattr(getattr(layer, outer), inner).weight for layer in layers], 1
        )


class GemmaExpertMap(Gemma):
    """A FusedMoE under expert parallelism, experts dealt round robin: rank r
    holds experts r, r + 2, ... in its ``_expert_map``."""

    def experts(self, w, r):
        mine = list(range(r, E, WORLD))
        routed = _mod(w2_weight=w["experts.down_proj"][mine])
        expert_map = torch.full((E,), -1, dtype=torch.int32)
        expert_map[mine] = torch.arange(len(mine), dtype=torch.int32)
        routed.register_buffer("_expert_map", expert_map, persistent=False)
        routed.global_num_experts = E
        return _mod(routed_experts=routed)

    def whole_experts(self, layers):
        out = torch.empty(
            E, H, INTER, dtype=layers[0].experts.routed_experts.w2_weight.dtype
        )
        for r, layer in enumerate(layers):
            out[list(range(r, E, WORLD))] = layer.experts.routed_experts.w2_weight
        return out


class Inkling:
    """HF ``model.language_model.layers.L``: ``self_attn.o_proj``, an FP32
    ``mlp.gate`` and ``mlp.experts.down_proj`` [E, H, INTER]. vLLM: ``attn.wo_ud``
    row-parallel, SkyRL's BF16 experts split by expert (``ep_rank``,
    ``num_experts`` per rank), whole intermediate."""

    prefix = "model.language_model.layers"
    transposed = False

    def hf(self, g):
        return {
            "self_attn.o_proj.weight": torch.randn(H, A, generator=g),
            "mlp.gate.weight": torch.randn(E, H, generator=g).float(),
            "mlp.experts.down_proj": torch.randn(E, H, INTER, generator=g),
        }

    def layer(self, w, r):
        experts = _mod(w2_weight=_rows(w["mlp.experts.down_proj"], r))
        experts.ep_rank, experts.num_experts, experts.ep_size = r, E // WORLD, WORLD
        return _mod(
            attn=_mod(wo_ud=_mod(weight=_cols(w["self_attn.o_proj.weight"], r))),
            mlp=_mod(gate=_mod(weight=w["mlp.gate.weight"]), experts=experts),
        )

    def whole(self, layers, key):
        if key == "self_attn.o_proj.weight":
            return torch.cat([layer.attn.wo_ud.weight for layer in layers], 1)
        if key == "mlp.gate.weight":
            return layers[0].mlp.gate.weight
        return torch.cat([layer.mlp.experts.w2_weight for layer in layers])


LAYOUTS = {
    "gpt_oss": GptOss(),
    "gemma": Gemma(),
    "gemma_expert_map": GemmaExpertMap(),
    "inkling": Inkling(),
}
N_LAYERS = 2


def _hf_weights(layout, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for L in range(N_LAYERS):
        for key, t in layout.hf(g).items():
            out[f"{layout.prefix}.{L}.{key}"] = (
                t if t.dtype == torch.float32 and "gate" in key else t.to(BF16)
            )
    return out


def _vllm_ranks(layout, hf):
    per_layer = [
        {
            k.split(f".{L}.", 1)[1]: v
            for k, v in hf.items()
            if k.startswith(f"{layout.prefix}.{L}.")
        }
        for L in range(N_LAYERS)
    ]
    return [_top([layout.layer(w, r) for w in per_layer]) for r in range(WORLD)]


def _whole(layout, tops, name):
    L, key = name[len(layout.prefix) + 1 :].split(".", 1)
    return layout.whole([top.model.layers[int(L)] for top in tops], key)


def _taps(seed):
    g = torch.Generator().manual_seed(seed)
    mean = torch.tensor([0.01, 0.02, 0.13, 0.25])
    std = torch.tensor([0.2, 0.3, 0.6, 1.07])
    return mean + std * torch.randn(H, K, generator=g)


def _plan(layout, hf):
    """Every edit kind the export records, on every tensor of the layout."""
    g = torch.Generator().manual_seed(7)
    edits = []
    for name, t in hf.items():
        d = torch.nn.functional.normalize(torch.randn(H, generator=g), dim=0)
        if t.dim() == 3:
            basis = None if layout.transposed else conv_span_basis(d, _taps(len(edits)))
            edits.append(
                TensorEdit(
                    name,
                    "ega",
                    d,
                    1.3,
                    True,
                    hidden_dim=H,
                    transposed=layout.transposed,
                    basis=basis,
                )
            )
            for expert in (1, 2):
                edits.append(
                    TensorEdit(
                        name,
                        "expert_project",
                        d,
                        0.7,
                        False,
                        hidden_dim=H,
                        transposed=layout.transposed,
                        expert=expert,
                    )
                )
        elif (
            name.endswith(("gate.weight", "router.weight", "proj.weight"))
            and t.shape[0] == E
        ):
            edits.append(
                TensorEdit(name, "row_scale", None, 0.0, False, rows=[0, 3], scale=0.4)
            )
        elif t.shape[0] == H:  # writes the stream: output side
            basis = conv_span_basis(d, _taps(len(edits))) if "o_proj" in name else None
            edits.append(
                TensorEdit(
                    name, "direct", d, 0.9, True, projection_side="output", basis=basis
                )
            )
        else:  # reads the stream: input side
            edits.append(
                TensorEdit(name, "direct", d, 0.8, True, projection_side="input")
            )
    return edits


@pytest.mark.parametrize("chunk", [16, 1])
@pytest.mark.parametrize("layout_name", list(LAYOUTS))
def test_workers_apply_what_the_export_replays(layout_name, chunk):
    layout = LAYOUTS[layout_name]
    hf = _hf_weights(layout)
    edits = _plan(layout, hf)
    tops = _vllm_ranks(layout, hf)
    for name, t in hf.items():  # the fakes reassemble to the HF tensors
        assert torch.equal(_whole(layout, tops, name), t), name
    workers, hub = _workers(tops)

    results = _run(
        workers,
        _worker_apply_plan,
        encode_plan(edits, {n: t.shape for n, t in hf.items()}, chunk),
    )
    assert len({r["tensors"] for r in results}) == 1

    for name, group in group_edits(edits).items():
        want = apply_tensor_edits(hf[name], group, round_between=hf[name].dtype).to(
            hf[name].dtype
        )
        got = _whole(layout, tops, name)
        assert got.dtype == want.dtype
        assert _ulps_apart(want, got) <= 1.0, name
        assert not torch.equal(got, hf[name]), name
    assert hub.gathers > 0 or layout_name in ("gemma_expert_map",)

    restored = _run(workers, _worker_restore_plan)
    assert restored[0] > 0
    for name, t in hf.items():
        assert torch.equal(_whole(layout, tops, name), t), name
    assert _run(workers, _worker_restore_plan) == [0, 0]  # nothing left to restore


def test_a_second_trial_starts_from_the_pristine_weights():
    layout = LAYOUTS["inkling"]
    hf = _hf_weights(layout)
    edits = _plan(layout, hf)
    tops = _vllm_ranks(layout, hf)
    workers, _ = _workers(tops)
    payload = encode_plan(edits, {n: t.shape for n, t in hf.items()})
    _run(workers, _worker_apply_plan, payload)
    first = {n: _whole(layout, tops, n).clone() for n in hf}
    _run(workers, _worker_restore_plan)
    _run(workers, _worker_apply_plan, payload)
    for n in hf:
        assert torch.equal(_whole(layout, tops, n), first[n]), n


def test_an_unknown_tensor_fails_on_every_rank_before_any_edit():
    layout = LAYOUTS["gemma"]
    hf = _hf_weights(layout)
    tops = _vllm_ranks(layout, hf)
    workers, hub = _workers(tops)
    name = f"{layout.prefix}.0.self_attn.o_proj.weight"
    bogus = f"{layout.prefix}.1.mlp.nonexistent.weight"
    d = torch.nn.functional.normalize(torch.randn(H), dim=0)
    edits = [
        TensorEdit(name, "direct", d, 1.0, True, projection_side="output"),
        TensorEdit(bogus, "direct", d, 1.0, True, projection_side="output"),
    ]
    with pytest.raises(ValueError, match="no counterpart"):
        _run(
            workers,
            _worker_apply_plan,
            encode_plan(edits, {name: (H, A), bogus: (H, A)}),
        )
    assert hub.gathers == 0
    assert torch.equal(_whole(layout, tops, name), hf[name])


def test_quantized_storage_without_a_codec_is_refused():
    layout = LAYOUTS["inkling"]
    hf = _hf_weights(layout)
    tops = _vllm_ranks(layout, hf)
    for top in tops:
        w = top.model.layers[0].attn.wo_ud
        w.weight = nn.Parameter(
            torch.zeros(H, A // WORLD, dtype=torch.uint8), requires_grad=False
        )
    workers, _ = _workers(tops)
    name = f"{layout.prefix}.0.self_attn.o_proj.weight"
    d = torch.nn.functional.normalize(torch.randn(H), dim=0)
    with pytest.raises(ValueError, match="no codec"):
        _run(
            workers,
            _worker_apply_plan,
            encode_plan(
                [TensorEdit(name, "direct", d, 1.0, True, projection_side="output")],
                {name: (H, A)},
            ),
        )


# ---------------------------------------------------------------------------
# From the HF engine: a plan recorded on the skeleton, applied by the workers,
# equals the engine's own edited weights
# ---------------------------------------------------------------------------


class _Conv(nn.Module):
    def __init__(self, taps):
        super().__init__()
        self.conv1d = nn.Conv1d(H, H, K, groups=H, bias=False)
        self.conv1d.weight.data.copy_(taps.unsqueeze(1))


def _hf_engine(hf):
    """An HF-like Inkling model holding ``hf``'s weights, with the engine surface
    plan recording and the steering functions use."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    layers = []
    for L in range(N_LAYERS):
        p = f"model.language_model.layers.{L}."
        o_proj = nn.Linear(A, H, bias=False, dtype=BF16)
        o_proj.weight.data.copy_(hf[p + "self_attn.o_proj.weight"])
        experts = _mod(down_proj=hf[p + "mlp.experts.down_proj"])
        gate = nn.Linear(H, E, bias=False)
        gate.weight.data.copy_(hf[p + "mlp.gate.weight"])
        layer = _mod(
            self_attn=_mod(o_proj=o_proj), mlp=_mod(experts=experts, gate=gate)
        )
        layer.post_attention_convolution = _Conv(_taps(L))
        layer.post_mlp_convolution = _Conv(_taps(L + 100))
        layers.append(layer)
    model.model.language_model.layers = nn.ModuleList(layers)

    def output_conv_taps(i, component):
        conv = {
            "attn.o_proj": layers[i].post_attention_convolution,
            "mlp.down_proj": layers[i].post_mlp_convolution,
        }.get(component)
        return None if conv is None else conv.conv1d.weight[:, 0, :].detach()

    return SimpleNamespace(
        model=model,
        transformer_layers=layers,
        steerable_modules=lambda i: {"attn.o_proj": [layers[i].self_attn.o_proj]},
        _locate_fused_weights=lambda layer: layer.mlp.experts.down_proj,
        _locate_router=lambda layer: layer.mlp.gate,
        has_expert_routing=lambda: True,
        _fused_down_proj_transposed=False,
        output_conv_taps=output_conv_taps,
    )


def _config():
    return SimpleNamespace(
        steering=SimpleNamespace(
            decay_kernel=DecayKernel.GAUSSIAN,
            weight_normalization=WeightNorm.PRE,
            direct_transform="standard",
            direct_transform_preserve_row_norm=True,
            conv_span_projection=True,
            discriminative_layer_selection=False,
        )
    )


PROFILES = {
    "attn.o_proj": SteeringProfile(2.0, 1.0, 0.5, 3.0),
    "mlp.down_proj": SteeringProfile(7.0, 1.0, 6.0, 3.0),
}


def test_engine_skeleton_and_workers_make_the_same_model():
    layout = LAYOUTS["inkling"]
    hf = _hf_weights(layout, seed=3)
    svs = torch.randn(N_LAYERS + 1, H)
    config = _config()

    live = _hf_engine(hf)
    live_plan = rp.record_steering_plan(live, svs, None, PROFILES, config, None)
    _apply_direct_steering(live, svs, None, PROFILES, config, None)
    _apply_ega_steering(live, svs, None, PROFILES, config, None)
    engine_weights = {
        rp.normalize_param_name(n): p.detach().clone()
        for n, p in live.model.named_parameters()
    }

    fresh = _hf_engine(hf)
    skeleton = SteeringEngine.plan_skeleton(fresh)
    assert all(p.is_meta for p in skeleton.model.parameters())
    plan = rp.record_steering_plan(skeleton, svs, None, PROFILES, config, None)
    assert {e.kind for e in plan} == {"direct", "ega"} and all(
        e.basis is not None for e in plan
    )
    assert len(plan) == len(live_plan)
    for a, b in zip(plan, live_plan):
        assert a.logical_name == b.logical_name and torch.equal(a.basis, b.basis)
        assert torch.equal(a.direction, b.direction) and a.strength == b.strength

    tops = _vllm_ranks(layout, hf)
    workers, _ = _workers(tops)
    _run(workers, _worker_apply_plan, encode_plan(plan, hf_shapes(skeleton)))
    for name in group_edits(plan):
        assert _ulps_apart(engine_weights[name], _whole(layout, tops, name)) <= 1.0, (
            name
        )


def test_dense_codec_rounds_like_the_engine_between_edits():
    """Two edits on one bf16 tensor: the engine stores the first result in bf16
    before the second; so does the dense codec."""
    layout = LAYOUTS["inkling"]
    hf = _hf_weights(layout)
    name = f"{layout.prefix}.0.mlp.experts.down_proj"
    d = torch.nn.functional.normalize(torch.randn(H), dim=0)
    edits = [
        TensorEdit(name, "ega", d, 1.1, True, hidden_dim=H),
        TensorEdit(name, "expert_project", d, 0.5, False, hidden_dim=H, expert=0),
    ]
    tops = _vllm_ranks(layout, hf)
    workers, _ = _workers(tops)
    _run(workers, _worker_apply_plan, encode_plan(edits, {name: hf[name].shape}))
    want = apply_tensor_edits(hf[name], edits, round_between=BF16).to(BF16)
    assert torch.equal(_whole(layout, tops, name), want)
    assert math.isfinite(_ulps_apart(want, _whole(layout, tops, name)))
