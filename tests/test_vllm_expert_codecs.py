"""vLLM's Inkling 2 experts in the release formats: a trial edits the weights
vLLM holds (rows shuffled, scales swizzled, experts split across ranks) into
what the bake writes for the same plan."""

import pytest
import torch
import torch.nn as nn

from abliterix.core import expert_formats as xf
from abliterix.core import fp4_repack as rp
from abliterix.core import vllm_expert_codecs as vc
from abliterix.core.fp4_repack import TensorEdit
from abliterix.core.vllm_plan_editor import (
    _worker_apply_plan,
    _worker_restore_plan,
    encode_plan,
)
from abliterix.weight_transforms import conv_span_basis

from test_vllm_plan_editor import _mod, _run, _top, _workers

H, I, E, WORLD = 128, 128, 4, 2  # swizzled scales need 128 rows and 4-column tiles
PREFIX = "model.language_model.layers"


def test_layout_functions_invert():
    x = torch.randint(0, 255, (3, 256, 16), dtype=torch.uint8)
    assert torch.equal(vc.unshuffle_rows(vc.shuffle_rows(x)), x)
    assert torch.equal(vc.unswizzle_block_scales(vc.swizzle_block_scales(x)), x)
    assert not torch.equal(vc.shuffle_rows(x), x)
    assert not torch.equal(vc.swizzle_block_scales(x), x)
    # As vLLM's code does it (its docstring states the inverse): row 4 * a + b
    # of a block of 32 lands at 8 * b + a.
    rows = torch.arange(64).reshape(1, 64, 1)
    assert vc.shuffle_rows(rows)[0, 8 * 3 + 1, 0] == 4 * 1 + 3


def _release(fmt, g):
    W = torch.randn(E, H, I, generator=g) * 0.05
    if fmt == "nvfp4":
        packed, scale, amax = xf.quantize_nvfp4_amax(W, 2688.0)
        return {"": packed, "_scale": scale, "_amax": amax,
                "_outer_scale_denominator": torch.tensor([2688.0])}
    weight, scale = xf.quantize_mxfp8(W)
    return {"": weight, "_scale": scale}


def _vllm_experts(stored, rank):
    """One rank's ``Inkling2Experts`` as vLLM loads it."""
    n = E // WORLD
    sl = slice(rank * n, (rank + 1) * n)
    params = {
        "w2_weight": vc.shuffle_rows(stored[""][sl]),
        "w2_weight_scale": vc.swizzle_block_scales(vc.shuffle_rows(stored["_scale"][sl])),
    }
    if "_amax" in stored:
        params["w2_weight_amax"] = stored["_amax"][sl]
        params["w2_weight_outer_scale_denominator"] = stored["_outer_scale_denominator"]
    experts = _mod(**params)
    experts.ep_rank, experts.num_experts, experts.ep_size = rank, n, WORLD
    return experts


def _whole(tops, layer):
    """The release-format tensors back from every rank's vLLM layout."""
    parts = [top.model.layers[layer].mlp.experts for top in tops]
    out = {
        "": torch.cat([vc.unshuffle_rows(p.w2_weight.data) for p in parts]),
        "_scale": torch.cat([vc.unshuffle_rows(vc.unswizzle_block_scales(p.w2_weight_scale.data))
                             for p in parts]),
    }
    if hasattr(parts[0], "w2_weight_amax"):
        out["_amax"] = torch.cat([p.w2_weight_amax.data for p in parts])
        out["_outer_scale_denominator"] = parts[0].w2_weight_outer_scale_denominator.data
    return out


def _bytes(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t


@pytest.mark.parametrize("chunk", [16, 1])
def test_a_trial_writes_what_the_bake_writes(chunk):
    g = torch.Generator().manual_seed(0)
    formats = ["nvfp4", "mxfp8"]  # layer 0 NVFP4, layer 1 MXFP8, as layers 0-39 and 40-47
    release = [_release(f, g) for f in formats]
    tops = [_top([_mod(mlp=_mod(experts=_vllm_experts(s, r))) for s in release]) for r in range(WORLD)]
    for layer, stored in enumerate(release):  # the fakes give back the release
        for k, v in _whole(tops, layer).items():
            assert torch.equal(_bytes(v), _bytes(stored[k])), k

    taps = 0.13 + 0.6 * torch.randn(H, 4, generator=g)
    edits = []
    for layer in range(len(formats)):
        d = torch.nn.functional.normalize(torch.randn(H, generator=g), dim=0)
        name = f"{PREFIX}.{layer}.mlp.experts.down_proj"
        edits.append(TensorEdit(name, "ega", d, 1.3, True, hidden_dim=H, basis=conv_span_basis(d, taps)))
        edits.append(TensorEdit(name, "expert_project", d, 0.7, False, hidden_dim=H, expert=3))
    shapes = {e.logical_name: (E, H, I) for e in edits}
    workers, _ = _workers(tops)
    results = _run(workers, _worker_apply_plan, encode_plan(edits, shapes, chunk))
    assert results[0]["tensors"] == 2

    for layer, (fmt, stored) in enumerate(zip(formats, release)):
        codec = xf.NVFP4_AMAX if fmt == "nvfp4" else xf.MXFP8
        keys = tuple("w" + k for k in stored)
        t = {"w" + k: v for k, v in stored.items()}
        group = [e for e in edits if e.logical_name.startswith(f"{PREFIX}.{layer}.")]
        want = codec.encode(rp.apply_tensor_edits(codec.decode(t, keys), group), t, keys)
        got = _whole(tops, layer)
        for k in stored:
            assert torch.equal(_bytes(got[k]), _bytes(want["w" + k])), (fmt, k)
        assert not torch.equal(_bytes(got[""]), _bytes(stored[""]))

    # Weight, scales and amax of the NVFP4 layer; weight and scales of the MXFP8 one.
    assert _run(workers, _worker_restore_plan)[0] == 5
    for layer, stored in enumerate(release):
        for k, v in _whole(tops, layer).items():
            assert torch.equal(_bytes(v), _bytes(stored[k])), k


def test_other_quantized_experts_get_no_inkling_codec():
    # ModelOpt-style NVFP4 FusedMoE: e4m3 scales but no per-expert amax.
    experts = _mod(w2_weight=torch.zeros(E, H, I // 2, dtype=torch.uint8),
                   w2_weight_scale=torch.zeros(E, H, I // 16).to(torch.float8_e4m3fn))
    tgt = type("T", (), {"owner": experts, "param": experts.w2_weight, "transposed": False})()
    assert vc.inkling_expert_codec(tgt) is None
    # MXFP4 (gpt-oss): packed uint8 with uint8 exponents.
    experts = _mod(w2_weight=torch.zeros(E, H, I // 2, dtype=torch.uint8),
                   w2_weight_scale=torch.zeros(E, H, I // 32, dtype=torch.uint8))
    tgt = type("T", (), {"owner": experts, "param": experts.w2_weight, "transposed": False})()
    assert vc.inkling_expert_codec(tgt) is None
