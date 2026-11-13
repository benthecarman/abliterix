"""The per-expert NVFP4 and MXFP8 codecs (``core.expert_formats``) and the bake
that uses them: Inkling 2's release layout, synthetic and, when it is on disk,
real."""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from abliterix.core import expert_formats as xf
from abliterix.core import fp4_repack as rp
from abliterix.weight_transforms import conv_span_basis

RELEASE = Path("/mnt/llm-models/inkling2-small-meta")
E, H, I = 8, 64, 96  # experts, hidden, intermediate: w2 is [E, H, I]


def _encoded(codec, W, name="w2_weight"):
    meta = {}
    if codec is xf.NVFP4_AMAX:
        packed, scale, amax = xf.quantize_nvfp4_amax(W, 6.0 * 448.0)
        t = {name: packed, f"{name}_scale": scale, f"{name}_amax": amax,
             f"{name}_outer_scale_denominator": torch.tensor([2688.0])}
    else:
        weight, scale = xf.quantize_mxfp8(W)
        t = {name: weight, f"{name}_scale": scale}
    meta = {k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in t.items()}
    return t, meta


@pytest.mark.parametrize("codec", [xf.NVFP4_AMAX, xf.MXFP8], ids=lambda c: c.name)
def test_codec_is_found_and_round_trips(codec):
    torch.manual_seed(0)
    W = torch.randn(E, H, I) * torch.rand(E, 1, 1)
    t, meta = _encoded(codec, W)
    found, keys = xf.find_codec("w2_weight", meta)
    assert found is codec and keys[0] == "w2_weight"
    W0 = codec.decode(t, keys)
    assert (W0 - W).norm() / W.norm() < (0.12 if codec is xf.NVFP4_AMAX else 0.04)
    xf.assert_reencode_fixed_point(codec, keys, t, name="w2_weight")


def test_codecs_leave_other_layouts_alone():
    # ModelOpt NVFP4: `.weight` + e4m3 `.weight_scale` + `.weight_scale_2`, no amax.
    modelopt = {
        "x.weight": torch.empty(4, 8, dtype=torch.uint8, device="meta"),
        "x.weight_scale": torch.empty(4, 1, dtype=torch.float8_e4m3fn, device="meta"),
        "x.weight_scale_2": torch.empty((), dtype=torch.float32, device="meta"),
    }
    assert xf.find_codec("x.weight", modelopt) is None
    # Block-wise FP8 with float32 inverse scales.
    blockwise = {
        "y.weight": torch.empty(256, 256, dtype=torch.float8_e4m3fn, device="meta"),
        "y.weight_scale_inv": torch.empty(2, 2, dtype=torch.float32, device="meta"),
    }
    assert xf.find_codec("y.weight", blockwise) is None


def test_nvfp4_amax_decodes_by_the_formula():
    packed = torch.tensor([[[0x7F, 0x18]]], dtype=torch.uint8)  # codes 15,7 then 8,1
    scale = torch.tensor([[[2.0]]]).to(torch.float8_e4m3fn).repeat(1, 1, 1)
    packed = packed.repeat(1, 1, 8)  # 16 values: one block
    W = xf.dequantize_nvfp4_amax(packed, scale, torch.tensor([2688.0]), torch.tensor([2688.0]))
    # low nibble first: 0xF -> -6, 0x7 -> 6, 0x8 -> -0, 0x1 -> 0.5; times scale 2
    assert W[0, 0, :4].tolist() == [-12.0, 12.0, 0.0, 1.0]


def _write_release(root: Path, layers: dict[int, object], extra_shard=True):
    """A two- or three-shard checkpoint in Inkling 2's names: an attention
    output per layer (BF16) and fused experts per layer in ``layers``' codec."""
    torch.manual_seed(1)
    shards, index = {}, {}
    for n, codec in layers.items():
        name = f"model.llm.layers.{n}.mlp.experts.w2_weight"
        t, _ = _encoded(codec, torch.randn(E, H, I) * 0.05, name)
        t[f"model.llm.layers.{n}.attn.wo_ud.weight"] = (torch.randn(H, 32) * 0.05).to(torch.bfloat16)
        shards[f"model-{n:05d}.safetensors"] = t
    if extra_shard:
        shards["model-99999.safetensors"] = {"model.llm.embed.weight": torch.randn(10, H).to(torch.bfloat16)}
    root.mkdir(parents=True, exist_ok=True)
    for fname, t in shards.items():
        save_file(t, str(root / fname))
        index.update({k: fname for k in t})
    import json

    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}))
    (root / "config.json").write_text(json.dumps({"model_type": "inkling2"}))
    (root / "tokenizer_config.json").write_text("{}")
    save_file({"draft.weight": torch.ones(2, 2)}, str(root / "mtp.safetensors"))
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "vocab.tiktoken").write_text("x")
    return shards


def _taps(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.tensor([0.01, 0.02, 0.13, 0.25]) + torch.tensor([0.2, 0.3, 0.6, 1.07]) * torch.randn(H, 4, generator=g)


def _plan(layers):
    r = torch.nn.functional.normalize(torch.randn(H, generator=torch.Generator().manual_seed(2)), dim=0)
    edits = []
    for n in layers:
        basis = conv_span_basis(r, _taps(n))
        edits.append(rp.TensorEdit(f"model.llm.layers.{n}.mlp.experts.w2_weight", "ega", r, 1.0, True,
                                   hidden_dim=H, basis=basis))
        edits.append(rp.TensorEdit(f"model.llm.layers.{n}.attn.wo_ud.weight", "direct", r, 2.0, True,
                                   basis=basis))
    return r, edits


def _load(path: Path) -> dict:
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def test_bake_edits_both_formats_and_carries_the_rest(tmp_path):
    layers = {0: xf.NVFP4_AMAX, 1: xf.MXFP8}
    src = tmp_path / "src"
    shards = _write_release(src, layers)
    r, edits = _plan(layers)
    dst = tmp_path / "dst"
    stats = rp.abliterate_fp4_to_disk(src, dst, edits, use_cuda=False, verbose=False)
    assert stats.codec_edited == 2 and stats.dense_edited == 2 and stats.linked_shards == 1
    # Unedited shards and weights outside the index are the source's own files.
    assert os.path.samefile(src / "model-99999.safetensors", dst / "model-99999.safetensors")
    assert torch.equal(_load(dst / "mtp.safetensors")["draft.weight"], torch.ones(2, 2))
    assert not os.path.samefile(src / "tokenizer_config.json", dst / "tokenizer_config.json")
    assert (dst / "tiktoken" / "vocab.tiktoken").read_text() == "x"
    for n, codec in layers.items():
        name = f"model.llm.layers.{n}.mlp.experts.w2_weight"
        before, after = shards[f"model-{n:05d}.safetensors"], _load(dst / f"model-{n:05d}.safetensors")
        assert set(after) == set(before)
        for k in before:
            assert after[k].dtype == before[k].dtype and after[k].shape == before[k].shape
        keys = codec.keys(name, before)
        W0, W1 = codec.decode(before, keys), codec.decode(after, keys)
        # The span is gone from the experts' output, to the format's precision.
        basis = edits[2 * n].basis
        left = (basis.T @ W1).norm() / (basis.T @ W0).norm()
        assert left < (0.35 if codec is xf.NVFP4_AMAX else 0.15), left
        # And the bake wrote exactly the codec's encoding of the edited tensor.
        expected = codec.encode(rp.apply_tensor_edits(W0, [edits[2 * n]]), before, keys)
        for k in keys:
            assert torch.equal(after[k].view(torch.uint8), expected[k].view(torch.uint8)), k


def test_expert_chunks_change_nothing(tmp_path, monkeypatch):
    layers = {0: xf.NVFP4_AMAX, 1: xf.MXFP8}
    src = tmp_path / "src"
    _write_release(src, layers, extra_shard=False)
    _, edits = _plan(layers)
    monkeypatch.setattr(rp, "EXPERT_CHUNK", 3)
    rp.abliterate_fp4_to_disk(src, tmp_path / "small", edits, use_cuda=False, verbose=False)
    monkeypatch.setattr(rp, "EXPERT_CHUNK", 1000)
    rp.abliterate_fp4_to_disk(src, tmp_path / "whole", edits, use_cuda=False, verbose=False)
    for n in layers:
        a, b = _load(tmp_path / "small" / f"model-{n:05d}.safetensors"), _load(tmp_path / "whole" / f"model-{n:05d}.safetensors")
        for k in a:
            assert torch.equal(a[k].view(torch.uint8) if a[k].dtype != torch.bfloat16 else a[k],
                               b[k].view(torch.uint8) if b[k].dtype != torch.bfloat16 else b[k]), k


@pytest.mark.skipif(not RELEASE.exists(), reason="Inkling 2 release shards not on disk")
@pytest.mark.parametrize("shard,layer", [("model-00001-of-00024.safetensors", 0),
                                         ("model-00015-of-00024.safetensors", 40)])
def test_reencoding_the_release_gives_back_its_codes(shard, layer):
    name = f"model.llm.layers.{layer}.mlp.experts.w2_weight"
    path = RELEASE / shard
    if not path.exists():
        pytest.skip(f"{shard} not downloaded")
    with safe_open(str(path), "pt") as f:
        meta = {k: torch.empty(f.get_slice(k).get_shape(), device="meta",
                               dtype=rp._ST_DTYPES[f.get_slice(k).get_dtype()]) for k in f.keys()
                if k.startswith(name)}
        codec, keys = xf.find_codec(name, meta)
        t = {k: f.get_tensor(k) if k.endswith("denominator") else f.get_slice(k)[:4] for k in keys}
    assert codec is (xf.NVFP4_AMAX if layer < 40 else xf.MXFP8)
    again = codec.encode(codec.decode(t, keys), t, keys)
    if codec is xf.MXFP8:
        for k in keys:
            assert torch.equal(again[k].view(torch.uint8), t[k].view(torch.uint8)), k
    else:
        w, s, a, _ = keys
        assert torch.equal(again[s].view(torch.uint8), t[s].view(torch.uint8))
        assert torch.allclose(again[a], t[a], rtol=1e-6, atol=0)
        # -0 and +0 are one value; compare codes as values.
        assert torch.equal(xf.dequantize_nvfp4_amax(again[w], t[s], t[a], t[keys[3]]),
                           xf.dequantize_nvfp4_amax(t[w], t[s], t[a], t[keys[3]]))
