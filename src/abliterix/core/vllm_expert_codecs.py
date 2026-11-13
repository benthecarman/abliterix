"""Codecs for vLLM's Inkling 2 experts, stored as the release stores them.

vLLM's Inkling 2 model (vllm-project/vllm#67325) keeps the release's routed
experts in their own formats, NVFP4 (layers 0-39) or MXFP8 (40-47), on an
``Inkling2Experts`` module: ``w2_weight`` ``[E, H, I/2]`` uint8 or ``[E, H, I]``
float8_e4m3fn, ``w2_weight_scale`` (E4M3 per 1x16 block or UE8M0 per 1x32),
and for NVFP4 ``w2_weight_amax`` ``[E]`` and ``w2_weight_outer_scale_denominator``.
At load it shuffles each expert's rows in blocks of 32 and swizzles the block
scales into 128x4 tiles for the TRT-LLM-gen GEMMs (``ops/block_scales.py``).

The codec undoes that layout, decodes with :mod:`.expert_formats` (the bake's
codecs, checked against the release), and encodes an edited tensor the same
way the bake writes it, then lays it out again. So a trial edits the weights
vLLM runs into exactly what the plan's replay writes to disk.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from . import expert_formats as xf

_E4M3 = getattr(torch, "float8_e4m3fn", None)


def shuffle_rows(x: Tensor) -> Tensor:
    """``[..., R, C]``: row ``4 * a + b`` of each block of 32 moves to ``8 * b + a``
    (vLLM's code; its docstring states the inverse)."""
    *lead, rows, cols = x.shape
    return x.reshape(-1, rows // 32, 8, 4, cols).transpose(2, 3).reshape(x.shape).contiguous()


def unshuffle_rows(x: Tensor) -> Tensor:
    *lead, rows, cols = x.shape
    return x.reshape(-1, rows // 32, 4, 8, cols).transpose(2, 3).reshape(x.shape).contiguous()


def swizzle_block_scales(s: Tensor) -> Tensor:
    """``[..., R, C]`` row-major scales to 128-row, 4-column tiles (R128c4)."""
    *lead, rows, cols = s.shape
    tiles = s.reshape(-1, rows // 128, 4, 32, cols // 4, 4)
    return tiles.permute(0, 1, 4, 3, 2, 5).reshape(s.shape).contiguous()


def unswizzle_block_scales(s: Tensor) -> Tensor:
    *lead, rows, cols = s.shape
    tiles = s.reshape(-1, rows // 128, cols // 4, 32, 4, 4)
    return tiles.permute(0, 1, 4, 3, 2, 5).reshape(s.shape).contiguous()


_LAYOUT_CHECKED: list[bool] = []


def _check_layout_against_vllm() -> None:
    """Our layout functions must be vLLM's. Where vLLM's Inkling 2 ops import,
    compare them once on random tensors; a different layout in the installed
    build fails here instead of editing the wrong rows."""
    if _LAYOUT_CHECKED:
        return
    try:
        from vllm.models.inkling2.ops import block_scales as ref  # type: ignore
    except Exception:  # noqa: BLE001 - not this vLLM build, or no vLLM (CPU tests)
        _LAYOUT_CHECKED.append(False)
        return
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 255, (2, 256, 8), generator=g, dtype=torch.uint8)
    if not torch.equal(ref.shuffle_rows(x), shuffle_rows(x)):
        raise RuntimeError("vLLM's Inkling 2 shuffle_rows is not the layout abliterix assumes")
    if not torch.equal(ref.swizzle_block_scales(x), swizzle_block_scales(x)):
        raise RuntimeError("vLLM's Inkling 2 block-scale swizzle is not the layout abliterix assumes")
    _LAYOUT_CHECKED.append(True)


class InklingExpertCodec:
    """Decode/encode a slice of experts of an ``Inkling2Experts.w2_weight``.

    Edits run in FP32 from decode to encode (``round_between`` None), as the
    bake does: the only rounding is the final encode."""

    round_between = None

    def __init__(self, owner: Any, weight: Tensor):
        self.weight = weight
        self.scale = owner.w2_weight_scale
        self.nvfp4 = weight.dtype == torch.uint8
        self.amax = getattr(owner, "w2_weight_amax", None) if self.nvfp4 else None
        self.denominator = (
            getattr(owner, "w2_weight_outer_scale_denominator", None) if self.nvfp4 else None
        )
        self.stored = [self.weight, self.scale] + ([self.amax] if self.nvfp4 else [])

    def decode(self, tgt: Any, rows: slice) -> Tensor:
        w = unshuffle_rows(self.weight.data[rows])
        s = unshuffle_rows(unswizzle_block_scales(self.scale.data[rows]))
        if self.nvfp4:
            return xf.dequantize_nvfp4_amax(w, s, self.amax.data[rows], self.denominator.data)
        return xf.dequantize_mxfp8(w, s)

    def encode(self, tgt: Any, rows: slice, values: Tensor) -> None:
        if self.nvfp4:
            denominator = float(self.denominator.data.float().reshape(-1)[0])
            w, s, amax = xf.quantize_nvfp4_amax(values, denominator)
            self.amax.data[rows].copy_(amax.to(self.amax.dtype))
        else:
            w, s = xf.quantize_mxfp8(values)
        self.weight.data[rows].copy_(shuffle_rows(w))
        self.scale.data[rows].copy_(swizzle_block_scales(shuffle_rows(s)))


def inkling_expert_codec(tgt: Any):
    """The codec for a plan target on an ``Inkling2Experts`` ``w2_weight`` in
    NVFP4 or MXFP8, else None."""
    owner = getattr(tgt, "owner", None)
    if owner is None or _E4M3 is None or getattr(tgt, "transposed", False):
        return None
    weight = getattr(owner, "w2_weight", None)
    scale = getattr(owner, "w2_weight_scale", None)
    if weight is None or scale is None or tgt.param is not weight or weight.dim() != 3:
        return None
    if weight.dtype == torch.uint8:
        if scale.dtype != _E4M3 or getattr(owner, "w2_weight_amax", None) is None:
            return None
        if getattr(owner, "w2_weight_outer_scale_denominator", None) is None:
            return None
    elif weight.dtype == _E4M3:
        if scale.dtype != torch.uint8:
            return None
    else:
        return None
    _check_layout_against_vllm()
    return InklingExpertCodec(owner, weight)
