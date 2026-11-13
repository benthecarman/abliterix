"""Block-scaled expert formats that carry their own scale rule.

The bake (:func:`fp4_repack.abliterate_fp4_to_disk`) decodes an edited tensor,
edits it and encodes it back. For gpt-oss, DeepSeek and ModelOpt checkpoints it
does that with the layouts in ``fp4_repack``. The formats here differ from
those in where the scales live and how they are chosen, so each has its own
codec. Each codec is checked against a real release: re-encoding a decoded
tensor gives back the stored codes and scales, so an edited tensor is written
the way its producer wrote the rest.

* ``nvfp4_amax``: Thinking Machines' NVFP4 (Inkling 2, layers 0-39). A packed
  ``uint8`` weight ``[E, N, K/2]`` (two E2M1 codes a byte, low nibble first,
  bit 3 the sign), an E4M3 scale per 1x16 block (``<name>_scale``), a
  per-expert ``<name>_amax`` and a ``<name>_outer_scale_denominator`` (6 * 448).
  ``W = code * scale * amax[e] / denominator``. Encoding takes each expert's
  amax afresh, so block scales stay inside E4M3 after an edit.
* ``mxfp8``: OCP MXFP8 with the scale as a sibling ``<name>_scale``. An
  ``float8_e4m3fn`` weight and one UE8M0 exponent per 1x32 block along the
  last axis: ``W = q * 2^(s - 127)``. A block's exponent is the smallest that
  keeps its amax within E4M3's 448 (``ceil``, not the OCP spec's ``floor`` of
  ``log2(amax) - 8``; the Inkling 2 release uses ``ceil``).

Both store experts on the leading axis with every scale per expert or finer,
so a tensor can be edited a slice of experts at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32

_E4M3 = getattr(torch, "float8_e4m3fn", None)


# ---------------------------------------------------------------------------
# NVFP4 with a per-expert amax
# ---------------------------------------------------------------------------


def _per_expert(t: Tensor, ndim: int) -> Tensor:
    return t.reshape(-1, *([1] * (ndim - 1))) if ndim > 1 else t.reshape(())


def dequantize_nvfp4_amax(
    packed: Tensor, scale: Tensor, amax: Tensor, denominator: Tensor
) -> Tensor:
    """FP32 ``[E, N, K]`` from packed ``[E, N, K/2]``, E4M3 scale ``[E, N, K/16]``
    and amax ``[E]`` (a 2-D weight takes a one-element amax)."""
    magnitude = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    table = torch.cat([magnitude, -magnitude])
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    values = table[codes].unflatten(-1, (-1, NVFP4_BLOCK))
    values = values * scale.to(packed.device).float().unsqueeze(-1)
    factor = amax.to(packed.device).float() / denominator.to(packed.device).float()
    return values.flatten(-2) * _per_expert(factor, packed.dim())


def quantize_nvfp4_amax(W: Tensor, denominator: float) -> tuple[Tensor, ...]:
    """``W`` encoded the way the release encodes it: each expert's amax, E4M3
    block scales of block amax / 6 in units of amax / denominator, nearest E2M1
    codes (ties to the lower magnitude). Returns ``(packed, scale, amax)``."""
    W = W.float()
    amax = W.abs().flatten(1).amax(dim=1) if W.dim() > 2 else W.abs().amax().reshape(1)
    factor = _per_expert(amax / denominator, W.dim())
    blocks = W.unflatten(-1, (-1, NVFP4_BLOCK))
    block_amax = blocks.abs().amax(dim=-1)
    scale = (block_amax / 6.0 / torch.where(factor == 0, 1.0, factor)).clamp(max=E4M3_MAX)
    scale = scale.to(_E4M3)
    divisor = scale.float() * factor
    ratio = blocks / torch.where(divisor == 0, 1.0, divisor).unsqueeze(-1)
    grid = torch.tensor(E2M1, dtype=torch.float32, device=W.device)
    index = torch.bucketize(ratio.abs(), (grid[:-1] + grid[1:]) / 2)
    codes = (index + 8 * (ratio < 0)).to(torch.uint8).flatten(-2)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed, scale, amax


# ---------------------------------------------------------------------------
# MXFP8 (UE8M0 exponent per 1x32 block)
# ---------------------------------------------------------------------------


def dequantize_mxfp8(weight: Tensor, scale: Tensor) -> Tensor:
    """FP32 from an E4M3 weight ``[..., K]`` and UE8M0 exponents ``[..., K/32]``."""
    exponent = torch.exp2(scale.to(weight.device).float() - 127.0).unsqueeze(-1)
    return (weight.float().unflatten(-1, (-1, MXFP8_BLOCK)) * exponent).flatten(-2)


def quantize_mxfp8(W: Tensor) -> tuple[Tensor, Tensor]:
    """Each block's exponent is the smallest power of two that keeps its amax
    within 448; codes are the nearest E4M3. Returns ``(weight, scale)``."""
    blocks = W.float().unflatten(-1, (-1, MXFP8_BLOCK))
    block_amax = blocks.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2(block_amax / E4M3_MAX)).clamp(-127, 127)
    exponent = torch.where(block_amax == 0, torch.full_like(exponent, -127.0), exponent)
    codes = (blocks / torch.exp2(exponent).unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
    return codes.to(_E4M3).flatten(-2), (exponent + 127).to(torch.uint8)


# ---------------------------------------------------------------------------
# Codecs: which keys a tensor is stored under, and both directions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Codec:
    """One stored format. ``keys`` finds the tensor's keys (weight first) from
    the checkpoint's meta tensors, or returns None if it is not stored this
    way. ``decode`` and ``encode`` take and return tensors keyed by those
    names. ``shared`` names the keys that hold one value for every expert; the
    rest have experts on their leading axis."""

    name: str
    keys: Callable[[str, dict[str, Tensor]], tuple[str, ...] | None]
    decode: Callable[[dict[str, Tensor], tuple[str, ...]], Tensor]
    encode: Callable[[Tensor, dict[str, Tensor], tuple[str, ...]], dict[str, Tensor]]
    shared: Callable[[tuple[str, ...]], tuple[str, ...]] = lambda keys: ()


def _nvfp4_amax_keys(name: str, meta: dict[str, Tensor]) -> tuple[str, ...] | None:
    keys = (name, f"{name}_scale", f"{name}_amax", f"{name}_outer_scale_denominator")
    if not all(k in meta for k in keys) or meta[name].dtype != torch.uint8:
        return None
    if meta[keys[1]].dtype != _E4M3:
        return None
    return keys


def _nvfp4_amax_decode(t: dict[str, Tensor], keys: tuple[str, ...]) -> Tensor:
    w, s, a, d = keys
    return dequantize_nvfp4_amax(t[w], t[s], t[a], t[d])


def _nvfp4_amax_encode(
    W: Tensor, t: dict[str, Tensor], keys: tuple[str, ...]
) -> dict[str, Tensor]:
    w, s, a, d = keys
    packed, scale, amax = quantize_nvfp4_amax(W, float(t[d].float().reshape(-1)[0]))
    return {w: packed, s: scale, a: amax.to(t[a].dtype).reshape(t[a].shape), d: t[d]}


def _mxfp8_keys(name: str, meta: dict[str, Tensor]) -> tuple[str, ...] | None:
    scale = f"{name}_scale"
    if name not in meta or scale not in meta or meta[name].dtype != _E4M3:
        return None
    w, s = meta[name], meta[scale]
    if s.dtype != torch.uint8 or s.shape[:-1] != w.shape[:-1]:
        return None
    if s.shape[-1] * MXFP8_BLOCK != w.shape[-1]:
        return None
    return (name, scale)


def _mxfp8_decode(t: dict[str, Tensor], keys: tuple[str, ...]) -> Tensor:
    return dequantize_mxfp8(t[keys[0]], t[keys[1]])


def _mxfp8_encode(W: Tensor, t: dict[str, Tensor], keys: tuple[str, ...]) -> dict[str, Tensor]:
    weight, scale = quantize_mxfp8(W)
    return {keys[0]: weight, keys[1]: scale}


NVFP4_AMAX = Codec(
    "nvfp4_amax", _nvfp4_amax_keys, _nvfp4_amax_decode, _nvfp4_amax_encode,
    shared=lambda keys: (keys[3],),
)
MXFP8 = Codec("mxfp8", _mxfp8_keys, _mxfp8_decode, _mxfp8_encode)

CODECS: tuple[Codec, ...] = (NVFP4_AMAX, MXFP8) if _E4M3 is not None else ()


def find_codec(name: str, meta: dict[str, Tensor]) -> tuple[Codec, tuple[str, ...]] | None:
    """The codec ``name`` (or ``name`` without ``.weight``) is stored with, and
    its keys; None when no codec here matches."""
    candidates = [name]
    if name.endswith(".weight"):
        candidates.append(name[: -len(".weight")])
    for cand in candidates:
        for codec in CODECS:
            keys = codec.keys(cand, meta)
            if keys is not None:
                return codec, keys
    return None


def expert_slices(codec: Codec, keys: tuple[str, ...], t: dict[str, Tensor], start: int, stop: int) -> dict[str, Tensor]:
    """Experts ``start:stop`` of every key, the shared ones whole."""
    shared = set(codec.shared(keys))
    return {k: t[k] if k in shared else t[k][start:stop] for k in keys}


def assert_reencode_fixed_point(codec: Codec, keys: tuple[str, ...], t: dict[str, Tensor], *, name: str) -> None:
    """Decode, encode and decode again must give the same values (to FP32
    rounding of the recomputed amax), or the layout handling does not match
    the producer: nibble order, scale rule, or block axis."""
    W0 = codec.decode(t, keys)
    W1 = codec.decode(codec.encode(W0, t, keys), keys)
    if not torch.allclose(W0, W1, rtol=1e-6, atol=0.0):
        drift = (W0 - W1).abs().max().item()
        raise AssertionError(
            f"{codec.name} re-encode of '{name}' is not a fixed point on unedited "
            f"weights (max drift {drift:g}); the format does not match this tensor."
        )
