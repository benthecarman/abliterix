# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read bitsandbytes int8 weights across its parameter/state lifecycle."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _unwrap(base_layer: nn.Module) -> nn.Module:
    seen: set[int] = set()
    while getattr(base_layer, "base_layer", None) is not None:
        if id(base_layer) in seen:
            raise RuntimeError(
                "Cyclic base_layer wrappers prevent int8 weight inspection."
            )
        seen.add(id(base_layer))
        base_layer = base_layer.base_layer
    return base_layer


def _has_float_master(base_layer: nn.Module, weight: Tensor) -> bool:
    state = getattr(base_layer, "state", None)
    return weight.is_floating_point() and (
        getattr(weight, "has_fp16_weights", False)
        or getattr(state, "has_fp16_weights", False)
    )


def is_bnb_int8_weight(base_layer: nn.Module) -> bool:
    """Recognize int8 storage even after bnb moves CB/SCB off the parameter."""
    base_layer = _unwrap(base_layer)
    weight = getattr(base_layer, "weight", None)
    if not isinstance(weight, Tensor) or _has_float_master(base_layer, weight):
        return False
    state = getattr(base_layer, "state", None)
    return weight.dtype == torch.int8 or any(
        getattr(owner, field, None) is not None
        for owner in (weight, state)
        for field in ("CB", "SCB")
    )


def _validated_pair(base_layer: nn.Module) -> tuple[Tensor, Tensor]:
    weight = getattr(base_layer, "weight", None)
    if not isinstance(weight, Tensor):
        raise RuntimeError(
            "Cannot dequantize bitsandbytes int8: the layer has no weight."
        )
    state = getattr(base_layer, "state", None)
    cb = scb = None
    source = ""
    sources = (("weight", weight), ("state", state))
    data_fallback = False
    for owner_name, owner in sources:
        owner_cb = getattr(owner, "CB", None)
        owner_scb = getattr(owner, "SCB", None)
        if owner_cb is not None and owner_scb is not None:
            cb, scb = owner_cb, owner_scb
            source = owner_name
            break
    if cb is None and weight.dtype == torch.int8:
        for owner_name, owner in sources:
            owner_scb = getattr(owner, "SCB", None)
            if owner_scb is not None:
                cb, scb = weight.detach(), owner_scb
                source = owner_name
                data_fallback = True
                break

    prefix = "Cannot dequantize bitsandbytes int8 weight"
    recovery = (
        " Reload the original checkpoint with its quantization metadata and "
        "check the resolved device map and bitsandbytes/PEFT compatibility. "
        "Raw int8 values cannot be used without their original row scales."
    )
    if cb is None or scb is None:
        raise RuntimeError(f"{prefix}: missing a coherent CB/SCB pair.{recovery}")
    if not isinstance(cb, Tensor) or not isinstance(scb, Tensor):
        raise RuntimeError(f"{prefix}: {source}.CB and SCB must be tensors.{recovery}")
    if weight.is_meta or cb.is_meta or scb.is_meta:
        raise RuntimeError(
            f"{prefix}: weight or {source} metadata is on meta.{recovery}"
        )
    if cb.dtype != torch.int8 or cb.ndim != 2:
        raise RuntimeError(
            f"{prefix}: {source}.CB must be a 2-D int8 tensor.{recovery}"
        )
    if weight.ndim != 2 or cb.shape != weight.shape:
        raise RuntimeError(
            f"{prefix}: {source}.CB shape differs from weight.{recovery}"
        )
    if scb.ndim != 1 or scb.shape[0] != cb.shape[0]:
        raise RuntimeError(
            f"{prefix}: {source}.SCB must have one scale per row.{recovery}"
        )
    if not scb.is_floating_point():
        raise RuntimeError(f"{prefix}: {source}.SCB must be floating point.{recovery}")
    if cb.device != scb.device or cb.device != weight.device:
        raise RuntimeError(
            f"{prefix}: weight, CB and SCB must share a device.{recovery}"
        )
    scales = scb.detach().to(torch.float32)
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise RuntimeError(
            f"{prefix}: SCB scales must be finite and nonnegative.{recovery}"
        )
    if data_fallback and (
        not weight.is_contiguous() or getattr(state, "formatB", "row") != "row"
    ):
        raise RuntimeError(
            f"{prefix}: raw data fallback requires row-major storage.{recovery}"
        )
    partial_cb = getattr(weight, "CB", None)
    missing_weight_scales = (
        partial_cb is not None and getattr(weight, "SCB", None) is None
    )
    if missing_weight_scales:
        same_storage = (
            isinstance(partial_cb, Tensor)
            and not partial_cb.is_meta
            and partial_cb.dtype == cb.dtype
            and partial_cb.device == cb.device
            and partial_cb.shape == cb.shape
            and partial_cb.stride() == cb.stride()
            and partial_cb.storage_offset() == cb.storage_offset()
            and partial_cb.untyped_storage().data_ptr()
            == cb.untyped_storage().data_ptr()
        )
        if not same_storage:
            raise RuntimeError(
                f"{prefix}: partial weight.CB does not share storage with the "
                f"validated CB/SCB pair.{recovery}"
            )
    if data_fallback and callable(getattr(base_layer, "init_8bit_state", None)):
        if state is None:
            raise RuntimeError(
                f"{prefix}: live layer has no int8 runtime state.{recovery}"
            )
        state.CB, state.SCB = cb, scb
        weight.CB = weight.SCB = None
    elif missing_weight_scales:
        weight.SCB = scb
    elif source == "state" and callable(getattr(base_layer, "init_8bit_state", None)):
        if getattr(weight, "CB", None) is None:
            weight.SCB = None
    return cb.detach(), scales


def validate_bnb_int8_weight(base_layer: nn.Module) -> None:
    """Validate and normalize known int8 metadata without a dense allocation.

    Matching partial weight metadata is completed from the existing state.
    A live bnb layer with only row-major data and scales receives the same
    state layout as ``init_8bit_state`` so its next forward cannot requantize
    raw integer codes. Tensor values are never changed or reconstructed.
    """
    base_layer = _unwrap(base_layer)
    weight = getattr(base_layer, "weight", None)
    if isinstance(weight, Tensor) and _has_float_master(base_layer, weight):
        return
    _validated_pair(base_layer)


def dequantize_bnb_int8_weight(base_layer: nn.Module) -> Tensor:
    """Return float32 row-major weights using a matching CB/SCB source.

    Before the first forward, metadata lives on ``weight``; afterwards it
    usually lives on ``state``. Serialized row-major int8 data may also retain
    only SCB. Never pair a CB from one owner with another owner's scales.
    Floating master weights remain authoritative when bnb retains them. Any
    unambiguous partial metadata is normalized as in the validation helper.
    """
    base_layer = _unwrap(base_layer)
    weight = getattr(base_layer, "weight", None)
    if isinstance(weight, Tensor) and _has_float_master(base_layer, weight):
        if weight.is_meta:
            raise RuntimeError(
                "Cannot read bitsandbytes floating master weights on meta."
            )
        return weight.detach().to(torch.float32)
    cb, scales = _validated_pair(base_layer)
    return cb.to(torch.float32) * (scales.unsqueeze(1) / 127.0)
