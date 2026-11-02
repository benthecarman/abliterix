# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate quantization metadata without materializing offloaded weights."""

from __future__ import annotations

from torch import Tensor, nn


def validate_quantization_state(
    nn_model: nn.Module, *, stage: str = "during model setup"
) -> None:
    """Reject bnb quantization metadata whose values were lost to meta tensors.

    Accelerate may legitimately keep an offloaded parameter on ``meta``. The
    tensors inside a bnb ``QuantState`` are different: ``Params4bit.to`` copies
    them directly, so a meta codebook or scale cannot be recovered by its normal
    offload hook. Check both aliases used by bnb without importing optional bnb
    kernels or changing any tensor. A missing state can be lazily initialized
    by bnb and is not, by itself, evidence of corruption.
    """
    for module_name, module in nn_model.named_modules():
        weight = getattr(module, "weight", None)
        states = (
            ("weight.quant_state", getattr(weight, "quant_state", None)),
            ("quant_state", getattr(module, "quant_state", None)),
        )
        seen: set[int] = set()
        pending = list(states)
        while pending:
            state_name, state = pending.pop()
            if state is None or id(state) in seen:
                continue
            seen.add(id(state))
            for field in ("code", "absmax", "offset"):
                tensor = getattr(state, field, None)
                if isinstance(tensor, Tensor) and tensor.is_meta:
                    path = ".".join(
                        part for part in (module_name, state_name, field) if part
                    )
                    raise RuntimeError(
                        "bitsandbytes quantization metadata is on the meta device "
                        f"{stage}: {path}. Its values cannot be recovered by "
                        "copying the meta tensor. Check CUDA_VISIBLE_DEVICES, "
                        "model.max_memory, and the loaded model's hf_device_map "
                        "for unexpected CPU/disk offload. Reload with enough "
                        "visible GPU memory or a supported offload configuration."
                    )
            pending.append((f"{state_name}.state2", getattr(state, "state2", None)))
