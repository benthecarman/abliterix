# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline "edit-and-repack" bake path for native FP4 (MXFP4 / NVFP4) models.

Direct/EGA abliteration edits only a *subset* of a model's tensors (fused
expert ``down_proj``, ``attn.o_proj``, dense ``mlp.down_proj``), and each edit
is *local* to one tensor. That is the lever this module pulls: instead of
expanding a 284B FP4 checkpoint to ~568GB of BF16 just to steer it (2×+ the
footprint, 4× B200), it streams the original FP4 checkpoint shard by shard and,
for each edited tensor only, does ``dequant → project → re-pack to the same
FP4 format`` — every other tensor is copied through byte-for-byte. Peak memory
is a single layer's expert block, and the output is a clean standalone FP4
checkpoint that loads in any NVFP4/MXFP4-capable stack with no special support.

This is the counterpart of :func:`abliterix.core.fp8_utils.dequant_model_to_disk`
(FP8 → BF16 on disk) but it keeps the model 4-bit end to end.

Two-phase workflow
------------------
1. **Search / export** (model loaded at any precision): call
   :func:`record_steering_plan` at the moment the engine would apply the best
   trial. It records the resolved per-tensor edits (direction + strength +
   projection geometry) keyed by canonical parameter name — cheap, no extra
   memory beyond the already-loaded model.
2. **Bake** (single GPU, streams disk): :func:`abliterate_fp4_to_disk` replays
   that plan against the original FP4 shards.

Faithfulness
    The projection math is the *shared* :mod:`abliterix.weight_transforms`
    kernels the in-engine HF path uses, so the abliteration fingerprint is
    bit-identical. EGA axis is re-resolved from the on-disk dequantised shape
    (via recorded ``hidden_dim`` + ``transposed``) so a producer that stores
    experts transposed (gpt-oss) still steers the correct axis. Before writing,
    each source FP4 tensor is cross-checked against the model's own dequant
    when a reference is supplied (see :func:`abliterate_fp4_to_disk`).
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from . import fp4_utils as f4
from . import fp8_utils as f8
from ..types import DecayKernel, DirectTransform, WeightNorm
from ..weight_transforms import (
    apply_direct_transform,
    apply_ega_projection,
    resolve_ega_axis,
)

# ---------------------------------------------------------------------------
# Edit specification
# ---------------------------------------------------------------------------


@dataclass
class TensorEdit:
    """One resolved abliteration edit targeting a single named weight tensor.

    ``logical_name`` is the canonical (base-model, PEFT-stripped) parameter
    name, e.g. ``model.layers.5.mlp.experts.down_proj``. The offline tool maps
    it to the on-disk FP4 sibling keys (``*_blocks`` / ``*_scales`` / global).

    Kinds:

    * ``"direct"`` — rank-1 projection of a 2-D Linear weight.
    * ``"ega"`` — projection of every expert slice of a fused expert tensor
      (or of one per-expert 2-D tensor).
    * ``"row_scale"`` — router suppression: ``rows`` of a 2-D router weight
      multiplied by ``scale``.
    * ``"expert_project"`` — the safety-expert step of MoE steering: a plain
      (unnormalised, not norm-preserving) projection of expert ``expert``'s
      slice of a fused expert tensor, or of a whole per-expert 2-D tensor when
      ``expert`` is ``None``.

    Several edits may name the same tensor; they apply in plan order, the
    order the engine applies them in.
    """

    logical_name: str
    kind: str  # "ega" | "direct" | "row_scale" | "expert_project"
    direction: Tensor | None  # 1-D float refusal direction (None for row_scale)
    strength: float
    preserve_row_norm: bool
    # EGA (fused 3-D expert tensor):
    hidden_dim: int | None = None
    transposed: bool = False
    # Direct (2-D Linear weight):
    projection_side: str | None = None  # "output" | "input"
    direct_transform: str = "standard"
    benign_dir: Tensor | None = None
    # row_scale (router suppression):
    rows: list[int] | None = None
    scale: float | None = None
    # expert_project (one expert's slice):
    expert: int | None = None


def apply_tensor_edit(W: Tensor, edit: TensorEdit) -> Tensor:
    """Apply one :class:`TensorEdit` to a dequantised float weight.

    Returns a new float32 tensor (caller re-quantises / casts). Raises
    ``ValueError`` if the tensor shape is incompatible with the edit; the
    streaming caller turns that into a skip-with-warning rather than aborting.
    """
    W32 = W.to(torch.float32)
    if edit.kind == "row_scale":
        if W32.dim() != 2 or not edit.rows or edit.scale is None:
            raise ValueError(
                f"row_scale edit '{edit.logical_name}' expects a 2-D router "
                f"weight, rows and a scale; got shape {tuple(W.shape)}"
            )
        if max(edit.rows) >= W32.shape[0]:
            raise ValueError(
                f"row_scale edit '{edit.logical_name}': row {max(edit.rows)} "
                f"out of range for {W32.shape[0]} experts"
            )
        out = W32.clone()
        rows = torch.tensor(edit.rows, device=out.device)
        out[rows] = out[rows] * edit.scale
        return out
    if edit.kind == "expert_project":
        return _apply_expert_project(W32, edit)
    if edit.kind == "ega":
        # Producers differ in how experts are stored: gpt-oss fuses them into a
        # 3-D (E, d0, d1) parameter, DeepSeek-V4 keeps one 2-D (out, in) tensor
        # per expert. Give the 2-D case a singleton expert axis so BOTH run the
        # identical kernel — the abliteration fingerprint must not depend on
        # how the checkpoint happens to group experts.
        per_expert_2d = W32.dim() == 2
        if per_expert_2d:
            W32 = W32.unsqueeze(0)
        elif W32.dim() != 3:
            raise ValueError(
                f"EGA edit '{edit.logical_name}' expects a fused 3-D expert "
                f"tensor or a 2-D single-expert weight, got shape "
                f"{tuple(W.shape)}"
            )
        hidden = edit.hidden_dim
        if hidden is None:
            hidden = edit.direction.shape[0]
        axis_is_in = resolve_ega_axis(
            tuple(W32.shape), hidden, transposed=edit.transposed
        )
        if axis_is_in is None:
            raise ValueError(
                f"EGA edit '{edit.logical_name}': hidden dim {hidden} matches "
                f"neither axis of {tuple(W32.shape)} (transposed={edit.transposed})"
            )
        out = apply_ega_projection(
            W32,
            edit.direction,
            strength=edit.strength,
            axis_is_in=axis_is_in,
            preserve_row_norm=edit.preserve_row_norm,
        )
        return out.squeeze(0) if per_expert_2d else out

    # Direct 2-D weight.
    if W32.dim() != 2:
        raise ValueError(
            f"Direct edit '{edit.logical_name}' expects a 2-D weight, got "
            f"shape {tuple(W32.shape)}"
        )
    # Directions come from a plan loaded onto CPU while the weight may already
    # be on GPU (the bake streams shards to CUDA); always follow the weight.
    dev = W32.device
    transform = DirectTransform(edit.direct_transform)
    if transform != DirectTransform.STANDARD:
        # ORBA / biprojected / householder carry their own norm handling and
        # output-side preference — identical to the engine's advanced branch.
        return apply_direct_transform(
            transform,
            W32,
            edit.direction.to(device=dev, dtype=torch.float32),
            None
            if edit.benign_dir is None
            else edit.benign_dir.to(device=dev, dtype=torch.float32),
            strength=edit.strength,
            preserve_row_norm=edit.preserve_row_norm,
        ).to(torch.float32)

    # Standard rank-1 ablation with the engine's exact side precedence
    # (output-side first for square matrices) + optional row-norm post-step.
    v = _normalise(edit.direction.to(device=dev, dtype=torch.float32))
    out_f, in_f = W32.shape
    side = edit.projection_side
    if side is None:
        side = "output" if v.shape[0] == out_f else "input"
    if side == "output":
        if v.shape[0] != out_f:
            raise ValueError(
                f"Direct edit '{edit.logical_name}': output-side direction "
                f"len {v.shape[0]} != out_features {out_f}"
            )
        W_new = W32 - edit.strength * v.unsqueeze(1) * (v @ W32).unsqueeze(0)
    else:
        if v.shape[0] != in_f:
            raise ValueError(
                f"Direct edit '{edit.logical_name}': input-side direction "
                f"len {v.shape[0]} != in_features {in_f}"
            )
        W_new = W32 - edit.strength * (W32 @ v).unsqueeze(1) * v.unsqueeze(0)
    if edit.preserve_row_norm:
        orig = torch.linalg.vector_norm(W32, dim=1, keepdim=True)
        new = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig / new)
    return W_new


def _apply_expert_project(W32: Tensor, edit: TensorEdit) -> Tensor:
    """The safety-expert step of ``core.steering._apply_moe_steering``.

    Same math as the engine: the raw (not re-normalised) direction, no row-norm
    preservation, the axis resolved on the fused shape exactly as EGA does.
    """
    if W32.dim() == 3:
        if edit.expert is None or not 0 <= edit.expert < W32.shape[0]:
            raise ValueError(
                f"expert_project edit '{edit.logical_name}': expert "
                f"{edit.expert} not in a fused tensor of {W32.shape[0]} experts"
            )
        fused_shape = tuple(W32.shape)
        W = W32[edit.expert]
    elif W32.dim() == 2:
        fused_shape = (1, *W32.shape)
        W = W32
    else:
        raise ValueError(
            f"expert_project edit '{edit.logical_name}' expects a fused 3-D "
            f"expert tensor or a 2-D expert weight, got shape {tuple(W32.shape)}"
        )
    hidden = edit.hidden_dim if edit.hidden_dim is not None else edit.direction.shape[0]
    axis_is_in = resolve_ega_axis(fused_shape, hidden, transposed=edit.transposed)
    if axis_is_in is None:
        raise ValueError(
            f"expert_project edit '{edit.logical_name}': hidden dim {hidden} "
            f"matches neither axis of {fused_shape}"
        )
    vf = edit.direction.to(device=W.device, dtype=torch.float32)
    if axis_is_in:
        W_new = W - edit.strength * torch.outer(W @ vf, vf)
    else:
        W_new = W - edit.strength * torch.outer(vf, vf @ W)
    if W32.dim() == 2:
        return W_new
    out = W32.clone()
    out[edit.expert] = W_new
    return out


def apply_tensor_edits(
    W: Tensor, edits: list[TensorEdit], *, round_between: torch.dtype | None = None
) -> Tensor:
    """Apply several edits to one tensor, in plan order. Returns float32.

    ``round_between`` rounds each intermediate result to that dtype, the way
    the engine writes every step back into the parameter's storage dtype; the
    bake leaves it ``None`` and keeps full precision until the final requant.
    """
    out = W.to(torch.float32)
    for i, edit in enumerate(edits):
        out = apply_tensor_edit(out, edit)
        if round_between is not None and i < len(edits) - 1:
            out = out.to(round_between).to(torch.float32)
    return out


def _normalise(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / torch.linalg.vector_norm(v).clamp(min=eps)


# ---------------------------------------------------------------------------
# Plan recording (engine → list[TensorEdit])
# ---------------------------------------------------------------------------


def _layer_strength(sp, layer_idx: int, kernel: DecayKernel) -> float | None:
    """Reproduce the per-layer decay used by _apply_direct/_apply_ega_steering.

    Returns ``None`` when this layer is out of the profile's falloff window or
    resolves to exactly zero strength (both are skips in the engine).
    """
    distance = abs(layer_idx - sp.max_weight_position)
    if distance > sp.min_weight_distance:
        return None
    t = distance / sp.min_weight_distance if sp.min_weight_distance else 0.0
    if kernel == DecayKernel.GAUSSIAN:
        strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
            -2.0 * t * t
        )
    elif kernel == DecayKernel.COSINE:
        strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
            0.5 * (1.0 + math.cos(math.pi * t))
        )
    else:  # LINEAR
        strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)
    if strength == 0:
        return None
    return float(strength)


def normalize_param_name(name: str) -> str:
    """Strip PEFT decorations so an in-memory name matches the on-disk name.

    ``base_model.model.model.layers.0.self_attn.o_proj.base_layer.weight`` →
    ``model.layers.0.self_attn.o_proj.weight``. Idempotent on already-canonical
    names.
    """
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    return name.replace(".base_layer.", ".")


def _steering_vector_for(
    steering_vectors: Tensor, global_vector: Tensor | None, layer_idx: int
) -> Tensor:
    if global_vector is not None:
        return global_vector
    return steering_vectors[layer_idx + 1]


def record_steering_plan(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict,
    config,
    discriminative_layers: set[int] | None = None,
    safety_experts: dict[int, list[tuple[int, float]]] | None = None,
    routing_config=None,
) -> list[TensorEdit]:
    """Record the edits the engine would apply for the best trial.

    Mirrors the walk of ``core.steering._apply_direct_steering``,
    ``_apply_ega_steering`` and, when ``safety_experts`` and
    ``routing_config`` are given, ``_apply_moe_steering`` (router suppression
    and the safety-expert projection), in that order, but emits
    :class:`TensorEdit` records instead of mutating weights. Call with the
    SAME arguments those functions receive.

    The model must be loaded (this uses ``engine.steerable_modules`` /
    ``engine._locate_fused_weights`` and the parameter-name maps). It does not
    depend on the weights being any particular precision — the recorded plan is
    then replayed against the original FP4 checkpoint on disk.
    """
    kernel = config.steering.decay_kernel
    preserve = config.steering.weight_normalization != WeightNorm.NONE
    direct_transform = getattr(
        config.steering, "direct_transform", DirectTransform.STANDARD
    )
    transposed = bool(getattr(engine, "_fused_down_proj_transposed", False))

    model = engine.model
    name_by_param_id = {id(p): n for n, p in model.named_parameters()}

    edits: list[TensorEdit] = []
    n_layers = len(engine.transformer_layers)

    # --- Direct (per steerable Linear module) ---
    for layer_idx in range(n_layers):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue
        for component, modules in engine.steerable_modules(layer_idx).items():
            sp = profiles.get(component)
            if sp is None:
                continue
            strength = _layer_strength(sp, layer_idx, kernel)
            if strength is None:
                continue
            v = _steering_vector_for(steering_vectors, global_vector, layer_idx)
            if v.ndim != 1:
                # Multi-direction subspace direct steering is not represented
                # by a single TensorEdit; skip (caller handles rank-k elsewhere).
                continue
            vf = v.to(torch.float32)
            for mod in modules:
                base_mod = getattr(mod, "base_layer", mod)
                weight = getattr(base_mod, "weight", None)
                if weight is None:
                    continue
                raw_name = name_by_param_id.get(id(weight))
                if raw_name is None:
                    continue
                out_f, in_f = weight.shape[0], weight.shape[1]
                # Engine precedence: output-side first, then input-side.
                if vf.shape[0] == out_f:
                    side = "output"
                elif vf.shape[0] == in_f:
                    side = "input"
                else:
                    continue  # matches neither axis → engine skips it
                edits.append(
                    TensorEdit(
                        logical_name=normalize_param_name(raw_name),
                        kind="direct",
                        direction=vf.detach().cpu().clone(),
                        strength=strength,
                        preserve_row_norm=preserve,
                        projection_side=side,
                        direct_transform=DirectTransform(direct_transform).value,
                    )
                )

    # --- EGA (fused expert down_proj) ---
    if engine.has_expert_routing():
        sp = profiles.get("mlp.down_proj")
        if sp is not None:
            for layer_idx in range(n_layers):
                if (
                    discriminative_layers is not None
                    and layer_idx not in discriminative_layers
                ):
                    continue
                layer = engine.transformer_layers[layer_idx]
                fused = engine._locate_fused_weights(layer)
                if fused is None:
                    continue
                strength = _layer_strength(sp, layer_idx, kernel)
                if strength is None:
                    continue
                v = _steering_vector_for(steering_vectors, global_vector, layer_idx)
                if v.ndim != 1:
                    continue
                raw_name = name_by_param_id.get(id(fused))
                if raw_name is None:
                    continue
                axis_is_in = resolve_ega_axis(
                    tuple(fused.shape), v.shape[0], transposed=transposed
                )
                if axis_is_in is None:
                    continue
                edits.append(
                    TensorEdit(
                        logical_name=normalize_param_name(raw_name),
                        kind="ega",
                        direction=v.to(torch.float32).detach().cpu().clone(),
                        strength=strength,
                        preserve_row_norm=preserve,
                        hidden_dim=int(v.shape[0]),
                        transposed=transposed,
                    )
                )

    # --- MoE steering (router suppression + safety-expert projection) ---
    # No discriminative-layer filter: the engine applies it to every layer
    # with profiled safety experts.
    if safety_experts and routing_config is not None:
        n_suppress = routing_config.n_suppress
        bias = routing_config.router_bias
        expert_w = routing_config.expert_ablation_weight
        for layer_idx in range(n_layers):
            top = safety_experts.get(layer_idx, [])[:n_suppress]
            if not top:
                continue
            v = _steering_vector_for(steering_vectors, global_vector, layer_idx)
            layer = engine.transformer_layers[layer_idx]
            gate = engine._locate_router(layer)
            if gate is not None and bias < 0:
                raw_name = name_by_param_id.get(id(gate.weight))
                if raw_name is not None:
                    edits.append(
                        TensorEdit(
                            logical_name=normalize_param_name(raw_name),
                            kind="row_scale",
                            direction=None,
                            strength=0.0,
                            preserve_row_norm=False,
                            rows=[int(eid) for eid, _ in top],
                            scale=max(0.0, 1.0 + bias / 10.0),
                        )
                    )
            fused = engine._locate_fused_weights(layer)
            if fused is None or expert_w <= 0:
                continue
            raw_name = name_by_param_id.get(id(fused))
            if raw_name is None:
                continue
            if (
                resolve_ega_axis(tuple(fused.shape), v.shape[0], transposed=transposed)
                is None
            ):
                continue
            for eid, _ in top:
                edits.append(
                    TensorEdit(
                        logical_name=normalize_param_name(raw_name),
                        kind="expert_project",
                        direction=v.to(torch.float32).detach().cpu().clone(),
                        strength=float(expert_w),
                        preserve_row_norm=False,
                        hidden_dim=int(v.shape[0]),
                        transposed=transposed,
                        expert=int(eid),
                    )
                )

    return edits


def build_per_expert_ega_plan(
    steering_vectors: Tensor,
    profiles: dict,
    *,
    n_layers: int,
    n_experts: int,
    hidden_dim: int,
    kernel: DecayKernel = DecayKernel.LINEAR,
    preserve_row_norm: bool = True,
    name_template: str = "layers.{layer}.ffn.experts.{expert}.w2",
    global_vector: Tensor | None = None,
    discriminative_layers: set[int] | None = None,
    component: str = "mlp.down_proj",
) -> list[TensorEdit]:
    """Build an EGA plan for checkpoints that store one tensor per expert.

    :func:`record_steering_plan` walks a *loaded* model and finds one fused 3-D
    expert parameter per layer. Producers like DeepSeek-V4-Flash instead store
    ``layers.{L}.ffn.experts.{N}.w2`` — 43 x 256 separate 2-D tensors — which
    is far too many to hand-write and needs no model to enumerate. This emits
    one :class:`TensorEdit` per (layer, expert) using the same per-layer decay
    and the same projection kernel as the fused path, so the two produce the
    same abliteration for the same profile.

    ``w2`` is the down-projection in the w1/w2/w3 convention (w1 gate, w3 up),
    i.e. the tensor EGA targets. ``name_template`` is formatted with ``layer``
    and ``expert``.
    """
    sp = profiles.get(component)
    if sp is None:
        return []
    edits: list[TensorEdit] = []
    for layer_idx in range(n_layers):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue
        strength = _layer_strength(sp, layer_idx, kernel)
        if strength is None:
            continue
        v = _steering_vector_for(steering_vectors, global_vector, layer_idx)
        if v.ndim != 1:
            continue
        vf = v.to(torch.float32).detach().cpu()
        for expert_idx in range(n_experts):
            edits.append(
                TensorEdit(
                    logical_name=name_template.format(
                        layer=layer_idx, expert=expert_idx
                    ),
                    kind="ega",
                    direction=vf.clone(),
                    strength=strength,
                    preserve_row_norm=preserve_row_norm,
                    hidden_dim=hidden_dim,
                )
            )
    return edits


def record_steering_plan_from_trial(
    engine,
    steering_vectors: Tensor,
    vector_index: float | None,
    profiles: dict,
    config,
    *,
    benign_states: Tensor | None = None,
    target_states: Tensor | None = None,
    safety_experts: dict[int, list[tuple[int, float]]] | None = None,
    routing_config=None,
) -> list[TensorEdit]:
    """Record a plan from the same inputs :func:`apply_steering` receives.

    Resolves the global vector and discriminative-layer selection exactly as
    ``apply_steering`` does (via the shared ``core.steering`` helpers), then
    delegates to :func:`record_steering_plan`. This is the entry point the
    export flow uses: it captures what the best trial *would* edit without
    mutating the model, so the plan can be replayed offline against the FP4
    checkpoint. Pass the trial's ``safety_experts`` and ``routing_config``
    for MoE models, as ``apply_steering`` receives them; without them the
    router suppression and safety-expert projection are not recorded.

    Only direct weight editing with one direction is representable; other
    modes raise rather than record a plan that differs from the model.
    """
    from ..types import SteeringMode
    from .steering import _detect_discriminative_layers, resolve_global_vector

    mode = getattr(config.steering, "steering_mode", SteeringMode.DIRECT)
    if mode != SteeringMode.DIRECT:
        raise ValueError(
            f"a steering plan records direct weight edits; steering_mode "
            f"{getattr(mode, 'value', mode)!r} cannot be baked"
        )
    if steering_vectors.ndim != 2:
        raise ValueError(
            "a steering plan records one direction per layer; multi-direction "
            "steering cannot be baked"
        )
    global_vector = resolve_global_vector(steering_vectors, vector_index)
    discriminative_layers = None
    if config.steering.discriminative_layer_selection:
        discriminative_layers = _detect_discriminative_layers(
            steering_vectors, benign_states, target_states
        )
    return record_steering_plan(
        engine,
        steering_vectors,
        global_vector,
        profiles,
        config,
        discriminative_layers,
        safety_experts=safety_experts,
        routing_config=routing_config,
    )


def plan_meta(engine) -> dict:
    """What a plan records about its source: the in-memory model class whose
    parameter names it uses, so a bake can map them onto a checkpoint."""
    model = engine.model
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return {
        "model_class": type(base).__name__,
        "model_type": getattr(getattr(base, "config", None), "model_type", None),
    }


def group_edits(edits: Iterable[TensorEdit]) -> dict[str, list[TensorEdit]]:
    """Edits per tensor name, each list in plan (= application) order."""
    groups: dict[str, list[TensorEdit]] = {}
    for e in edits:
        groups.setdefault(e.logical_name, []).append(e)
    return groups


class PlanMismatch(AssertionError):
    """A recorded plan does not reproduce what the engine applied."""


@dataclass
class PlanCheck:
    tensors: int = 0
    worst_ulps: float = 0.0


def _ulps_apart(a: Tensor, b: Tensor) -> float:
    """Largest difference between ``a`` and ``b`` in units of ``a``'s dtype eps."""
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    eps = torch.finfo(a.dtype).eps if a.dtype.is_floating_point else 0.0
    if eps == 0.0:
        return 0.0 if torch.equal(a, b) else float("inf")
    tiny = torch.finfo(a.dtype).tiny
    scale = torch.maximum(a32.abs(), b32.abs()) * eps + tiny
    return float(((a32 - b32).abs() / scale).max()) if a.numel() else 0.0


def verify_plan_against_engine(
    engine, edits: list[TensorEdit], *, max_ulps: float = 1.0
) -> PlanCheck:
    """Check a plan against the edits the engine actually applied.

    Call after ``apply_steering`` on a freshly loaded engine. The engine keeps
    the original of everything it edits, for ``restore_baseline``: whole
    tensors in ``_direct_weight_originals``, router rows in
    ``_router_originals``, expert slices in ``_expert_deltas``. The tensors
    changed must be exactly the tensors the plan names, and replaying the
    plan onto each original must reproduce the edited tensor to within
    ``max_ulps`` of its storage dtype (the plan's float32 math against the
    engine's, both rounded to that dtype). Raises :class:`PlanMismatch`.
    """
    model = engine.model
    params: dict[str, Tensor] = {}
    name_of: dict[int, str] = {}
    for raw, p in model.named_parameters():
        name = normalize_param_name(raw)
        params[name] = p
        name_of[id(p)] = name

    def originals_from_whole() -> dict[str, Tensor]:
        found = {}
        for w, orig in getattr(engine, "_direct_weight_originals", {}).items():
            if id(w) not in name_of:
                raise PlanMismatch("an edited tensor is not a model parameter")
            found[name_of[id(w)]] = orig
        return found

    whole = originals_from_whole()
    patches: dict[str, list[tuple[int, Tensor]]] = {}
    layers = engine.transformer_layers
    for layer_idx, eid, row in getattr(engine, "_router_originals", []):
        gate = engine._locate_router(layers[layer_idx])
        patches.setdefault(name_of[id(gate.weight)], []).append((eid, row))
    for layer_idx, eid, piece in getattr(engine, "_expert_deltas", []):
        fused = engine._locate_fused_weights(layers[layer_idx])
        patches.setdefault(name_of[id(fused)], []).append((eid, piece))

    groups = group_edits(edits)
    changed = set(whole) | set(patches)
    if changed != set(groups):
        unplanned = sorted(changed - set(groups))
        unapplied = sorted(set(groups) - changed)
        raise PlanMismatch(
            f"the plan names {len(groups)} tensors, the engine edited "
            f"{len(changed)}; edited but not planned: {unplanned[:5]}, "
            f"planned but not edited: {unapplied[:5]}"
        )

    check = PlanCheck()
    for name, group in groups.items():
        p = params[name]
        if name in whole:
            orig = whole[name].to(p.device)
        else:
            # Restore the patched rows/slices; the first saved copy of a row is
            # its true original, so apply the saves newest first.
            orig = p.data.clone()
            for idx, saved in reversed(patches[name]):
                orig[idx] = saved.to(device=orig.device, dtype=orig.dtype)
        expected = apply_tensor_edits(orig, group, round_between=p.dtype).to(p.dtype)
        ulps = _ulps_apart(expected, p.data)
        del orig, expected
        if ulps > max_ulps:
            raise PlanMismatch(
                f"replaying the plan onto '{name}' differs from the engine's "
                f"edit by {ulps:.1f} ulps ({[e.kind for e in group]})"
            )
        check.tensors += 1
        check.worst_ulps = max(check.worst_ulps, ulps)
    return check


# ---------------------------------------------------------------------------
# On-disk FP4 tensor layout
# ---------------------------------------------------------------------------


@dataclass
class _Fp4KeySet:
    """The safetensors keys that together encode one FP4 logical weight."""

    blocks: str  # packed nibble uint8
    scales: str  # per-block scale (ue8m0 uint8 for MXFP4, e4m3 for NVFP4)
    global_scale: str | None = None  # per-tensor fp32 (NVFP4 only)


# ---------------------------------------------------------------------------
# On-disk layouts
#
# The *element* format (E2M1 + block scale) is shared, but every producer wraps
# it differently: key suffixes, packed rank, and whether the model transposes
# the decoded tensor. Declare those differences once here instead of
# special-casing them at each call site.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fp4Layout:
    """How one producer stores an FP4 tensor on disk.

    ``weight_suffix`` / ``scale_suffix`` are appended to the *logical* tensor
    name to get the safetensors keys. ``packed_ndim`` is the rank of the stored
    packed tensor: 4 for gpt-oss's fused ``(E, rows, n_blocks, bytes)``, 2 for
    DeepSeek's flat per-expert ``(out, in/2)``. ``model_transposes`` records
    that the model's own dequant ends with ``transpose(1, 2)`` (gpt-oss only).
    """

    name: str
    weight_suffix: str
    scale_suffix: str
    packed_ndim: int
    model_transposes: bool
    global_suffixes: tuple[str, ...] = ()


# gpt-oss / transformers MXFP4: `<name>_blocks` (E, rows, n_blocks, 16) uint8
# + `<name>_scales` (E, rows, n_blocks) uint8. Reference dequant transposes.
GPT_OSS_LAYOUT = Fp4Layout("gpt_oss", "_blocks", "_scales", 4, True)

# DeepSeek-V4-Flash: `<name>.weight` I8 (out, in/2) + `<name>.scale` F8_E8M0
# (out, in/32) — flat 2-D, one tensor per expert, no model-side transpose.
# Measured on the real checkpoint; E2M1 + power-of-two E8M0 scales, block 32,
# i.e. the same MXFP4 math fp4_utils already decodes bit-exactly.
DEEPSEEK_V4_LAYOUT = Fp4Layout("deepseek_v4", ".weight", ".scale", 2, False)

# NVFP4 as ModelOpt exports it: `<name>.weight` U8 (out, in/2), e4m3
# `<name>.weight_scale` (out, in/16) and an fp32 `<name>.weight_scale_2` that
# multiplies (dequant = code * block scale * global). Checked against
# nvidia/GLM-5.3-Flash-NVFP4: an expert decodes to within 9% of the BF16
# release, the same error as a fresh NVFP4 quant. llm-compressor's layout
# (`weight_packed`, and a `weight_global_scale` that divides) is different and
# deliberately not matched.
NVFP4_LAYOUT = Fp4Layout(
    "nvfp4",
    "",
    "_scale",
    2,
    False,
    ("_scale_2", "_weight_scale_2"),
)

_ALL_LAYOUTS = (GPT_OSS_LAYOUT, DEEPSEEK_V4_LAYOUT, NVFP4_LAYOUT)


def detect_layout(present: set[str], fmt: f4.Fp4Format) -> Fp4Layout | None:
    """Infer the on-disk layout from the keys actually in the checkpoint."""
    if any(k.endswith("_blocks") for k in present):
        return GPT_OSS_LAYOUT
    if fmt.kind == "nvfp4":
        return NVFP4_LAYOUT
    if any(k.endswith(".scale") for k in present):
        return DEEPSEEK_V4_LAYOUT
    return None


def resolve_fp4_keys(
    logical_name: str,
    present: set[str],
    fmt: f4.Fp4Format,
    layout: Fp4Layout | None = None,
    tensors: dict[str, Tensor] | None = None,
) -> _Fp4KeySet | None:
    """Find the FP4 sibling keys for ``logical_name`` among ``present`` keys.

    ``layout`` defaults to trying every known producer, so callers that do not
    care (tests, one-off tools) keep working. Returns ``None`` if the tensor is
    not FP4-encoded under any layout — e.g. a plain BF16 weight that should be
    edited without repacking.

    ``tensors`` disambiguates by dtype when supplied, which a mixed checkpoint
    needs: DeepSeek-V4 names both its 4-bit experts and its block-wise **FP8**
    attention ``<name>.weight`` + ``<name>.scale``, so key names alone would
    hand an e4m3 tensor to the nibble decoder. Packed FP4 is always an integer
    dtype; float8 storage never is.
    """
    for lay in (layout,) if layout is not None else _ALL_LAYOUTS:
        w = logical_name + lay.weight_suffix
        s = logical_name + lay.scale_suffix
        if w in present and s in present:
            if tensors is not None:
                wt = tensors.get(w)
                if wt is not None and wt.dtype.is_floating_point:
                    continue  # FP8 (or bf16) storage, not packed nibbles
            g = None
            for gsuf in lay.global_suffixes:
                if logical_name + gsuf in present:
                    g = logical_name + gsuf
                    break
            return _Fp4KeySet(w, s, g)
    return None


def resolve_fp8_keys(
    logical_name: str, present: set[str], tensors: dict[str, Tensor]
) -> tuple[str, str] | None:
    """Find ``(weight_key, scale_key)`` if ``logical_name`` is block-wise FP8.

    ``logical_name`` may already be the weight key (``...wo_b.weight``) or the
    module prefix (``...wo_b``). Returns ``None`` when the tensor is not FP8, so
    the caller falls through to the FP4 or dense path.
    """
    w_key = logical_name if logical_name in present else f"{logical_name}.weight"
    if w_key not in present:
        return None
    w = tensors.get(w_key)
    if w is None or w.dtype not in f8._FP8_DTYPES or w.dim() != 2:
        return None
    prefix = w_key.rsplit(".", 1)[0] if "." in w_key else w_key
    for suffix in f8._SCALE_ATTRS:
        s_key = f"{prefix}.{suffix}"
        if s_key in present and tensors.get(s_key) is not None:
            if tensors[s_key].dim() == 2:
                return w_key, s_key
    return None


def _as_blocked(blocks: Tensor, scales: Tensor) -> Tensor:
    """Reshape a flat packed weight so its last axis is one block's bytes.

    gpt-oss already stores ``(..., n_blocks, bytes)``. DeepSeek stores the
    packed weight flat as ``(out, in/2)`` with scales ``(out, n_blocks)``, so
    split the byte axis into ``(out, n_blocks, bytes_per_block)`` before the
    kernel sees it. Detected from the shapes, not hard-coded per producer.
    """
    if blocks.dim() == scales.dim() + 1:
        return blocks  # already blocked: scales index every axis but the bytes
    n_blocks = scales.shape[-1]
    if blocks.shape[-1] % n_blocks != 0:
        raise ValueError(
            f"packed last axis {blocks.shape[-1]} not divisible by "
            f"{n_blocks} blocks — layout mismatch"
        )
    return blocks.reshape(*blocks.shape[:-1], n_blocks, blocks.shape[-1] // n_blocks)


def _as_flat(blocks: Tensor, like: Tensor) -> Tensor:
    """Inverse of :func:`_as_blocked`: restore the producer's stored shape."""
    return blocks.reshape(like.shape) if blocks.shape != like.shape else blocks


def _store_as(t: Tensor, like: Tensor) -> Tensor:
    """Bit-preserving cast of a ``uint8`` payload back to the producer's dtype.

    Packed nibbles and E8M0 scales are **byte payloads, not numbers**. DeepSeek
    stores them as ``int8`` / ``float8_e8m0fnu``, so ``.to(dtype)`` would
    *convert*: any packed byte above 127 (i.e. every pair whose high nibble is
    negative) would clamp to 127, and a scale byte would be reinterpreted as a
    float value. Bit-cast instead.
    """
    if t.dtype == like.dtype:
        return t
    return t.contiguous().view(like.dtype)


def _scale_bytes(scales: Tensor) -> Tensor:
    """View a block-scale tensor as the ``uint8`` biased exponent the kernel wants.

    DeepSeek stores E8M0 scales as ``torch.float8_e8m0fnu``; its bit pattern is
    exactly the ue8m0 biased exponent (value ``2**(byte-127)``) that
    :func:`fp4_utils.dequantize_mxfp4` already expects, so this is a bit-cast,
    not a conversion.
    """
    return scales if scales.dtype == torch.uint8 else scales.view(torch.uint8)


def _packed_moe_is_transposed(blocks: Tensor) -> bool:
    """True when a packed FP4 tensor stores the MoE layout the model transposes.

    **Verified against transformers 5.9's reference dequant**
    (``integrations.mxfp4._convert_moe_packed_tensors``): a 4-D packed tensor
    ``(E, rows, n_blocks, bytes)`` decodes to ``(E, rows, K)`` element-wise and
    is then ``.transpose(1, 2)``-ed, so the weight the *model* sees is
    ``(E, K, rows)``. Our element decoding is bit-identical to the reference;
    only this final axis swap is model-side.

    That matters because the steering plan records axis semantics against the
    **in-memory** tensor. Without undoing the swap, an EGA edit replayed
    offline would project the wrong axis — silently, and undetectably on
    gpt-oss where hidden == intermediate makes both axes the same length.

    2-D/3-D packed tensors (plain Linear weights) carry no such swap.
    """
    return blocks.dim() == 4


def _to_logical(W: Tensor, transposed: bool) -> Tensor:
    """On-disk orientation → the orientation the model/plan uses."""
    return W.transpose(1, 2).contiguous() if transposed else W


def _to_ondisk(W: Tensor, transposed: bool) -> Tensor:
    """Inverse of :func:`_to_logical`."""
    return W.transpose(1, 2).contiguous() if transposed else W


def _dequant_fp4_tensor(
    tensors: dict[str, Tensor], keys: _Fp4KeySet, fmt: f4.Fp4Format
) -> Tensor:
    """Dequantise an on-disk FP4 tensor into the model's logical orientation.

    Siblings are pulled onto ``blocks``' device: a shard's tensors may be a
    mix of freshly-repacked GPU tensors and untouched CPU ones. MXFP4 scales
    are exponent bytes; NVFP4 scales are e4m3 values and stay as they are.
    """
    stored = tensors[keys.blocks]
    scales = tensors[keys.scales].to(stored.device)
    if fmt.kind == "mxfp4":
        scales = _scale_bytes(scales)
    blocks = _as_blocked(stored, scales)
    g = tensors[keys.global_scale].to(stored.device) if keys.global_scale else None
    W = f4.dequantize_fp4(fmt, blocks, scales, global_scale=g, out_dtype=torch.float32)
    return _to_logical(W, _packed_moe_is_transposed(stored))


def _requant_fp4_tensor(
    W: Tensor,
    keys: _Fp4KeySet,
    fmt: f4.Fp4Format,
    orig: dict[str, Tensor],
    scale_search: int = 0,
) -> dict[str, Tensor]:
    """Re-pack ``W`` (float) into the same FP4 key layout.

    ``W`` is in the model's **logical** orientation (as produced by
    :func:`_dequant_fp4_tensor` and edited); it is rotated back to the on-disk
    orientation here so the written bytes keep the producer's convention.

    NVFP4 keeps the source tensor's global scale so block scales stay in the
    e4m3 range the producer chose. ``scale_search`` enables the MSE-optimal
    MXFP4 block-scale search (see :func:`fp4_utils.quantize_to_mxfp4`).
    """
    g = orig.get(keys.global_scale) if keys.global_scale else None
    if g is not None:
        g = g.to(W.device)  # plan/shard tensors are CPU; W may be on GPU
    stored = orig[keys.blocks]
    W = _to_ondisk(W, _packed_moe_is_transposed(stored))
    blocks, scales, new_g = f4.quantize_fp4(
        fmt, W, global_scale=g, scale_search=scale_search
    )
    # Restore the producer's stored shape (flat for DeepSeek) and byte dtype.
    out: dict[str, Tensor] = {
        keys.blocks: _store_as(_as_flat(blocks, stored), stored),
        keys.scales: _store_as(scales, orig[keys.scales]),
    }
    if keys.global_scale is not None:
        out[keys.global_scale] = (
            new_g if new_g is not None else orig[keys.global_scale]
        ).to(orig[keys.global_scale].dtype)
    return out


def _assert_layout_roundtrip(
    src: dict[str, Tensor],
    keys: _Fp4KeySet,
    fmt: f4.Fp4Format,
    *,
    name: str = "<tensor>",
) -> None:
    """Verify the FULL repack path is a fixed point on unedited weights.

    Stronger than the kernel-level check in :mod:`fp4_utils`: this runs the
    producer's real bytes through dequant → requant → dequant, so a wrong
    reshape (flat vs blocked), a lossy dtype cast (int8 clamping a packed byte
    above 127), or a bad transpose shows up as value drift here rather than as
    silently corrupted weights on disk.

    NVFP4 is checked against the producer's own block scales: re-deriving them
    from decoded values is not a fixed point (an e4m3-rounded scale rarely
    leaves a block's largest code at the top of the E2M1 range), so encoding
    with the stored scales must reproduce the stored bytes exactly instead.
    """
    if fmt.kind == "nvfp4":
        _assert_nvfp4_layout_roundtrip(src, keys, fmt, name=name)
        return
    W0 = _dequant_fp4_tensor(src, keys, fmt)
    re = _requant_fp4_tensor(W0, keys, fmt, src, scale_search=0)
    W1 = _dequant_fp4_tensor({**src, **re}, keys, fmt)
    if not torch.equal(W0, W1):
        drift = (W0 - W1).abs().max().item()
        raise AssertionError(
            f"FP4 repack of '{name}' is not a fixed point on unedited weights "
            f"(max drift {drift:g}) — the on-disk layout handling (shape, byte "
            "dtype, or orientation) does not match this producer."
        )


def _assert_nvfp4_layout_roundtrip(
    src: dict[str, Tensor], keys: _Fp4KeySet, fmt: f4.Fp4Format, *, name: str
) -> None:
    """Decode, re-encode with the stored scales, and compare the codes.

    Producers may write a tiny negative value as -0 (code 8); the encoder
    canonicalises zero to +0, so the stored side is canonicalised the same way.
    """
    stored = src[keys.blocks]
    scales = src[keys.scales].to(stored.device)
    g = src[keys.global_scale].to(device=stored.device, dtype=torch.float32)
    W = _to_ondisk(
        _dequant_fp4_tensor(src, keys, fmt), _packed_moe_is_transposed(stored)
    )
    blocked = W.reshape(*scales.shape, fmt.block_size)
    eff = (scales.to(torch.float32) * g).unsqueeze(-1)
    codes = f4.quantize_to_e2m1_codes(blocked / torch.where(eff == 0, 1.0, eff))
    original = f4.unpack_nibbles(_as_blocked(stored.view(torch.uint8), scales))
    original = torch.where(original == 8, torch.zeros_like(original), original)
    if not torch.equal(codes, original):
        bad = int((codes != original).sum())
        raise AssertionError(
            f"NVFP4 re-encode of '{name}' with its own scales changes {bad} "
            "codes — the on-disk layout handling (shape, byte dtype, nibble "
            "order, or orientation) does not match this producer."
        )


# ---------------------------------------------------------------------------
# Streaming bake
# ---------------------------------------------------------------------------


@dataclass
class RepackStats:
    tensors_written: int = 0
    fp4_edited: int = 0
    fp8_edited: int = 0
    dense_edited: int = 0
    copied: int = 0
    skipped_edits: list[str] = field(default_factory=list)
    unmatched_edits: list[str] = field(default_factory=list)
    requant_rel_err: dict[str, float] = field(default_factory=dict)


def _iter_shards(src_dir: Path) -> dict[str, list[str]]:
    """Return ``{shard_filename: [tensor_key, ...]}`` for the source model."""
    idx_path = src_dir / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text())["weight_map"]
    else:
        from safetensors import safe_open

        weight_map = {}
        for s in sorted(src_dir.glob("*.safetensors")):
            with safe_open(s, framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = s.name
    shards: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        shards.setdefault(fname, []).append(key)
    return shards


# safetensors header dtype names → torch dtypes, for reading a checkpoint's
# layout without loading it. Unknown names map to uint8 (a byte payload), which
# the encoding resolver treats as not editable in place.
_ST_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
for _st, _attr in (
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
    ("F8_E8M0", "float8_e8m0fnu"),
):
    if hasattr(torch, _attr):
        _ST_DTYPES[_st] = getattr(torch, _attr)


def _checkpoint_layout(
    src_dir: Path, shards: dict[str, list[str]]
) -> dict[str, Tensor]:
    """Every tensor key → a meta tensor of its stored shape and dtype."""
    from safetensors import safe_open

    meta: dict[str, Tensor] = {}
    for fname, keys in shards.items():
        with safe_open(src_dir / fname, framework="pt") as f:
            for k in keys:
                sl = f.get_slice(k)
                dtype = _ST_DTYPES.get(sl.get_dtype(), torch.uint8)
                meta[k] = torch.empty(sl.get_shape(), dtype=dtype, device="meta")
    return meta


@dataclass
class _Encoding:
    """How one edited tensor is stored: its keys (primary first) and format."""

    kind: str  # "fp4" | "fp8" | "dense"
    keys: tuple[str, ...]
    fp4: _Fp4KeySet | None = None
    fmt: f4.Fp4Format | None = None
    fp8: tuple[str, str] | None = None


def _infer_fp4_format(meta: dict[str, Tensor], keys: _Fp4KeySet) -> f4.Fp4Format:
    """The FP4 element format, from the stored tensors themselves.

    An e4m3 block scale is NVFP4, a byte (or e8m0) one MXFP4; the block size is
    the elements one scale covers. Reading it off the tensor rather than the
    config handles mixed checkpoints and configs that name no FP4 method
    (DeepSeek-V4 ships ``quant_method: fp8`` with FP4 experts).
    """
    blocks, scales = meta[keys.blocks], meta[keys.scales]
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    kind = "nvfp4" if scales.dtype == e4m3 else "mxfp4"
    if blocks.dim() == scales.dim() + 1:
        bytes_per_block = blocks.shape[-1]
    else:
        bytes_per_block = blocks.shape[-1] // max(1, scales.shape[-1])
    if kind == "nvfp4" and keys.global_scale is None:
        raise ValueError(
            f"'{keys.blocks}' has e4m3 block scales but no per-tensor global "
            "scale; this NVFP4 layout is not supported"
        )
    return f4.Fp4Format(kind, int(bytes_per_block) * 2)


def _resolve_encoding(
    name: str,
    meta: dict[str, Tensor],
    fmt: f4.Fp4Format | None = None,
    layout: Fp4Layout | None = None,
) -> _Encoding | None:
    """Find how the tensor an edit names is stored, across all shards.

    ``name`` may be a parameter name (``...o_proj.weight``) or a module prefix
    (``...experts.3.w2``). Returns ``None`` when no key matches. Raises when
    the tensor exists but its storage cannot be edited safely: packed or FP8
    bytes without a recognised scale, which a dense edit would corrupt.
    """
    present = set(meta)
    candidates = [name]
    if name.endswith(".weight"):
        candidates.append(name[: -len(".weight")])
    for cand in candidates:
        keys = resolve_fp4_keys(cand, present, fmt, layout, tensors=meta)
        if keys is not None:
            found = fmt or _infer_fp4_format(meta, keys)
            all_keys = (keys.blocks, keys.scales) + (
                (keys.global_scale,) if keys.global_scale else ()
            )
            return _Encoding("fp4", all_keys, fp4=keys, fmt=found)
    for cand in candidates:
        fp8_keys = resolve_fp8_keys(cand, present, meta)
        if fp8_keys is not None:
            return _Encoding("fp8", fp8_keys, fp8=fp8_keys)
    for cand in (name, f"{name}.weight"):
        if cand not in present:
            continue
        dtype = meta[cand].dtype
        if dtype in f8._FP8_DTYPES:
            # An FP8 weight whose scale sibling we could not find. The dense
            # path would project raw e4m3 values as if they were the real
            # weights — silently wrong. Refuse.
            raise ValueError(
                f"'{cand}' is FP8 but no scale sibling was found among "
                f"{f8._SCALE_ATTRS}; editing it as a dense tensor would "
                "ignore the block scaling and corrupt the weight."
            )
        if not dtype.is_floating_point:
            raise ValueError(
                f"'{cand}' is stored as {dtype} with no recognised scale "
                "sibling; it is packed quantised data, and editing it as a "
                "dense tensor would corrupt it."
            )
        return _Encoding("dense", (cand,))
    return None


def _edit_encoded(
    enc: _Encoding,
    group: list[TensorEdit],
    tensors: dict[str, Tensor],
    device: str,
    *,
    verify_idempotent: bool,
    reference_dequant: Any,
    scale_search: int,
) -> tuple[dict[str, Tensor], float | None]:
    """Dequantise, edit and re-encode one tensor. Returns every key of its
    encoding (on CPU) and the requant relative error, if it was requantised.
    Raises ``ValueError`` when an edit does not fit the tensor."""
    name = enc.keys[0]
    if enc.kind == "fp4":
        keys, fmt = enc.fp4, enc.fmt
        W = _dequant_fp4_tensor(tensors, keys, fmt).to(device)
        if verify_idempotent:
            _assert_layout_roundtrip(tensors, keys, fmt, name=name)
        if reference_dequant is not None:
            ref = reference_dequant(group[0].logical_name, W)
            if ref is not None:
                f4.assert_matches_reference(W, ref.to(device), name=name)
        W_new = apply_tensor_edits(W, group)
        repacked = _requant_fp4_tensor(
            W_new, keys, fmt, tensors, scale_search=scale_search
        )
        # Requant round-trip error (steered vs re-decoded), for the log.
        W_chk = _dequant_fp4_tensor({**tensors, **repacked}, keys, fmt).to(device)
        err = float((W_chk - W_new).abs().mean() / W_new.abs().mean().clamp(min=1e-9))
        return {k: v.cpu() for k, v in repacked.items()}, err
    if enc.kind == "fp8":
        # Block-wise FP8 (DeepSeek-V4 attention + shared experts). Editing this
        # through the dense branch would project the raw e4m3 values with the
        # block scale ignored, so it gets its own dequant -> project -> requant,
        # preserving the stored dtypes.
        w_key, s_key = enc.fp8
        w, s = tensors[w_key].to(device), tensors[s_key].to(device)
        W = f8.dequant_blockwise(w, s, is_inv=True, out_dtype=torch.float32)
        if verify_idempotent:
            f8.assert_fp8_repack_idempotent(w, s, name=name)
        W_new = apply_tensor_edits(W, group)
        br = max(1, w.shape[0] // s.shape[0])
        bc = max(1, w.shape[1] // s.shape[1])
        new_w, new_s = f8.quantize_to_fp8_blockwise(
            W_new, (br, bc), weight_dtype=w.dtype, scale_dtype=s.dtype
        )
        chk = f8.dequant_blockwise(new_w, new_s, is_inv=True, out_dtype=torch.float32)
        err = float((chk - W_new).abs().mean() / W_new.abs().mean().clamp(min=1e-9))
        return {w_key: new_w.cpu(), s_key: new_s.cpu()}, err
    stored = tensors[name]
    W_new = apply_tensor_edits(stored.to(device), group)
    return {name: W_new.to(stored.dtype).cpu()}, None


def abliterate_fp4_to_disk(
    src_dir: str | Path,
    dst_dir: str | Path,
    edits: Iterable[TensorEdit],
    *,
    fmt: f4.Fp4Format | None = None,
    layout: Fp4Layout | None = None,
    use_cuda: bool = True,
    verify_idempotent: bool = True,
    reference_dequant: Any = None,
    scale_search: int = 1,
    strict: bool = True,
    verbose: bool = True,
) -> RepackStats:
    """Replay ``edits`` against a checkpoint, streaming shard by shard.

    Edited FP4 tensors are dequantised, projected, and re-packed into the same
    FP4 layout; block-wise FP8 tensors likewise; edited plain tensors (e.g.
    gpt-oss ``attn.o_proj``, which is not quantised) are projected and written
    back in their storage dtype; every other tensor is copied through
    unchanged. ``config.json`` is kept as is, because the output keeps the
    source's precision.

    Each edit is resolved against the whole checkpoint first, from the shard
    headers: a tensor's packed weight and its scales may sit in different
    shards (ModelOpt exports do this), and each piece is written back to the
    shard it came from. Several edits on one tensor apply in plan order.

    Parameters
    ----------
    fmt
        FP4 format override; by default each FP4 tensor's format is read from
        its own scale dtype and shape.
    verify_idempotent
        Cross-check each source FP4 tensor's pack/unpack self-consistency
        before editing (cheap; catches an internal kernel bug).
    reference_dequant
        Optional callable ``(logical_name, our_W) -> ref_W | None``. When it
        returns a tensor, fp4_utils' dequant is asserted to match it — the real
        layout-faithfulness guard against a producer whose nibble order or
        scale encoding differs from what fp4_utils assumes.
    scale_search
        MSE-optimal MXFP4 block-scale search depth for re-packing edited
        tensors (default 1 = try the amax-tight exponent and one tighter). 0
        restores the plain amax scale. Higher trades a little bake time for
        lower requant error; ignored for NVFP4. See
        :func:`fp4_utils.quantize_to_mxfp4`.
    strict
        Raise, before anything is written, when an edit names no tensor in the
        checkpoint, and raise when an edit does not fit its tensor. With
        ``strict=False`` those edits are skipped and listed in the stats; the
        output then lacks part of the plan.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    cfg = json.loads((src_dir / "config.json").read_text())

    stats = RepackStats()
    shards = _iter_shards(src_dir)
    shard_of = {k: fname for fname, keys in shards.items() for k in keys}
    meta = _checkpoint_layout(src_dir, shards)

    resolved: dict[str, tuple[_Encoding, list[TensorEdit]]] = {}
    for name, group in group_edits(edits).items():
        enc = _resolve_encoding(name, meta, fmt, layout)
        if enc is None:
            stats.unmatched_edits.append(name)
        else:
            resolved[name] = (enc, group)
    if stats.unmatched_edits and strict:
        raise ValueError(
            f"{len(stats.unmatched_edits)} planned edits match no tensor in "
            f"{src_dir} (first few: {sorted(stats.unmatched_edits)[:3]}); "
            "map the plan's names onto this checkpoint first"
        )
    if verbose:
        formats = sorted(
            {
                f"{e.fmt.kind}/{e.fmt.block_size}" if e.fmt else e.kind
                for e, _ in resolved.values()
            }
        )
        print(f"* {len(resolved)} tensors to edit ({', '.join(formats)})")

    def read(key: str, loaded: dict[str, Tensor]) -> Tensor:
        if key in loaded:
            return loaded[key]
        with safe_open(src_dir / shard_of[key], framework="pt") as f:
            return f.get_tensor(key)

    dst_dir.mkdir(parents=True, exist_ok=True)
    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    t0 = time.time()
    skipped: set[str] = set()

    for i, (fname, keys) in enumerate(sorted(shards.items()), 1):
        with safe_open(src_dir / fname, framework="pt") as f:
            src = {k: f.get_tensor(k) for k in keys}

        out: dict[str, Tensor] = {}
        for name, (enc, group) in resolved.items():
            here = [k for k in enc.keys if shard_of[k] == fname]
            if not here or name in skipped:
                continue
            # A tensor split across shards is re-derived for each shard it
            # touches (deterministically), which keeps memory to one tensor.
            tensors = {k: read(k, src) for k in enc.keys}
            try:
                edited, err = _edit_encoded(
                    enc,
                    group,
                    tensors,
                    device,
                    verify_idempotent=verify_idempotent,
                    reference_dequant=reference_dequant,
                    scale_search=scale_search,
                )
            except ValueError as e:
                if strict:
                    raise
                if verbose:
                    print(f"  [yellow]skip {enc.kind} edit {name}: {e}[/]")
                stats.skipped_edits.append(name)
                skipped.add(name)
                continue
            for k in here:
                out[k] = edited[k]
            if shard_of[enc.keys[0]] == fname:
                setattr(
                    stats,
                    f"{enc.kind}_edited",
                    getattr(stats, f"{enc.kind}_edited") + 1,
                )
                if err is not None:
                    stats.requant_rel_err[name] = err
            del tensors, edited

        # Copy through everything not produced by an edit.
        for k in keys:
            if k not in out:
                out[k] = src[k]
                stats.copied += 1

        save_file(out, str(dst_dir / fname), metadata={"format": "pt"})
        for k in out:
            new_weight_map[k] = fname
        total_bytes += (dst_dir / fname).stat().st_size
        if verbose:
            print(
                f"[{i}/{len(shards)}] {fname}: "
                f"{stats.fp4_edited} fp4-edited, {stats.fp8_edited} fp8-edited, "
                f"{stats.dense_edited} dense-edited "
                f"({total_bytes / 1e9:.1f} GB, {(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        del src, out
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    stats.tensors_written = len(new_weight_map)

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map},
            indent=2,
        )
    )
    # Keep config.json AS-IS (same precision) but write it out for a standalone dir.
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    _copy_aux_files(src_dir, dst_dir)

    if (stats.unmatched_edits or stats.skipped_edits) and verbose:
        print(
            f"  [yellow]{len(stats.unmatched_edits)} planned edits matched no "
            f"tensor, {len(stats.skipped_edits)} did not fit theirs (first few: "
            f"{sorted(stats.unmatched_edits + stats.skipped_edits)[:3]})[/]"
        )
    if verbose:
        rerr = stats.requant_rel_err
        worst = max(rerr.values()) if rerr else 0.0
        print(
            f"\nDone. Model at {dst_dir} ({total_bytes / 1e9:.1f} GB). "
            f"{stats.fp4_edited} FP4 + {stats.fp8_edited} FP8 + "
            f"{stats.dense_edited} dense tensors edited, "
            f"{stats.copied} copied. Worst requant rel-err {worst:.4f}."
        )
    return stats


def _copy_aux_files(src_dir: Path, dst_dir: Path) -> None:
    """Every non-weight file of the source (tokenizer, chat template, processor
    and quantisation configs, remote code), so the output stands alone."""
    for p in src_dir.iterdir():
        if not p.is_file() or p.suffix == ".safetensors":
            continue
        if p.name in ("model.safetensors.index.json", "config.json"):
            continue
        shutil.copy2(p, dst_dir / p.name)


# ---------------------------------------------------------------------------
# Plan (de)serialisation
# ---------------------------------------------------------------------------


def save_plan(
    edits: list[TensorEdit], path: str | Path, meta: dict | None = None
) -> None:
    """Serialise a recorded plan to a ``.pt`` file (directions kept as tensors).

    ``meta`` records where the plan came from (e.g. ``model_class``, the
    in-memory model class whose parameter names the plan uses), which lets a
    bake map those names onto another checkpoint's keys.
    """

    def cpu(t: Tensor | None) -> Tensor | None:
        return None if t is None else t.detach().cpu()

    payload = {
        "version": 2,
        "meta": dict(meta or {}),
        "edits": [
            {
                "logical_name": e.logical_name,
                "kind": e.kind,
                "direction": cpu(e.direction),
                "strength": e.strength,
                "preserve_row_norm": e.preserve_row_norm,
                "hidden_dim": e.hidden_dim,
                "transposed": e.transposed,
                "projection_side": e.projection_side,
                "direct_transform": e.direct_transform,
                "benign_dir": cpu(e.benign_dir),
                "rows": None if e.rows is None else list(e.rows),
                "scale": e.scale,
                "expert": e.expert,
            }
            for e in edits
        ],
    }
    torch.save(payload, str(path))


def load_plan_with_meta(path: str | Path) -> tuple[list[TensorEdit], dict]:
    """Read a plan and its metadata. Version-1 plans (a bare list) have none."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        return [TensorEdit(**entry) for entry in payload], {}
    return [TensorEdit(**entry) for entry in payload["edits"]], payload.get("meta", {})


def load_plan(path: str | Path) -> list[TensorEdit]:
    return load_plan_with_meta(path)[0]
