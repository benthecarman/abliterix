# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RDO — gradient-based Refusal Direction Optimization.

Implements the optimization-based refusal-direction extractor from
`Wollschläger et al., ICML 2025 <https://arxiv.org/abs/2502.17420>`_ —
*The Geometry of Refusal in Large Language Models: Concept Cones and
Representational Independence*.

Every other ``vector_method`` in abliterix derives the refusal direction
from **activation statistics** of cached residual streams (a mean
difference, an SVD component, an optimal-transport map, …).  RDO instead
**learns** the direction by back-propagating through the frozen model:
a single unit vector ``r`` in hidden space is the only trainable
parameter, optimised by AdamW to minimise

    L = λ_abl · CE( f_ablate(r)(p_harm),  t_answer  )   # answer harmful
      + λ_add · CE( f_add(α·r, l_add)(p_safe),  t_refusal )  # induce refusal
      + λ_ret · KL( f_ablate(r)(p_safe) ‖ f(p_safe) )   # retain benign

where

* ``f_ablate(r)`` projects ``r`` out of the residual stream at **every**
  transformer layer, ``h ← h − (h·r̂) r̂`` (the exact activation-space
  equivalent of abliterix's rank-1 direct weight edit), and the model is
  teacher-forced toward a short affirmative continuation ``t_answer`` on
  harmful prompts;
* ``f_add(α·r, l_add)`` adds ``α·r̂`` at a single layer and the model is
  teacher-forced toward a short refusal continuation ``t_refusal`` on
  harmless prompts;
* the retain term keeps the ablated model's next-token distribution close
  to the untouched model on harmless prompts.

The pay-off reported by the paper is the exact quantity abliterix's Optuna
loop co-minimises: the same refusal-removal efficacy as difference-in-means
at markedly lower capability/KL cost.  RDO learns only the **direction**;
abliterix's existing per-layer strength profile + Optuna search still pick
the magnitude, so the two compose cleanly.

Loss weights, learning rate, and step count default to the paper's values
(λ = 1.0/0.2/1.0, AdamW lr 0.01) but are exposed as config knobs so the
Optuna loop can tune them.  Unlike the paper, ``r`` is warm-started from
the mean-difference direction by default (cheaper convergence than the
paper's random init); pass ``rdo_init = "random"`` to reproduce the paper.

Requires a loaded HuggingFace model with autograd (``engine.model`` is not
``None``): the fast-extraction vLLM path cannot back-propagate, so RDO
fails fast there with a clear message.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .settings import AbliterixConfig
from .types import ChatMessage
from .util import print

__all__ = ["optimize_rdo_direction"]


def _mean_diff_direction(
    benign_states: Tensor,
    target_states: Tensor,
    layer_idx: int,
) -> Tensor:
    """Unit mean-difference direction at ``layer_idx`` (warm-start seed)."""
    diff = target_states[:, layer_idx, :].float().mean(0) - benign_states[
        :, layer_idx, :
    ].float().mean(0)
    return F.normalize(diff, p=2, dim=0)


def _teacher_forced_batch(engine, messages, target_text: str):
    """Build a teacher-forced ``(inputs, labels)`` pair for a shared target.

    Reuses ``engine._tokenize`` (which applies the chat template and
    left-pads exactly as the rest of abliterix does), then appends the same
    ``target_text`` tokens to every row.  Because the prompts are left-padded
    to a common length and the target is identical, the target tokens land at
    the final ``len(target_ids)`` positions of every sequence, so the loss
    mask is a simple suffix.
    """
    inputs = engine._tokenize(messages)
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]
    device = input_ids.device
    batch = input_ids.shape[0]

    tgt = engine.tokenizer(
        target_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    tgt = tgt.expand(batch, -1)  # (batch, T) — shared target
    t_len = tgt.shape[1]

    full_ids = torch.cat([input_ids, tgt], dim=1)
    full_attn = torch.cat([attn, torch.ones_like(tgt)], dim=1)

    labels = torch.full_like(full_ids, -100)
    labels[:, -t_len:] = tgt
    return {"input_ids": full_ids, "attention_mask": full_attn}, labels, t_len


def _causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Standard next-token cross-entropy with a ``-100`` ignore mask."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


class _Intervention:
    """Registers forward hooks that ablate / add ``r̂`` on the residual stream.

    ``r`` is held by reference (the live :class:`torch.nn.Parameter`) so the
    hook is differentiable w.r.t. the direction — gradients flow back through
    every layer's activations into ``r``.
    """

    def __init__(self, layers, r: Tensor, mode: str, add_layer: int, add_scale: float):
        self.layers = layers
        self.r = r
        self.mode = mode  # "ablate" (all layers) | "add" (single layer)
        self.add_layer = add_layer
        self.add_scale = add_scale
        self.handles: list = []

    def _apply(self, h: Tensor) -> Tensor:
        rhat = F.normalize(self.r.float(), p=2, dim=0).to(h.dtype)
        if self.mode == "ablate":
            proj = (h * rhat).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
            return h - proj * rhat
        # add
        return h + self.add_scale * rhat

    def _make_hook(self, layer_idx: int):
        def hook(module, inp, out):
            if self.mode == "add" and layer_idx != self.add_layer:
                return out
            if isinstance(out, tuple):
                return (self._apply(out[0]),) + out[1:]
            return self._apply(out)

        return hook

    def __enter__(self):
        if self.mode == "ablate":
            targets = list(enumerate(self.layers))
        else:
            targets = [(self.add_layer, self.layers[self.add_layer])]
        for idx, layer in targets:
            self.handles.append(layer.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


def optimize_rdo_direction(
    engine,
    target_msgs: list[ChatMessage],
    benign_msgs: list[ChatMessage],
    config: AbliterixConfig,
    *,
    benign_states: Tensor | None = None,
    target_states: Tensor | None = None,
) -> Tensor:
    """Learn a single refusal direction via gradient descent through the model.

    Parameters
    ----------
    engine : SteeringEngine
        Must expose a loaded HF ``model`` (autograd-capable), ``tokenizer``,
        ``transformer_layers`` and ``_tokenize``.
    target_msgs, benign_msgs : list[ChatMessage]
        Harmful and harmless prompt message lists.
    config : AbliterixConfig
        Reads the ``steering.rdo_*`` knobs.
    benign_states, target_states : Tensor, optional
        Cached residuals ``(n, layers+1, hidden)`` used only to warm-start
        ``r`` from the mean-difference direction.  Random init if absent.

    Returns
    -------
    Tensor
        Steering vectors of shape ``(layers+1, hidden_dim)`` — the single
        learned unit direction broadcast to every layer, matching the layout
        every other ``vector_method`` returns.  Winsorization and
        projected/orthogonal projection are applied to match the config, so
        RDO composes with those flags.
    """
    model = getattr(engine, "model", None)
    if model is None:
        raise RuntimeError(
            "vector_method='rdo' requires a loaded HuggingFace model with "
            "autograd, but engine.model is None. This usually means the "
            "fast-extraction vLLM path is active (which cannot back-propagate). "
            "Run RDO on the HF path (disable the vLLM fast-extraction backend)."
        )

    s = config.steering
    layers = engine.transformer_layers
    n_layers = len(layers)
    # Clear stale VLM rope_deltas between forwards of differing seq length
    # (teacher-forced batches, retain batch) — same guard as extract_hidden_states.
    reset_pos = getattr(engine, "_reset_position_cache", lambda: None)
    device = next(model.parameters()).device
    hidden_dim = (
        target_states.shape[-1]
        if target_states is not None
        else int(model.config.hidden_size)
    )
    add_layer = max(0, min(n_layers - 1, round(s.rdo_add_layer_frac * (n_layers - 1))))

    _seed = s.rdo_seed if s.rdo_seed is not None else config.seed
    if _seed is not None:
        torch.manual_seed(_seed)

    # --- Initialise the trainable direction -------------------------------
    if (
        s.rdo_init == "mean_diff"
        and benign_states is not None
        and target_states is not None
    ):
        # Warm-start from the mean-diff direction at the addition layer
        # (hidden_states[add_layer+1] is the output of transformer_layers[add_layer]).
        seed = _mean_diff_direction(benign_states, target_states, add_layer + 1)
        r0 = seed.to(device=device, dtype=torch.float32)
        init_desc = f"mean-diff @ layer {add_layer}"
    else:
        r0 = F.normalize(torch.randn(hidden_dim, device=device), p=2, dim=0)
        init_desc = "random"
    r = torch.nn.Parameter(r0.clone())

    # --- Freeze the model (only r is trainable); restore afterwards -------
    prev_requires_grad = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in prev_requires_grad:
        p.requires_grad_(False)
    was_training = model.training
    model.eval()

    # abliterix disables autograd globally at startup (cli.configure_libraries
    # calls torch.set_grad_enabled(False) for fast inference).  RDO needs a
    # backward pass, so re-enable grad locally and restore it in the finally.
    prev_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)

    opt = torch.optim.AdamW([r], lr=s.rdo_lr)

    n = min(len(target_msgs), len(benign_msgs), s.rdo_max_prompts)
    harm = target_msgs[:n]
    safe = benign_msgs[:n]
    bs = max(1, s.rdo_batch_size)

    print(
        f"* RDO: learning refusal direction ({init_desc} init, {s.rdo_steps} steps, "
        f"lr {s.rdo_lr}, add-layer {add_layer}, {n} prompts, batch {bs})"
    )

    try:
        for step in range(s.rdo_steps):
            i = (step * bs) % max(1, n)
            harm_b = harm[i : i + bs] or harm[:bs]
            safe_b = safe[i : i + bs] or safe[:bs]
            opt.zero_grad(set_to_none=True)

            # Each term is back-propagated separately so only one forward
            # graph is alive at a time (summing all three would keep three
            # full-model graphs in memory → OOM on large models). Gradients
            # accumulate into r.grad across the three backward calls.

            # (1) Ablation loss — answer harmful prompts after removing r.
            abl_inputs, abl_labels, _ = _teacher_forced_batch(
                engine, harm_b, s.rdo_affirmative_target
            )
            reset_pos()
            with _Intervention(layers, r, "ablate", add_layer, s.rdo_add_scale):
                abl_logits = model(**abl_inputs).logits
            loss_abl = _causal_lm_loss(abl_logits, abl_labels)
            (s.rdo_lambda_ablation * loss_abl).backward()

            # (2) Addition loss — induce refusal on harmless prompts.
            add_inputs, add_labels, _ = _teacher_forced_batch(
                engine, safe_b, s.rdo_refusal_target
            )
            reset_pos()
            with _Intervention(layers, r, "add", add_layer, s.rdo_add_scale):
                add_logits = model(**add_inputs).logits
            loss_add = _causal_lm_loss(add_logits, add_labels)
            (s.rdo_lambda_addition * loss_add).backward()

            # (3) Retain loss — ablation must not change benign behaviour.
            ret_inputs = engine._tokenize(safe_b)
            reset_pos()
            with torch.no_grad():
                clean_logits = model(**ret_inputs).logits
            reset_pos()
            with _Intervention(layers, r, "ablate", add_layer, s.rdo_add_scale):
                ret_logits = model(**ret_inputs).logits
            # Forward KL(ablate ‖ clean) per the paper — sum p_ablate·(log
            # p_ablate − log p_clean).  clean_logits comes from a no_grad
            # forward, so only the ablated branch carries gradient to r.
            mask = ret_inputs["attention_mask"].bool().reshape(-1)
            logp_ablate = F.log_softmax(
                ret_logits.reshape(-1, ret_logits.size(-1)).float(), -1
            )[mask]
            logp_clean = F.log_softmax(
                clean_logits.reshape(-1, clean_logits.size(-1)).float(), -1
            )[mask]
            loss_ret = (logp_ablate.exp() * (logp_ablate - logp_clean)).sum(-1).mean()
            (s.rdo_lambda_retain * loss_ret).backward()

            opt.step()

            if step == 0 or (step + 1) % max(1, s.rdo_steps // 5) == 0:
                total = (
                    s.rdo_lambda_ablation * loss_abl.item()
                    + s.rdo_lambda_addition * loss_add.item()
                    + s.rdo_lambda_retain * loss_ret.item()
                )
                print(
                    f"    step {step + 1}/{s.rdo_steps}  "
                    f"L={total:.4f}  "
                    f"abl={loss_abl.item():.4f} add={loss_add.item():.4f} "
                    f"ret={loss_ret.item():.4f}"
                )
    finally:
        torch.set_grad_enabled(prev_grad_enabled)
        for p, rg in prev_requires_grad:
            p.requires_grad_(rg)
        if was_training:
            model.train()

    # --- Broadcast the single learned direction to every layer ------------
    rhat = F.normalize(r.detach().float(), p=2, dim=0)
    vectors = rhat.unsqueeze(0).repeat(n_layers + 1, 1)  # (layers+1, hidden)

    # Return on the same device as the residual states — the convention every
    # other vector_method follows (typically CPU when offload_outputs_to_cpu is
    # enabled).  ``r`` trains on the model device (GPU); without this the
    # projection below and downstream steering hit a cross-device mismatch.
    if benign_states is not None:
        vectors = vectors.to(benign_states.device)

    # --- Compose with winsorization / projection to honour the config -----
    if s.winsorize_vectors:
        from .vectors import _winsorize

        vectors = _winsorize(vectors, quantile=s.winsorize_quantile)
        vectors = F.normalize(vectors, p=2, dim=1)

    if (
        s.projected_abliteration or s.orthogonal_projection
    ) and benign_states is not None:
        benign_dir = F.normalize(benign_states.mean(dim=0).float(), p=2, dim=1)
        proj_scalar = torch.sum(vectors * benign_dir, dim=1, keepdim=True)
        vectors = F.normalize(vectors - proj_scalar * benign_dir, p=2, dim=1)

    return vectors
