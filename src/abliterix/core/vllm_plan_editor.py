"""Apply a recorded steering plan to a running vLLM model, in place.

A trial on the vLLM backend should score the model the export writes. The
export records the trial as a list of :class:`~.fp4_repack.TensorEdit` (the
plan) and applies it with :func:`~.fp4_repack.apply_tensor_edits`; this module
applies the same plan, with the same function, to the weights each vLLM worker
holds. Search, export and plan replay then share one set of edits: direct and
EGA projections (with the conv-span basis and row-norm preservation), router
row scaling and the safety-expert projection.

Each worker finds the plan's tensors (named as the HF model names its
parameters) among its own:

* a parameter at the same path under the decoder layer (``self_attn.o_proj``,
  ``mlp.down_proj``, ``mlp.gate``, ``mlp.router``, ``router.proj``);
* the attention output under another name (Inkling 2's ``attn.wo_ud``);
* a slice of the fused ``qkv_proj`` for ``q_proj``/``k_proj``/``v_proj``;
* the fused routed experts' ``w2_weight`` (vLLM keeps it ``[E, hidden,
  intermediate]``; gpt-oss's HF ``down_proj`` is the transpose), whether on a
  FusedMoE (``.routed_experts`` under a MoE runner, an ``_expert_map`` under
  expert parallelism) or on SkyRL's Inkling 2 BF16 experts (contiguous
  ``ep_rank * num_experts`` slices).

Under tensor parallelism a worker holds a slice of a tensor. The edits are
not all slice-local (row norms span the whole row; an output-side projection
sums over every output row), so the worker gathers the full tensor across the
TP group, applies the edits to it exactly as the export does, and keeps its own
slice. Fused experts go in chunks of experts to bound the memory. Every rank
walks the plan in the same order and resolves every tensor before the first
collective, so the ranks cannot disagree about which collectives run.

Each edited tensor goes through a codec: decode the stored tensor(s) to
values, apply the edits, encode back into the stored format and the layout the
kernel reads. Only the dense codec (BF16/FP16/FP32 weights) exists here;
quantized formats register theirs with :func:`register_codec`. The first time a
tensor is edited, the codec's stored tensors are copied to CPU as they are
(quantized or not); :func:`_worker_restore_plan` writes them back after each
trial.
"""

from __future__ import annotations

import dataclasses
import io
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .fp4_repack import TensorEdit, apply_tensor_edits, group_edits
from .vllm_moe_editor import _worker_resolve_model

EXPERT_CHUNK = 16

# Edits whose result for a row depends on that row alone (and on its full
# length): a replicated row subset can take them without seeing other rows.
_ROW_LOCAL_SIDES = {"input"}


# ---------------------------------------------------------------------------
# Tensor-parallel communication
# ---------------------------------------------------------------------------


class _SingleComm:
    rank = 0
    world = 1

    def all_gather(self, t: Tensor, dim: int) -> Tensor:
        return t


class _GroupComm:
    """vLLM's TP group coordinator."""

    def __init__(self, group: Any):
        self.group = group
        self.rank = int(group.rank_in_group)
        self.world = int(group.world_size)

    def all_gather(self, t: Tensor, dim: int) -> Tensor:
        return self.group.all_gather(t.contiguous(), dim)


def _worker_comm(worker: Any):
    comm = getattr(worker, "_abliterix_comm", None)  # tests inject one
    if comm is not None:
        return comm
    try:
        from vllm.distributed import get_tp_group

        group = get_tp_group()
    except Exception:
        return _SingleComm()
    return _SingleComm() if group.world_size == 1 else _GroupComm(group)


# ---------------------------------------------------------------------------
# Finding a plan tensor among a worker's parameters
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
_ATTN_OUT_RE = re.compile(
    r"^(?:self_attn|attn|attention)\.(?:o_proj|out_proj|dense)\.weight$"
)
_QKV_RE = re.compile(r"^(?:self_attn|attention)\.([qkv])_proj\.weight$")
_EXPERT_2D_RE = re.compile(r"experts\.(\d+)\.(?:down_proj|w2)\.weight$")
_ATTN_MODULES = ("self_attn", "attn", "attention")
_ATTN_OUT_NAMES = ("o_proj", "wo_ud", "out_proj", "dense")
_EXPERT_PATHS = (
    "mlp.experts",
    "experts",
    "block_sparse_moe.experts",
    "moe.experts",
    "mixer.experts",
    "feed_forward.experts",
)
_ROUTER_PATHS = (
    "mlp.gate",
    "mlp.router",
    "router.proj",
    "block_sparse_moe.gate",
    "moe.gate",
    "mixer.gate",
)


@dataclass
class _Target:
    """Where one plan tensor lives on this worker.

    ``local`` is a writable view in the HF tensor's orientation; ``param`` is
    the parameter it views (backed up whole). ``shard_dim`` is the dim of
    ``local`` split contiguously across the TP ranks, if any. ``expert_ids``
    gives the global expert of each row of a 3-D ``local`` when the worker
    holds only some experts. ``row_subset`` marks a 2-D slice holding an
    unknown subset of the HF rows (replicated KV heads); only row-local edits
    fit it. ``owner`` is the module holding ``param`` and ``transposed`` says
    ``local`` is ``param`` with its last two dims swapped, for codecs that
    read sibling tensors (scales) or a kernel layout.
    """

    name: str
    param: Tensor
    local: Tensor | None
    hf_shape: tuple[int, ...]
    shard_dim: int | None = None
    expert_ids: list[int] | None = None
    row_subset: bool = False
    owner: Any = None
    transposed: bool = False


def _navigate(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _experts_module(layer: Any) -> Any:
    for path in _EXPERT_PATHS:
        obj = _navigate(layer, path)
        if obj is None:
            continue
        inner = getattr(obj, "routed_experts", None)  # vLLM's MoE runner
        if inner is not None and isinstance(getattr(inner, "w2_weight", None), Tensor):
            return inner
        if isinstance(getattr(obj, "w2_weight", None), Tensor):
            return obj
    return None


def _local_expert_ids(moe: Any, n_local: int, n_global: int) -> list[int] | None:
    """The global expert of each local expert row, or None when all are here."""
    if n_local == n_global:
        return None
    expert_map = getattr(moe, "_expert_map", None)
    if isinstance(expert_map, Tensor) and expert_map.numel() == n_global:
        ids = [-1] * n_local
        for g, local in enumerate(expert_map.tolist()):
            if local >= 0:
                ids[int(local)] = g
        if -1 not in ids:
            return ids
    ep_rank = getattr(moe, "ep_rank", None)
    per_rank = getattr(moe, "num_experts", None)
    if isinstance(ep_rank, int) and per_rank == n_local:
        return list(range(ep_rank * n_local, (ep_rank + 1) * n_local))
    raise ValueError(
        f"{type(moe).__name__} holds {n_local} of {n_global} experts but exposes "
        "no expert map to tell which"
    )


def _shard_dim(
    local: tuple[int, ...], full: tuple[int, ...], world: int, dims
) -> int | None:
    """The one dim among ``dims`` split evenly across ``world`` ranks."""
    differ = [d for d in dims if local[d] != full[d]]
    if not differ:
        return None
    if len(differ) == 1 and local[differ[0]] * world == full[differ[0]]:
        return differ[0]
    raise ValueError(
        f"local shape {local} is not an even TP split of {full} over {world} ranks"
    )


def _resolve(
    decoder: Any,
    name: str,
    hf_shape: tuple[int, ...],
    group: list[TensorEdit],
    world: int,
) -> _Target:
    m = _LAYER_RE.search(name)
    if m is None:
        raise ValueError(f"plan tensor {name!r} names no decoder layer")
    idx, rest = int(m.group(1)), m.group(2)
    if idx >= len(decoder.layers):
        raise ValueError(
            f"plan tensor {name!r}: layer {idx} not in a {len(decoder.layers)}-layer model"
        )
    layer = decoder.layers[idx]

    if len(hf_shape) == 3:  # fused routed experts
        moe = _experts_module(layer)
        if moe is None:
            raise ValueError(
                f"plan tensor {name!r}: no fused experts with w2_weight in layer {idx}"
            )
        w2 = moe.w2_weight
        transposed = any(e.transposed for e in group)
        local = w2.data.transpose(1, 2) if transposed else w2.data
        ids = _local_expert_ids(moe, local.shape[0], hf_shape[0])
        shard = _shard_dim(tuple(local.shape), hf_shape, world, (1, 2))
        if ids is not None and shard is not None:
            raise ValueError(
                f"plan tensor {name!r}: experts split across ranks AND sharded within an expert"
            )
        return _Target(
            name, w2, local, hf_shape, shard, ids, owner=moe, transposed=transposed
        )

    m2 = _EXPERT_2D_RE.search(rest)
    if m2 is not None:  # one expert of a per-expert checkpoint, fused in vLLM
        moe = _experts_module(layer)
        if moe is None:
            raise ValueError(
                f"plan tensor {name!r}: no fused experts with w2_weight in layer {idx}"
            )
        w2 = moe.w2_weight
        ids = _local_expert_ids(moe, w2.shape[0], _n_experts(moe, w2))
        expert = int(m2.group(1))
        row = expert if ids is None else (ids.index(expert) if expert in ids else None)
        if row is None:
            return _Target(name, w2, None, hf_shape, owner=moe)  # another rank's expert
        local = w2.data[row]
        return _Target(
            name,
            w2,
            local,
            hf_shape,
            _shard_dim(tuple(local.shape), hf_shape, world, (0, 1)),
            owner=moe,
        )

    found = _navigate(layer, rest)
    if not isinstance(found, Tensor) and _ATTN_OUT_RE.match(rest):
        for mod in _ATTN_MODULES:
            for out in _ATTN_OUT_NAMES:
                found = _navigate(layer, f"{mod}.{out}.weight")
                if isinstance(found, Tensor):
                    break
            if isinstance(found, Tensor):
                break
    if not isinstance(found, Tensor) and rest.endswith(
        ("gate.weight", "router.weight", "router.proj.weight")
    ):
        for path in _ROUTER_PATHS:
            found = _navigate(layer, f"{path}.weight")
            if isinstance(found, Tensor):
                break
    if isinstance(found, Tensor):
        if found.dim() != 2:
            raise ValueError(
                f"plan tensor {name!r} is 2-D in HF but {tuple(found.shape)} on the worker"
            )
        return _Target(
            name,
            found,
            found.data,
            hf_shape,
            _shard_dim(tuple(found.shape), hf_shape, world, (0, 1)),
        )

    q = _QKV_RE.match(rest)
    if q is not None:
        for mod in _ATTN_MODULES:
            attn = getattr(layer, mod, None)
            qkv = getattr(getattr(attn, "qkv_proj", None), "weight", None)
            if (
                qkv is None
                or not hasattr(attn, "q_size")
                or not hasattr(attn, "kv_size")
            ):
                continue
            q_size, kv_size = int(attn.q_size), int(attn.kv_size)
            lo, hi = {
                "q": (0, q_size),
                "k": (q_size, q_size + kv_size),
                "v": (q_size + kv_size, q_size + 2 * kv_size),
            }[q.group(1)]
            local = qkv.data[lo:hi]
            if local.shape[1] != hf_shape[1]:
                raise ValueError(
                    f"plan tensor {name!r}: qkv slice {tuple(local.shape)} vs HF {hf_shape}"
                )
            if local.shape[0] == hf_shape[0]:
                return _Target(name, qkv, local, hf_shape)
            if local.shape[0] * world == hf_shape[0]:
                return _Target(name, qkv, local, hf_shape, 0)
            # Fewer KV heads than ranks: each rank holds copies of some heads.
            return _Target(name, qkv, local, hf_shape, row_subset=True)
    raise ValueError(
        f"plan tensor {name!r} has no counterpart in layer {idx} of the vLLM model"
    )


def _n_experts(moe: Any, w2: Tensor) -> int:
    for attr in ("global_num_experts", "n_routed_experts"):
        n = getattr(moe, attr, None)
        if isinstance(n, int):
            return n
    expert_map = getattr(moe, "_expert_map", None)
    if isinstance(expert_map, Tensor):
        return expert_map.numel()
    per_rank, ep_size = getattr(moe, "num_experts", None), getattr(moe, "ep_size", None)
    if isinstance(per_rank, int) and isinstance(ep_size, int):
        return per_rank * ep_size
    return w2.shape[0]


# ---------------------------------------------------------------------------
# Applying a plan on one worker
# ---------------------------------------------------------------------------


def _take(t: Tensor, dim: int | None, comm) -> Tensor:
    if dim is None:
        return t
    size = t.shape[dim] // comm.world
    return t.narrow(dim, comm.rank * size, size)


def _backup(worker: Any, stored: list[Tensor]) -> None:
    backup = worker.__dict__.setdefault("_abliterix_plan_backup", {})
    touched = worker.__dict__.setdefault("_abliterix_plan_touched", set())
    for t in stored:
        key = id(t)
        if key not in backup:
            cpu = t.data.detach().to("cpu", copy=True)
            if torch.cuda.is_available():
                try:
                    cpu = cpu.pin_memory()
                except RuntimeError:
                    pass
            backup[key] = (t, cpu)
        touched.add(key)


# ---------------------------------------------------------------------------
# Codecs: stored tensor(s) <-> values in the HF orientation
# ---------------------------------------------------------------------------


class DenseCodec:
    """Unquantized weights: the stored tensor is the values.

    ``decode(tgt, rows)`` returns the values of ``tgt.local[rows]`` (rows of
    experts for a 3-D tensor, everything for ``slice(None)``) in any float
    dtype; ``encode`` writes values back. ``round_between`` is the dtype each
    intermediate edit is rounded to: the engine writes every step into the
    parameter's dtype, and a dense weight does the same. ``stored`` lists the
    tensors a restore must bring back.
    """

    def __init__(self, tgt: _Target):
        self.round_between = tgt.local.dtype
        self.stored = [tgt.param]

    def decode(self, tgt: _Target, rows: slice) -> Tensor:
        return tgt.local[rows]

    def encode(self, tgt: _Target, rows: slice, values: Tensor) -> None:
        tgt.local[rows].copy_(values.to(tgt.local.dtype))


def _dense_codec(tgt: _Target):
    dtype = tgt.param.dtype
    if dtype.is_floating_point and dtype.itemsize >= 2:
        return DenseCodec(tgt)
    return None


# Factories tried in order (newest first); each returns a codec or None.
_CODECS: list = [_dense_codec]


def register_codec(factory) -> None:
    """Add a codec factory ``factory(target) -> codec | None`` (tried before
    the ones already registered). A codec has ``decode``, ``encode``,
    ``round_between`` and ``stored`` as :class:`DenseCodec` does."""
    _CODECS.insert(0, factory)


def _codec_for(tgt: _Target):
    for factory in _CODECS:
        codec = factory(tgt)
        if codec is not None:
            return codec
    raise ValueError(
        f"plan tensor {tgt.name!r} is stored as {tgt.param.dtype} on the worker "
        "and no codec reads that format"
    )


# ---------------------------------------------------------------------------
# Editing one tensor
# ---------------------------------------------------------------------------


def _edit_2d(tgt: _Target, codec, group: list[TensorEdit], comm) -> None:
    rows = slice(None)
    values = codec.decode(tgt, rows)
    if tgt.row_subset:
        if not all(
            e.kind == "direct"
            and e.projection_side in _ROW_LOCAL_SIDES
            and e.basis is None
            for e in group
        ):
            raise ValueError(
                f"plan tensor {tgt.name!r}: rows replicated across ranks take input-side edits only"
            )
        codec.encode(
            tgt,
            rows,
            apply_tensor_edits(values, group, round_between=codec.round_between),
        )
        return
    full = values if tgt.shard_dim is None else comm.all_gather(values, tgt.shard_dim)
    new = apply_tensor_edits(full, group, round_between=codec.round_between)
    codec.encode(tgt, rows, _take(new, tgt.shard_dim, comm))


def _chunk_edits(group: list[TensorEdit], experts: list[int]) -> list[TensorEdit]:
    """The edits for a chunk of experts, expert indices made chunk-relative."""
    position = {g: i for i, g in enumerate(experts)}
    out = []
    for e in group:
        if e.expert is None:
            out.append(e)
        elif e.expert in position:
            out.append(dataclasses.replace(e, expert=position[e.expert]))
    return out


def _edit_3d(tgt: _Target, codec, group: list[TensorEdit], comm, chunk: int) -> None:
    n_local = tgt.local.shape[0]
    for start in range(0, n_local, chunk):
        stop = min(start + chunk, n_local)
        experts = (
            tgt.expert_ids[start:stop]
            if tgt.expert_ids is not None
            else list(range(start, stop))
        )
        edits = _chunk_edits(group, experts)
        # Only ranks holding all experts gather; they all see the same chunks
        # and the same edits, so they skip (or not) together.
        if not edits:
            continue
        rows = slice(start, stop)
        values = codec.decode(tgt, rows)
        full = (
            values if tgt.shard_dim is None else comm.all_gather(values, tgt.shard_dim)
        )
        new = apply_tensor_edits(full, edits, round_between=codec.round_between)
        codec.encode(tgt, rows, _take(new, tgt.shard_dim, comm))


def _worker_apply_plan(worker: Any, payload: bytes) -> dict[str, Any]:
    """Apply a serialized plan (see :func:`encode_plan`) to this worker's weights."""
    data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    edits = [TensorEdit(**d) for d in data["edits"]]
    shapes = {k: tuple(v) for k, v in data["shapes"].items()}
    chunk = int(data.get("chunk", EXPERT_CHUNK))
    comm = _worker_comm(worker)
    decoder = _worker_resolve_model(worker)

    groups = group_edits(edits)
    missing = [n for n in groups if n not in shapes]
    if missing:
        raise ValueError(f"plan has no HF shape for {missing[:3]}")
    # Resolve everything before the first collective: a failure here is the
    # same on every rank, so no rank is left waiting in an all-gather.
    targets = [
        (_resolve(decoder, n, shapes[n], g, comm.world), g) for n, g in groups.items()
    ]
    codecs = [None if tgt.local is None else _codec_for(tgt) for tgt, _ in targets]

    applied = skipped = 0
    with torch.no_grad():
        for (tgt, group), codec in zip(targets, codecs):
            if tgt.local is None:
                skipped += 1
                continue
            group = [_to_device(e, tgt.local.device) for e in group]
            _backup(worker, codec.stored)
            if tgt.local.dim() == 3:
                _edit_3d(tgt, codec, group, comm, chunk)
            else:
                _edit_2d(tgt, codec, group, comm)
            applied += 1
    return {"tensors": applied, "edits": len(edits), "not_local": skipped}


def _to_device(e: TensorEdit, device: torch.device) -> TensorEdit:
    moved = {
        f: getattr(e, f).to(device)
        for f in ("direction", "basis", "benign_dir", "left", "right")
        if isinstance(getattr(e, f), Tensor)
    }
    return dataclasses.replace(e, **moved) if moved else e


def _worker_restore_plan(worker: Any) -> int:
    backup = getattr(worker, "_abliterix_plan_backup", None) or {}
    touched = getattr(worker, "_abliterix_plan_touched", None) or set()
    with torch.no_grad():
        for key in touched:
            param, cpu = backup[key]
            param.data.copy_(cpu.to(param.device, non_blocking=True))
    n = len(touched)
    touched.clear()
    return n


def _edit_dict(e: TensorEdit) -> dict:
    """``TensorEdit(**d)`` gives the edit back; tensors go to CPU."""
    return {
        f.name: (v.detach().cpu() if isinstance(v, Tensor) else v)
        for f in dataclasses.fields(e)
        for v in (getattr(e, f.name),)
    }


def encode_plan(
    edits: list[TensorEdit],
    shapes: dict[str, tuple[int, ...]],
    chunk: int = EXPERT_CHUNK,
) -> bytes:
    """The plan as bytes for collective_rpc, with each tensor's HF shape (the
    full shape, which a worker compares with its slice)."""
    buf = io.BytesIO()
    names = {e.logical_name for e in edits}
    torch.save(
        {
            "edits": [_edit_dict(e) for e in edits],
            "shapes": {n: list(shapes[n]) for n in names if n in shapes},
            "chunk": chunk,
        },
        buf,
    )
    return buf.getvalue()


def hf_shapes(engine: Any) -> dict[str, tuple[int, ...]]:
    """Every parameter's shape, by the plan's (PEFT-stripped) name."""
    from .fp4_repack import normalize_param_name

    return {
        normalize_param_name(n): tuple(p.shape)
        for n, p in engine.model.named_parameters()
    }


# ---------------------------------------------------------------------------
# Driver side
# ---------------------------------------------------------------------------


class VLLMPlanEditor:
    """Applies each trial's steering plan to the vLLM workers' weights.

    ``engine`` is the HF engine's plan skeleton (:meth:`SteeringEngine.plan_skeleton`):
    the model's structure without its weights, which is all plan recording
    reads. Usage per trial::

        edits = record_steering_plan_from_trial(editor.engine, ...)
        editor.apply(edits)
        # ... generate ...
        editor.restore()
    """

    def __init__(self, llm: Any, engine: Any):
        self.llm = llm
        self.engine = engine
        self.shapes = hf_shapes(engine)
        self._applied = False

    def _rpc(self, fn, args: tuple = ()):
        return self.llm.llm_engine.collective_rpc(fn, args=args)

    def apply(self, edits: list[TensorEdit]) -> dict[str, Any]:
        if not edits:
            return {"tensors": 0, "edits": 0, "not_local": 0}
        payload = encode_plan(edits, self.shapes)
        self._applied = True  # a raise on one worker may leave others edited
        try:
            results = self._rpc(_worker_apply_plan, args=(payload,))
        except BaseException:
            self.restore()
            raise
        return (
            results[0]
            if results
            else {"tensors": 0, "edits": len(edits), "not_local": 0}
        )

    def restore(self) -> int:
        if not self._applied:
            return 0
        results = self._rpc(_worker_restore_plan)
        self._applied = False
        return int(results[0]) if results else 0
