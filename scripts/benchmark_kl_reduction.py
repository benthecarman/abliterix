#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare dense and chunked KL reductions on identical finite random inputs.

Each method runs in a fresh subprocess. Memory is measured before warmup;
timings are measured separately after warmup. No model or dataset is loaded.

Example:
    python scripts/benchmark_kl_reduction.py --output kl-benchmark.json
    python scripts/benchmark_kl_reduction.py --device cuda:0 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KL_SOURCE = ROOT / "src" / "abliterix" / "eval" / "kl.py"
METHODS = ("dense", "chunked")
ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-5


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--steps", type=_positive_int, default=8)
    parser.add_argument("--vocab-size", type=_positive_int, default=32768)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--device", default="cpu", help="cpu or a CUDA device, e.g. cuda:0"
    )
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--warmup", type=_positive_int, default=1)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--max-chunk-elements", type=_positive_int, default=1_048_576)
    parser.add_argument("--rss-sample-ms", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.rss_sample_ms) or args.rss_sample_ms <= 0:
        parser.error("--rss-sample-ms must be finite and positive")
    return args


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dense_kl(current, baseline):
    """Preserve the previous dense formula, including its cleaning allocations."""
    import torch
    import torch.nn.functional as F

    current = F.log_softmax(
        torch.nan_to_num(current, nan=-30.0, posinf=0.0, neginf=-30.0),
        dim=-1,
        dtype=torch.float32,
    )
    baseline = F.log_softmax(
        torch.nan_to_num(baseline, nan=-30.0, posinf=0.0, neginf=-30.0),
        dim=-1,
        dtype=torch.float32,
    )
    per_distribution = F.kl_div(
        current, baseline, reduction="none", log_target=True
    ).sum(dim=-1)
    kl = per_distribution.mean()
    return max(0.0, float(kl.item())) if torch.isfinite(kl).item() else float("inf")


def _cpu_memory_pass(reduce, interval_seconds):
    import psutil

    process = psutil.Process()
    stop = threading.Event()
    ready = threading.Event()
    rss_samples = []

    def sample():
        rss_samples.append(process.memory_info().rss)
        ready.set()
        while not stop.wait(interval_seconds):
            rss_samples.append(process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    ready.wait()
    baseline = process.memory_info().rss
    try:
        value = reduce()
        rss_samples.append(process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    peak = max(baseline, *rss_samples)
    return value, {
        "method": "sampled_process_rss_delta",
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "peak_extra_bytes": peak - baseline,
        "sample_count": len(rss_samples),
        "requested_sample_interval_ms": interval_seconds * 1000,
        "limitations": [
            "RSS is an estimate of resident process memory, not tensor allocation bytes.",
            "Sampling and OS scheduling can miss short-lived peaks, especially for small workloads.",
            "Allocator reuse, residency changes, and sampler overhead affect the delta; a zero delta does not imply zero allocations.",
        ],
    }


def run_worker(args):
    import psutil
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from abliterix.eval.kl import mean_token_kl

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA memory measurements are supported.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is not available in this PyTorch environment."
            )
        torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    shape = (args.batch_size, args.steps, args.vocab_size)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    current = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    baseline = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    for tensor in (current, baseline):
        for chunk in tensor.view(-1).split(1_048_576):
            if not torch.isfinite(chunk).all().item():
                raise RuntimeError(
                    "The finite-input workload contains nonfinite values."
                )
    del tensor, chunk
    gc.collect()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def reduce():
        with torch.no_grad():
            if args.worker_method == "dense":
                return _dense_kl(current, baseline)
            return mean_token_kl(
                current, baseline, max_chunk_elements=args.max_chunk_elements
            )

    # Measure before either implementation has populated its allocator caches.
    synchronize()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory_value = reduce()
        synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        memory = {
            "method": "cuda_max_memory_allocated_delta",
            "baseline_bytes": allocated,
            "peak_bytes": peak,
            "peak_extra_bytes": peak - allocated,
            "limitations": [
                "Measures PyTorch live device allocations; excludes reserved allocator memory and non-PyTorch CUDA allocations.",
                "Does not measure host RAM or CUDA driver/context memory.",
            ],
        }
    else:
        memory_value, memory = _cpu_memory_pass(reduce, args.rss_sample_ms / 1000)

    for _ in range(args.warmup):
        reduce()
    synchronize()
    seconds, values = [], []
    for _ in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        value = reduce()
        synchronize()
        seconds.append(time.perf_counter() - start)
        values.append(value)
    return {
        "method": args.worker_method,
        "worker_pid": os.getpid(),
        "kl_source_sha256": _sha256(KL_SOURCE),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "psutil": psutil.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "input_bytes": current.numel() * current.element_size() * 2,
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "values": values,
        "memory_pass_value": memory_value,
        "memory": memory,
    }


def run_benchmark(args):
    source_hash = _sha256(KL_SOURCE)
    common = [sys.executable, str(Path(__file__).resolve())]
    for name in (
        "batch_size",
        "steps",
        "vocab_size",
        "dtype",
        "device",
        "repeats",
        "warmup",
        "threads",
        "seed",
        "max_chunk_elements",
        "rss_sample_ms",
    ):
        common.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    workers = {}
    for method in METHODS:
        try:
            process = subprocess.run(
                [*common, "--worker-method", method],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"{method} benchmark worker failed:\n{error.stderr}"
            ) from error
        workers[method] = json.loads(process.stdout)
    if any(worker["kl_source_sha256"] != source_hash for worker in workers.values()):
        raise RuntimeError("kl.py changed while the benchmark was running.")
    dense_value = workers["dense"]["values"][0]
    all_values = [
        value
        for worker in workers.values()
        for value in [worker["memory_pass_value"], *worker["values"]]
    ]
    passed = all(
        math.isfinite(value)
        and math.isclose(
            value, dense_value, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE
        )
        for value in all_values
    )
    return {
        "schema_version": "abliterix-kl-reduction-benchmark-v1",
        "kl_source_sha256": source_hash,
        "benchmark_source_sha256": _sha256(Path(__file__)),
        "workload": {
            "shape": [args.batch_size, args.steps, args.vocab_size],
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "inputs": "Two independent finite torch.randn tensors from one seeded device generator; unnormalized log-probabilities.",
            "max_chunk_elements": args.max_chunk_elements,
            "warmup_calls_after_memory_pass": args.warmup,
            "timed_repeats": args.repeats,
        },
        "methodology": {
            "isolation": "One fresh subprocess per method, run sequentially.",
            "timing": "Median perf_counter walltime after warmup; CUDA synchronized before and after each call. Excludes setup, input construction, and memory sampling.",
            "memory": "First reduction before warmup; subtract pre-call memory with both input tensors already resident. CPU uses sampled RSS; CUDA uses PyTorch peak allocated bytes.",
            "limits": [
                "Synthetic finite inputs measure reduction cost only, not end-to-end model performance.",
                "Identical RNG seed and construction are used in both workers; random values may differ across devices, dtypes, or PyTorch versions.",
                "Small shapes can be dominated by launch, thread, or Python overhead; timing and memory are separate passes.",
            ],
        },
        "methods": workers,
        "parity": {
            "passed": passed,
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": REL_TOLERANCE,
            "absolute_difference": abs(workers["chunked"]["values"][0] - dense_value),
            "max_absolute_difference_across_passes": max(
                abs(value - dense_value) for value in all_values
            ),
        },
    }


def main(argv=None):
    args = parse_args(argv)
    report = run_worker(args) if args.worker_method else run_benchmark(args)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=args.output.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(temporary, args.output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(payload, end="")
    return 0 if args.worker_method or report["parity"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
