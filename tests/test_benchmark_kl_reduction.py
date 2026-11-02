"""Exercise benchmark isolation and its machine-readable measurement contract."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_small_cpu_benchmark_runs_isolated_workers_and_checks_parity(tmp_path):
    output = tmp_path / "results" / "kl.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_kl_reduction.py"),
            "--batch-size",
            "2",
            "--steps",
            "2",
            "--vocab-size",
            "32",
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--threads",
            "1",
            "--max-chunk-elements",
            "32",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    report = json.loads(completed.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["parity"]["passed"] is True
    assert report["workload"]["shape"] == [2, 2, 32]
    assert (
        report["kl_source_sha256"]
        == hashlib.sha256(
            (ROOT / "src" / "abliterix" / "eval" / "kl.py").read_bytes()
        ).hexdigest()
    )
    workers = report["methods"]
    assert workers["dense"]["worker_pid"] != workers["chunked"]["worker_pid"]
    for worker in workers.values():
        assert worker["input_bytes"] == 2 * 2 * 32 * 4 * 2
        assert len(worker["seconds"]) == len(worker["values"]) == 2
        assert all(seconds > 0 for seconds in worker["seconds"])
        assert worker["memory"]["method"] == "sampled_process_rss_delta"
        assert worker["memory"]["peak_extra_bytes"] >= 0
        assert worker["memory"]["sample_count"] >= 2
        assert worker["environment"]["torch_threads"] == 1
        assert worker["kl_source_sha256"] == report["kl_source_sha256"]
    assert not list(output.parent.glob("*.tmp"))
