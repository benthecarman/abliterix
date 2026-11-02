# Evaluation efficiency and evidence integrity: November 2026 review

Research cutoff: **2026-11-01**. This review covers probability measurement,
evaluation memory, benchmark comparability, judge-cache identity, and manifest
replay. The implementation keeps the existing model-loading dependencies and
the full-vocabulary, mean-per-token KL contract.

## Research decisions

The most useful recent work separates three concerns: reducing intermediate
storage, preserving the quantity being measured, and establishing whether a
reported improvement is supported by independent evidence. A faster proxy is
not automatically an interchangeable metric.

| Primary source | What applies here | Decision and limitation |
| --- | --- | --- |
| [Efficient Knowledge Distillation: Offline Top-K Logits and a Fused Chunked KL Loss](https://arxiv.org/abs/2608.03796), August 2026 | Chunking and avoiding large intermediate tensors are useful systems principles. | Implement a separate, ordinary PyTorch reduction over complete vocabulary rows. The paper concerns top-K teacher targets and fused output-head training; this change does not implement that kernel or inherit its throughput claims. |
| [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009v2), ICLR 2025 | Output logits can dominate memory even when the model itself fits. | Limit single-token forward outputs when the model supports it. CCE computes cross-entropy and is not a drop-in replacement for full-distribution KL. |
| [Liger Kernel](https://arxiv.org/abs/2410.10989v3), January 2025 revision | Fusion and chunking can reduce intermediate memory, with correctness and convergence checks. | Keep a portable reference path. Adding a GPU kernel requires parity and hardware measurements first; no Triton dependency is added. |
| [PyTorch 2.11 log_softmax](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.log_softmax.html) and [KL reduction](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html) | Normalization dtype matters; `batchmean` divides only by the first dimension. | Use FP32 normalization and average over every leading distribution, including token steps. These APIs match the installed runtime. |
| [Transformers Llama forward API](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM.forward) | `logits_to_keep=1` avoids output-head work for unused prompt positions. | Inspect explicit forward parameters, including PEFT/compiled wrappers. Generic `**kwargs` is insufficient evidence of support. Legacy/custom models retain their existing call contract. |
| [Measuring all the noises of LLM Evals](https://arxiv.org/abs/2512.21326v3), October 2026 revision | Paired comparisons distinguish question-sampling and prediction uncertainty. | Preserve the existing prompt-paired and topic-cluster bootstrap tools. Document that repeatedly selected evaluation prompts are not a held-out estimate of generalization. |
| [Frozen Judges, Moving Agents](https://arxiv.org/abs/2609.34198v2), September 2026 | A fixed judge can make different errors on different model versions. | Keep judge provenance separate from evidence of accuracy. Require current-output audits for release claims; the paper's agent-domain effect sizes are not transferred to this benchmark. |
| [SLSA v1.2 provenance](https://slsa.dev/spec/v1.2/build-provenance) | Parameters, resolved inputs, environment and output artifacts have distinct identities. | Test real manifest roundtrips and reject inconsistent artifacts. A self-contained checksum is not publisher authentication, and this project does not claim SLSA certification. |
| [vLLM reproducibility](https://docs.vllm.ai/en/stable/usage/reproducibility/) | Hardware, library version and scheduling constrain reproducibility. | Correct the cross-hardware greedy-decoding claim. Do not enable newer backend switches without checking the pinned version's support. |
| [PyTorch 2.14 release](https://pytorch.org/blog/pytorch-2-14-release-blog/), September 2026 | NVGEMM and compiler improvements are promising for supported GPU workloads. | Keep the tested lockfile. This host's installed Torch is CPU-only 2.11; a dependency upgrade requires a separate CUDA/quantization compatibility matrix. |

## Changes implemented

### Bounded full-vocabulary reduction

[`mean_token_kl`](../../src/abliterix/eval/kl.py) computes
`mean(KL(baseline || current))` over every prompt/token distribution. It uses
complete vocabulary rows, FP32 normalization, and scalar FP64 accumulation on
devices that support it. MPS uses FP32 accumulation.

The default chunk budget is 1,048,576 elements. Each temporary is bounded by
that budget or one complete vocabulary, whichever is larger. Cross-device
baseline transfers are chunked too. Strided inputs are traversed through views
and slices instead of copying the entire tensor just to flatten it.

The caller's input tensors still occupy memory. This is a bounded **reduction**,
not a fused output-head implementation or a streaming baseline store. No top-K
approximation or change to the unit `nats/token` is introduced.

Invalid numerical outputs now produce infinite damage rather than silently
replacing NaN with a finite probability. Legitimate negative-infinity
log-probabilities retain their zero-mass meaning, including zero contributions
from baseline tokens with no mass. Empty or structurally invalid inputs fail
explicitly. Historical scores involving invalid outputs must be remeasured.

### Probability capture and retention

Both the generation-and-scoring batch path and fixed-continuation batch path
honor `inference.offload_outputs_to_cpu` before retaining each batch's output.
Single-token direct scoring disables autograd, normalizes in FP32, and requests
only final-position logits and no KV cache when those keywords are explicitly
supported. Internal model errors propagate without a second inference attempt.

Mixed normalization precision was reproducibly capable of creating a nonzero
KL from identical BF16 logits. FP32 normalization removes that discrepancy.
Offline tests include an actual randomly initialized tiny Transformers Llama,
both plain and PEFT-wrapped, compared against its full-logit forward pass.

### Comparable benchmark inputs

The frozen benchmark rejects a changed KL token count before loading a model.
Its single-token contract cannot be silently replaced by comparing independently
generated multi-token trajectories.

Before comparing models, the runner checks the complete token-to-ID mapping,
rendered benign prompts, and actual unpadded token IDs. Equal vocabulary size
alone does not establish equal token meaning. Template, prefix, or tokenization
mismatches fail clearly; padding from different batch sizes is excluded from
the comparison. This validates comparability, not the claimed ancestry of a
checkpoint. The runner and scorer share the same KL reduction.

### Replay and cache identity

Manifest validation now accepts `vector_index=None` for a saved per-layer
recipe, while requiring a finite numeric index for a global recipe. A saved
recipe must explicitly identify its scope. Tests build actual recipes, write
and load manifests, and extract their replay artifacts.

Judge-cache keys now encode their fields as a versioned JSON tuple. Previously,
joining arbitrary strings with `|` made `(prompt="a|b", response="c")` collide
with `(prompt="a", response="b|c")` before hashing. Version 1 entries remain on
disk but cannot satisfy version 2 lookups. Opted-in external judging may
therefore need fresh API calls; no such calls were made in this validation.

## Measured performance

The [benchmark script](../../scripts/benchmark_kl_reduction.py) runs each method
in a fresh subprocess on identical seeded finite random inputs. Memory is
measured separately from timing, with the input tensors already resident.
Timing uses one warmup followed by three repetitions and reports the median.

Host: Windows, Python 3.12.2, PyTorch **2.11.0+cpu**, four intra-op threads.
Workload: `(128, 8, 32768)` FP32 tensors, 256 MiB of combined input storage.
CPU memory below is sampled process RSS growth, not exact tensor allocations.

| Run | Dense median | Chunked median | Dense extra peak RSS | Chunked extra peak RSS | Absolute KL difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Default 1M-element chunks](../../benchmarks/reliability/2026-11-01-cpu-kl.json) | 53.78 ms | 59.02 ms | 534.30 MiB | 14.40 MiB | 1.37e-8 |
| [4M-element chunks](../../benchmarks/reliability/2026-11-01-cpu-kl-4m.json) | 53.21 ms | 50.59 ms | 620.41 MiB | 81.25 MiB | 1.37e-8 |

The default reduced the observed extra peak by about **97%**, with about **10%**
more reduction time in this run. A larger chunk traded more memory for lower
loop overhead. Keep the conservative default for constrained-memory evaluation;
the helper's `max_chunk_elements` argument and benchmark flag allow controlled
experiments. Neither measurement establishes an end-to-end model speedup.

RSS sampling can miss brief peaks and is affected by allocator reuse, residency
and OS scheduling; even the two dense measurements differ. Raw JSON retains
individual times, environment details, source hashes, numerical tolerances and
measurement limitations. CUDA mode uses PyTorch peak allocated memory with
synchronization, but CUDA and MPS measurements were not available in the
installed runtime. The physical host GPU does not make a CPU Torch build a GPU
validation environment.

Reproduce the default measurement:

```bash
uv run python scripts/benchmark_kl_reduction.py --output kl-benchmark.json
```

## Validation and next acceptance gates

Regression coverage checks independent FP64 probability references, asymmetric
KL orientation, equal per-token weighting, uneven chunks, strided inputs,
FP16/BF16 normalization, zero-probability limits, invalid outputs, offload order,
legacy forward signatures, actual tiny-model parity, tokenizer mismatches,
manifest roundtrips, cache collisions and cache persistence.

Further work has specific evidence requirements:

1. **GPU profiling:** repeat the reduction and model-forward measurements on
   representative vocabulary sizes, prompt lengths, quantizers and hardware.
   Measure peak live/reserved memory and end-to-end latency separately.
2. **Streaming baseline storage:** profile whether retained input distributions
   dominate after this change. A disk-backed or streamed representation must
   preserve prompt order, token identity and dtype before it can replace the
   current tensors.
3. **Optional compiled/fused kernels:** compare against this portable reference
   for numeric extremes and all supported precision modes; include compilation
   cost and repeated weight-update behavior in measurements.
4. **Independent model-quality evidence:** use immutable public inputs, locked
   held-out prompts, paired per-item outcomes, confidence intervals, and a
   capability task with executable answers. Existing Qwen/GSM8K A/B tools cover
   part of this workflow; unavailable historical corpora remain unavailable.
5. **Release provenance:** bind declared base-model and dataset commits to
   complete artifacts, and distinguish digest integrity from publisher trust.
   Tokenizer compatibility alone cannot certify the declared base model.

These gates prevent an engineering improvement from being reported as an
unmeasured improvement in model behavior.
