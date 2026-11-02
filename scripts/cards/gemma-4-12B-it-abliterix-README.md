---
license: gemma
base_model: google/gemma-4-12B-it
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
tags:
  - abliterix
  - abliterated
  - uncensored
  - decensored
  - gemma-4
  - projected-abliteration
  - reproducible
---

# Gemma-4-12B-it — Abliterated (abliterix)

An uncensored, refusal-suppressed version of
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it), produced
by **directional ablation** (no fine-tuning, no new data) with
[**abliterix**](https://github.com/wuwangzhang1216/abliterix).

The model's safety-refusal behaviour is removed by orthogonally projecting a
single *refusal direction* out of two write-path projections (`attn.o_proj`,
`mlp.down_proj`) across the decoder stack, while a norm-preserving transform keeps
the rest of the model's behaviour as close to the original as possible. The
result keeps Gemma-4's capabilities intact and answers prompts the base model
would refuse.

> ⚠️ **Responsible-use notice.** This model has had its safety guardrails removed.
> It will attempt to answer harmful, unethical, or dangerous requests. You are
> solely responsible for how you use it and for complying with the
> [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and all applicable law.
> Intended for safety research, red-teaming, and evaluation.

## Results

Refusal rate is measured on a held-out set of **100 harmful prompts** using an
LLM judge (`google/gemini-3.1-flash-lite`), which is considerably stricter than
the keyword-based detectors typically reported for abliterated models. KL
divergence is the first-token KL from the base model over 100 benign prompts
(lower = closer to the original model).

| Metric | Base `gemma-4-12B-it` | **This model** |
|---|---|---|
| Refusals (LLM judge, 100 harmful prompts) | **99 / 100** | **26 / 100** |
| Refusal reduction | — | **−73.7 pp** |
| First-token KL vs base (benign) | 0.0000 | **0.0735** |

### Comparison with the reference Heretic abliteration

Evaluated apples-to-apples — **the same 100 harmful prompts, the same
`gemini-3.1-flash-lite` judge, the same generation settings** — against the
widely-used Heretic abliteration of this exact base model
([`zaakirio/gemma-4-12b-it-uncensored`](https://huggingface.co/zaakirio/gemma-4-12b-it-uncensored)):

| Model | Refusals (gemini LLM judge, 100 harmful prompts) |
|---|---|
| Base `gemma-4-12B-it` | 99 / 100 |
| `zaakirio/gemma-4-12b-it-uncensored` (Heretic) | 51 / 100 |
| **This model (abliterix)** | **26 / 100** |

The Heretic model card reports ≈23/100 using its built-in keyword detector; under
a stricter LLM judge on the same prompts it refuses 51/100. At the operating point
shipped here, this model refuses **26/100** — roughly **half** the residual refusals
of the reference abliteration, under identical evaluation. (Both are directional-
ablation derivatives of the same base; this comparison measures refusal removal,
not a matched-KL capability trade-off.)

### Why this operating point

Abliteration is a trade-off: removing more refusals perturbs the model more
(higher KL → more capability/coherence risk). abliterix runs a 120-trial
multi-objective (TPE) search and returns the full Pareto front; this release ships
a point on the **knee** of that front — strong refusal removal at a modest,
capability-preserving KL. The full front ranged from `33/100 @ KL 0.043`
(most conservative) to `15/100 @ KL 0.124` (most aggressive); `26/100 @ KL 0.074`
was chosen as the best balance.

## Method

- **Technique:** directional ablation in `direct` (weight-edit) mode — required
  for Gemma-4, whose 4×-RMSNorm-per-layer + Per-Layer-Embedding architecture
  neutralises LoRA/hook-based steering.
- **Direction:** a single mean-difference (harmful − benign) refusal direction,
  computed per layer over 800 benign / 800 harmful prompts.
- **Projected abliteration** ([grimjim](https://huggingface.co/blog/grimjim/projected-abliteration)):
  only the component of the refusal direction orthogonal to the benign direction
  is removed, preserving the helpful signal and keeping KL low.
- **Norm-preserving edit** (`weight_normalization = "full"`): a rank-3 SVD
  approximation restores each weight row's original magnitude after the edit.
- **Targets:** `attn.o_proj` and `mlp.down_proj` only, with a per-layer linear
  "tent" weight profile; Q/K/V and MLP gate/up are left untouched.
- **Search:** 120 Optuna TPE trials, 2-D Pareto over (refusals, KL), deterministic
  under a fixed global seed.

### Selected steering parameters (trial 39)

| Component | max_weight | peak layer | min_weight | tent half-width |
|---|---|---|---|---|
| `attn.o_proj` | 0.955 | 34.9 | 0.773 | 14.4 |
| `mlp.down_proj` | 0.664 | 32.7 | 0.229 | 15.9 |

- Direction scope: **per-layer** · Vector method: mean-difference · Decay: linear
- Global seed: `20260622` · abliterix `v1.8.0`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "wangzhang/gemma-4-12B-it-abliterix"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

msgs = [{"role": "user", "content": "Your prompt here"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

This is a full BF16 merge — drop-in compatible with `transformers`, vLLM, SGLang,
TGI, and any tooling that loads the base model.

## Reproducibility

The run is fully deterministic under the published global seed (`20260622`). The
exact abliterix configuration used to produce this model is included in this
repository as [`abliterix_config.toml`](./abliterix_config.toml); together with the
seed and the trial-39 parameters listed above, the edit can be reproduced or
audited end-to-end. Built with abliterix `v1.8.0` (transformers ≥ 5.10).

## Intended use & limitations

- **Intended for:** safety research, red-teaming, robustness/alignment evaluation,
  and studying refusal mechanisms in LLMs.
- **Not intended for:** producing harmful content or any unlawful purpose.
- Abliteration removes refusals but does not add knowledge; factual accuracy,
  reasoning, and multilingual ability are inherited from the base model.
- Light residual refusals remain (≈26%); this is the chosen capability-preserving
  operating point, not the model's floor.

## Acknowledgments & citation

- Base model: **Google Gemma-4-12B-it**.
- Tooling: **[abliterix](https://github.com/wuwangzhang1216/abliterix)**.
- Method lineage: Arditi et al. (refusal directions, arXiv:2406.11717),
  grimjim (projected / norm-preserving abliteration), and
  [p-e-w/heretic](https://github.com/p-e-w/heretic) (automated multi-objective
  abliteration), whose search formulation this recipe mirrors.

```bibtex
@software{abliterix,
  title  = {abliterix: automated abliteration of large language models},
  author = {Wu, Steve},
  url    = {https://github.com/wuwangzhang1216/abliterix}
}
```

## License

Use is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
This derivative is distributed under the same terms as the base model.
