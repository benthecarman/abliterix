# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full-vocabulary KL reduction with bounded temporary storage."""

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_MAX_CHUNK_ELEMENTS = 1_048_576


def _distribution_chunks(
    current: Tensor, baseline: Tensor, max_elements: int
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield complete distributions without copying a strided input in full."""
    vocab_size = current.shape[-1]
    try:
        current_rows = current.view(-1, vocab_size)
        baseline_rows = baseline.view(-1, vocab_size)
    except RuntimeError:
        # Transposed leading dimensions cannot always be flattened as views.
        # Descend through slices instead of reshape(), which would copy the
        # entire input before the first bounded reduction could start.
        for index in range(current.shape[0]):
            yield from _distribution_chunks(
                current[index], baseline[index], max_elements
            )
        return

    rows_per_chunk = max(1, max_elements // vocab_size)
    for start in range(0, current_rows.shape[0], rows_per_chunk):
        yield (
            current_rows[start : start + rows_per_chunk],
            baseline_rows[start : start + rows_per_chunk],
        )


@torch.no_grad()
def mean_token_kl(
    current_logprobs: Tensor,
    baseline_logprobs: Tensor,
    *,
    max_chunk_elements: int = DEFAULT_MAX_CHUNK_ELEMENTS,
) -> float:
    """Return mean KL(baseline || current), in nats per distribution.

    The last dimension is the complete vocabulary; every leading index is
    equally weighted. Normalization and vocabulary reduction use FP32.
    Chunking bounds temporary tensors and cross-device transfers, not the
    storage of the caller's inputs. A chunk always contains at least one
    complete vocabulary, even when it exceeds ``max_chunk_elements``.

    Negative infinity represents zero probability. NaN, positive infinity,
    and all-zero-probability rows produce infinite damage, rather than a
    fabricated finite score. Inputs are never modified.
    """
    if current_logprobs.shape != baseline_logprobs.shape:
        raise ValueError(
            "Current and baseline log-probabilities must have the same shape: "
            f"got {tuple(current_logprobs.shape)} and "
            f"{tuple(baseline_logprobs.shape)}"
        )
    if current_logprobs.ndim == 0 or current_logprobs.numel() == 0:
        raise ValueError("KL requires at least one non-empty distribution.")
    if not (
        current_logprobs.is_floating_point() and baseline_logprobs.is_floating_point()
    ):
        raise ValueError("KL log-probabilities must be floating-point tensors.")
    if (
        isinstance(max_chunk_elements, bool)
        or not isinstance(max_chunk_elements, int)
        or max_chunk_elements < 1
    ):
        raise ValueError("max_chunk_elements must be a positive integer.")

    # Accumulate scalar sums in FP64 where supported. MPS does not support
    # FP64; the vocabulary arithmetic remains FP32 on every device.
    device = current_logprobs.device
    sum_dtype = torch.float32 if device.type == "mps" else torch.float64
    total = torch.zeros((), device=device, dtype=sum_dtype)
    count = current_logprobs.numel() // current_logprobs.shape[-1]
    for current_chunk, baseline_chunk in _distribution_chunks(
        current_logprobs, baseline_logprobs, max_chunk_elements
    ):
        current = F.log_softmax(current_chunk, dim=-1, dtype=torch.float32)
        baseline = F.log_softmax(baseline_chunk.to(device), dim=-1, dtype=torch.float32)
        terms = F.kl_div(current, baseline, reduction="none", log_target=True)
        # PyTorch's log-target formula evaluates 0 * (-inf) as NaN. The
        # mathematical contribution of a zero-mass baseline token is zero,
        # including when the current distribution also assigns it zero.
        terms.masked_fill_(baseline.isneginf(), 0.0)
        total += terms.sum(dim=-1).sum(dtype=sum_dtype)
        del current, baseline, terms

    result = float((total / count).item())
    return max(0.0, result) if torch.isfinite(total).item() else float("inf")
