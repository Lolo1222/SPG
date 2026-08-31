"""Small helpers for bounding vocabulary-logit allocations."""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def selected_token_confidence(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Probability of selected tokens without allocating ``softmax(logits)``."""
    selected = torch.gather(logits, -1, token_ids.unsqueeze(-1)).squeeze(-1)
    return torch.exp(selected - torch.logsumexp(logits, dim=-1))


def chunked_token_nll(
    model,
    get_logits: Callable,
    noisy_batch: torch.Tensor,
    targets: torch.Tensor,
    logits_to_keep: int,
    micro_batch_size: Optional[int],
    prompt_index: torch.Tensor,
    cfg_scale: float,
    mask_id: int,
    prompt_mask: Optional[torch.Tensor] = None,
    need_prob: bool = False,
):
    """Compute per-token NLL in bounded chunks of the leading batch dimension.

    ``noisy_batch`` may be ``[B, T, L]`` or ``[B, L]``. The returned tensors
    have the matching ``[B, T, logits_to_keep]``/``[B, logits_to_keep]`` shape.
    Chunking is also applied before CFG's unconditional duplication inside
    ``get_logits``.
    """
    if noisy_batch.ndim == 2:
        noisy_batch = noisy_batch.unsqueeze(1)
        squeeze_time = True
    elif noisy_batch.ndim == 3:
        squeeze_time = False
    else:
        raise ValueError(f"expected [B,L] or [B,T,L], got {tuple(noisy_batch.shape)}")
    if targets.ndim == 2:
        targets = targets.unsqueeze(1)
    batch_size = noisy_batch.size(0)
    chunk_size = batch_size if micro_batch_size is None else int(micro_batch_size)
    if chunk_size < 1:
        raise ValueError("logits_micro_batch_size must be at least 1")
    nll_chunks = []
    prob_chunks = []
    for start in range(0, batch_size, min(chunk_size, batch_size)):
        end = min(start + chunk_size, batch_size)
        target_chunk = targets[start:end].expand(-1, noisy_batch.size(1), -1)
        prompt_chunk = None if prompt_mask is None else prompt_mask[start:end]

        def _loss_and_prob(sequence_chunk, target_chunk, prompt_chunk):
            logits = get_logits(
                model,
                sequence_chunk,
                prompt_index,
                cfg_scale,
                mask_id,
                prompt_chunk,
            )
            completion_logits = logits[..., -logits_to_keep:, :]
            flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
            flat_targets = target_chunk.reshape(-1)
            nll = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(
                end - start, noisy_batch.size(1), logits_to_keep
            )
            if need_prob:
                prob = selected_token_confidence(flat_logits, flat_targets).view(
                    end - start, noisy_batch.size(1), logits_to_keep
                )
            else:
                prob = torch.empty(0, device=nll.device, dtype=nll.dtype)
            return nll, prob

        sequence_chunk = noisy_batch[start:end]
        if torch.is_grad_enabled():
            nll_chunk, prob_chunk = checkpoint(
                _loss_and_prob, sequence_chunk, target_chunk, prompt_chunk, use_reentrant=False
            )
        else:
            nll_chunk, prob_chunk = _loss_and_prob(sequence_chunk, target_chunk, prompt_chunk)
        nll_chunks.append(nll_chunk)
        if need_prob:
            prob_chunks.append(prob_chunk)
        del sequence_chunk, target_chunk, nll_chunk, prob_chunk
    nll = torch.cat(nll_chunks, dim=0)
    probs = torch.cat(prob_chunks, dim=0) if need_prob else None
    if squeeze_time:
        nll = nll.squeeze(1)
        if probs is not None:
            probs = probs.squeeze(1)
    return nll, probs
