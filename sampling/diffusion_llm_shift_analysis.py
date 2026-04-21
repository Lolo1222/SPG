#!/usr/bin/env python3
"""Analyze distribution shift between a diffusion model and an autoregressive LLM.

This script runs blockwise diffusion generation, records per-token and per-block
log probabilities, then scores the same generated text with an independent AR
LLM and compares block-level likelihoods. It saves sample-level JSONL, flattened
CSV tables, summary statistics, and plots.
"""

from __future__ import annotations
from tqdm import tqdm
import argparse
import json
import os
import random
import warnings
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(1, str(EVAL_DIR))

from eval.countdown import CTDDataset
from eval.gsm8k import GSM8KDataset
from eval.math500 import MATH500Dataset
from eval.sudoku import SudokuDataset


DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}


@dataclass
class BlockRecord:
    block_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    token_ids: List[int]
    token_logps_diffusion: List[float]
    token_logps_llm: List[float]
    block_logp_diffusion: float
    block_logp_llm: float
    block_gap: float


@dataclass
class SampleRecord:
    sample_index: int
    dataset_index: int
    question: str
    original_prompt: str
    model_prompt_text: str
    completion_text: str
    ground_truth: Any
    completion_ids: List[int]
    completion_tokens: List[str]
    completion_token_logps_diffusion: List[float]
    completion_token_logps_llm: List[float]
    blocks: List[BlockRecord]
    total_logp_diffusion: float
    total_logp_llm: float
    total_gap: float
    mean_block_gap: float
    mean_abs_block_gap: float


@dataclass
class ESSResult:
    ess: float
    ess_ratio: float
    log_weights_mean: float
    log_weights_std: float
    n_samples: int


def init_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))
    return str(obj)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def json_default(obj: Any) -> Any:
    return to_jsonable(obj)


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.mask_token_id is not None:
        tokenizer.pad_token = tokenizer.mask_token
    elif tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token


def load_diffusion_model(model_path: str, checkpoint_path: str, dtype: torch.dtype, device: torch.device):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
    if checkpoint_path:
        model = PeftModel.from_pretrained(model, checkpoint_path, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model


def load_llm_model(model_path: str, dtype: torch.dtype, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    ensure_tokenizer_padding(tokenizer)
    return tokenizer


def load_training_records(dataset_name: str):
    if dataset_name == "gsm8k":
        return load_dataset("gsm8k", "main", split="train")

    if dataset_name == "math":
        train_path = REPO_ROOT / "dataset" / "math500_train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(f"MATH train file not found: {train_path}")
        records = []
        with train_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                problem = item.get("problem")
                answer = item.get("solution", item.get("answer"))
                if isinstance(problem, str) and isinstance(answer, str):
                    records.append({"problem": problem, "answer": answer})
        return records

    if dataset_name == "sudoku":
        train_path = REPO_ROOT / "dataset" / "train_sudoku_split_new.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Sudoku train file not found: {train_path}")
        df = pd.read_csv(train_path, dtype={"Puzzle": str, "Solution": str})
        return df.to_dict(orient="records")

    if dataset_name == "countdown":
        raise NotImplementedError(
            "Countdown training split is not available in this repository. Provide a train file or add a dedicated loader."
        )

    raise KeyError(f"Unknown dataset: {dataset_name}")


def build_dataset(tokenizer, args):
    dataset_kwargs = {
        "num_examples": args.few_shot,
        "add_reasoning": True,
        "subsample": -1,
    }
    if args.dataset == "math":
        dataset_kwargs.update(
            {
                "subset": args.math500_subset,
                "split_file": args.math500_split_file if args.math500_split_file else None,
                "split_seed": args.math500_split_seed,
            }
        )
    dataset = DATASET_MAP[args.dataset](tokenizer, **dataset_kwargs)
    train_records = load_training_records(args.dataset)

    dataset.dataset = train_records

    if args.num_samples > 0 and args.num_samples < len(train_records):
        selected = np.random.choice(len(train_records), size=args.num_samples, replace=False)
        dataset.subsample = np.asarray(selected)
        print(f"Using a random subset of {len(dataset.subsample)} training examples from {args.dataset}")
    else:
        dataset.subsample = np.arange(len(train_records))
        print(f"Using all {len(train_records)} training examples from {args.dataset}")
    return dataset


def add_gumbel_noise(logits: torch.Tensor, temperature: float, dtype: torch.dtype) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.to(dtype)
    noise = torch.rand_like(logits, dtype=dtype)
    gumbel_noise = (-torch.log(noise)).pow(temperature)
    return logits.exp() / gumbel_noise


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """Apply top-k and/or nucleus (top-p) filtering to logits.

    This function expects logits with shape [..., vocab_size].
    """
    filtered = logits

    if top_k is not None and top_k > 0:
        k = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, k)[0][..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits.to(torch.float32), dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(filtered, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        filtered = filtered.masked_fill(indices_to_remove, filter_value)

    return filtered


def compare_vocabularies(diff_tokenizer, llm_tokenizer) -> Dict[str, Any]:
    diff_vocab = diff_tokenizer.get_vocab()
    llm_vocab = llm_tokenizer.get_vocab()

    diff_tokens = set(diff_vocab.keys())
    llm_tokens = set(llm_vocab.keys())
    shared_tokens = diff_tokens & llm_tokens

    same_id_count = 0
    diff_id_count = 0
    for token in shared_tokens:
        if diff_vocab[token] == llm_vocab[token]:
            same_id_count += 1
        else:
            diff_id_count += 1

    only_diff = sorted(diff_tokens - llm_tokens)
    only_llm = sorted(llm_tokens - diff_tokens)

    report = {
        "diff_vocab_size": int(len(diff_vocab)),
        "llm_vocab_size": int(len(llm_vocab)),
        "shared_token_count": int(len(shared_tokens)),
        "same_token_same_id_count": int(same_id_count),
        "same_token_different_id_count": int(diff_id_count),
        "token_set_exact_match": bool(diff_tokens == llm_tokens),
        "id_mapping_exact_match": bool(diff_tokens == llm_tokens and diff_id_count == 0),
        "only_in_diffusion_example": only_diff[:20],
        "only_in_llm_example": only_llm[:20],
    }
    return report


def prompt_to_model_text(tokenizer, prompt_ids: torch.Tensor, prompt_mask: Optional[torch.Tensor] = None) -> str:
    if prompt_mask is not None:
        valid_ids = prompt_ids[prompt_mask.to(torch.bool)]
    else:
        valid_ids = prompt_ids
    return tokenizer.decode(valid_ids.detach().cpu().tolist(), skip_special_tokens=False)


def encode_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer {tokenizer.__class__.__name__} is not fast and cannot return offset mappings."
        )
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if len(input_ids) != len(offsets):
        raise RuntimeError("Tokenizer offsets and input ids are misaligned.")
    return input_ids, offsets


def block_char_spans_from_diffusion_tokenizer(
    tokenizer,
    completion_ids: Sequence[int],
    block_size: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    prefix_char_lengths: List[int] = []
    for prefix_end in range(len(completion_ids) + 1):
        prefix_text = tokenizer.decode(
            list(completion_ids[:prefix_end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prefix_char_lengths.append(len(prefix_text))

    for block_index in range((len(completion_ids) + block_size - 1) // block_size):
        token_start = block_index * block_size
        token_end = min(token_start + block_size, len(completion_ids))
        char_start = prefix_char_lengths[token_start]
        char_end = prefix_char_lengths[token_end]
        spans.append((char_start, char_end))
    return spans


def assign_offsets_to_blocks(
    token_offsets: Sequence[Tuple[int, int]],
    block_spans: Sequence[Tuple[int, int]],
) -> List[int]:
    assignment: List[int] = []
    for token_start, token_end in token_offsets:
        best_index = None
        best_overlap = -1
        for idx, (block_start, block_end) in enumerate(block_spans):
            overlap = max(0, min(token_end, block_end) - max(token_start, block_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = idx
        if best_index is None or best_overlap <= 0:
            raise RuntimeError(
                f"Could not assign token offset {(token_start, token_end)} to any block span."
            )
        assignment.append(best_index)
    return assignment


@torch.no_grad()
def generate_blockwise_with_scores(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int,
    block_size: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    top_k: Optional[int],
    top_p: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[float], List[List[float]]]:
    if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1:
        raise ValueError("generate_blockwise_with_scores expects a single prompt with shape [1, seq_len].")

    device = prompt_ids.device
    bs = 1
    x = torch.full((bs, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : prompt_ids.shape[1]] = prompt_ids.clone()
    prompt_index = x != mask_id

    num_blocks = (gen_length + block_size - 1) // block_size
    completion_token_logps = torch.full((bs, gen_length), float("nan"), device=device, dtype=torch.float32)
    block_logps: List[float] = []
    block_token_logps: List[List[float]] = []

    for block_index in range(num_blocks):
        start_idx = prompt_ids.shape[1] + block_index * block_size
        end_idx = min(start_idx + block_size, prompt_ids.shape[1] + gen_length)
        block_mask_index = x[:, start_idx:end_idx] == mask_id

        with (
            torch.autocast(device_type=device.type, dtype=dtype)
            if device.type == "cuda"
            else nullcontext()
        ):
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                raw_logits = model(x_).logits
                raw_logits, un_logits = torch.chunk(raw_logits, 2, dim=0)
                raw_logits = un_logits + (cfg_scale + 1.0) * (raw_logits - un_logits)
            else:
                raw_logits = model(x).logits

            sampling_logits = top_k_top_p_filtering(raw_logits, top_k=top_k, top_p=top_p)

            logits_for_sampling = add_gumbel_noise(sampling_logits, temperature=temperature, dtype=dtype)
            x0 = torch.argmax(logits_for_sampling, dim=-1)

            if remasking == "low_confidence":
                probabilities = F.softmax(sampling_logits.to(torch.float32), dim=-1)
                x0_p = torch.gather(probabilities, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, end_idx:] = -np.inf
            x0 = torch.where(x == mask_id, x0, x)
            confidence = torch.where(x == mask_id, x0_p, torch.full_like(x0_p, -np.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            current_block_token_logps: List[float] = []
            num_tokens = int(block_mask_index.sum().item())
            if num_tokens > 0:
                # Directly take masked positions inside the current block.
                # This is equivalent to the intended behavior and avoids unnecessary top-k logic.
                selected_rel = torch.where(block_mask_index[0])[0]
                if int(selected_rel.numel()) != num_tokens:
                    raise RuntimeError(
                        f"Block mask count mismatch at block {block_index}: "
                        f"num_tokens={num_tokens}, selected={int(selected_rel.numel())}"
                    )
                selected_positions = (selected_rel + start_idx).sort().values
                transfer_index[0, selected_positions] = True
                token_log_probs = F.log_softmax(raw_logits.to(torch.float32), dim=-1)
                for pos in selected_positions.tolist():
                    token_id = int(x0[0, pos].item())
                    token_logp = float(token_log_probs[0, pos, token_id].item())
                    completion_token_logps[0, pos - prompt_ids.shape[1]] = token_logp
                    current_block_token_logps.append(token_logp)

            x[transfer_index] = x0[transfer_index]
            block_logps.append(float(np.sum(current_block_token_logps)))
            block_token_logps.append(current_block_token_logps)

    return x, block_logps, block_token_logps


@torch.no_grad()
def score_with_llm(
    model,
    tokenizer,
    prompt_text: str,
    completion_text: str,
    block_char_spans: Sequence[Tuple[int, int]],
    device: torch.device,
) -> Tuple[List[float], List[List[float]], float]:
    full_text = prompt_text + completion_text
    full_ids, full_offsets = encode_with_offsets(tokenizer, full_text)
    prompt_ids, _ = encode_with_offsets(tokenizer, prompt_text)
    prompt_len = len(prompt_ids)
    prompt_char_len = len(prompt_text)

    shifted_block_spans = [
        (block_start + prompt_char_len, block_end + prompt_char_len) for block_start, block_end in block_char_spans
    ]

    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)

    token_logps: List[float] = []
    completion_offsets: List[Tuple[int, int]] = []
    for token_pos in range(prompt_len, len(full_ids)):
        target_id = int(full_ids[token_pos])
        token_logps.append(float(log_probs[0, token_pos - 1, target_id].item()))
        completion_offsets.append(full_offsets[token_pos])

    token_to_block = assign_offsets_to_blocks(completion_offsets, shifted_block_spans)
    block_logps: List[List[float]] = [[] for _ in range(len(shifted_block_spans))]
    for token_logp, block_index in zip(token_logps, token_to_block):
        block_logps[block_index].append(token_logp)

    block_sums = [float(np.sum(values)) for values in block_logps]
    return token_logps, block_logps, float(np.sum(block_sums))


def corrcoef_safe(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_rank = pd.Series(list(x)).rank(method="average").to_numpy()
    y_rank = pd.Series(list(y)).rank(method="average").to_numpy()
    return corrcoef_safe(x_rank, y_rank)


def compute_ess_from_log_weights(log_weights: Sequence[float]) -> ESSResult:
    lw = np.asarray(list(log_weights), dtype=np.float64)
    n = int(lw.size)
    if n == 0:
        return ESSResult(
            ess=float("nan"),
            ess_ratio=float("nan"),
            log_weights_mean=float("nan"),
            log_weights_std=float("nan"),
            n_samples=0,
        )

    max_lw = float(np.max(lw))
    stabilized = lw - max_lw
    log_norm = max_lw + float(np.log(np.sum(np.exp(stabilized))))
    normalized = np.exp(lw - log_norm)
    ess = float(1.0 / np.sum(normalized * normalized))
    return ESSResult(
        ess=ess,
        ess_ratio=float(ess / n),
        log_weights_mean=float(np.mean(lw)),
        log_weights_std=float(np.std(lw)),
        n_samples=n,
    )


def summarize_ess(
    sample_rows: List[Dict[str, Any]],
    block_rows: List[Dict[str, Any]],
    vocab_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Sequence-level ESS is always available because each model scores the same completion text.
    seq_logw_diff_over_llm: List[float] = []
    for row in sample_rows:
        if np.isfinite(row["total_logp_diffusion"]) and np.isfinite(row["total_logp_llm"]):
            seq_logw_diff_over_llm.append(float(row["total_logp_diffusion"] - row["total_logp_llm"]))

    # Token-level ESS requires index-aligned token log probs from both models.
    tok_logw_diff_over_llm: List[float] = []
    token_aligned_samples = 0
    token_misaligned_samples = 0
    for row in sample_rows:
        diff_list = row.get("completion_token_logps_diffusion", [])
        llm_list = row.get("completion_token_logps_llm", [])
        if len(diff_list) != len(llm_list):
            token_misaligned_samples += 1
            continue
        token_aligned_samples += 1
        for d, l in zip(diff_list, llm_list):
            if np.isfinite(d) and np.isfinite(l):
                tok_logw_diff_over_llm.append(float(d - l))

    # Block-level ESS is robust to tokenizer mismatch because spans are text-aligned by character offsets.
    blk_logw_diff_over_llm = [
        float(row["diff_block_logp"] - row["llm_block_logp"])
        for row in block_rows
        if np.isfinite(row["diff_block_logp"]) and np.isfinite(row["llm_block_logp"])
    ]

    seq_forward = compute_ess_from_log_weights(seq_logw_diff_over_llm)
    tok_forward = compute_ess_from_log_weights(tok_logw_diff_over_llm)
    blk_forward = compute_ess_from_log_weights(blk_logw_diff_over_llm)

    seq_backward = compute_ess_from_log_weights([-x for x in seq_logw_diff_over_llm])
    tok_backward = compute_ess_from_log_weights([-x for x in tok_logw_diff_over_llm])
    blk_backward = compute_ess_from_log_weights([-x for x in blk_logw_diff_over_llm])

    id_mapping_exact_match = True
    if vocab_report is not None:
        id_mapping_exact_match = bool(vocab_report.get("id_mapping_exact_match", True))

    return {
        "sequence_level_diffusion_over_llm": asdict(seq_forward),
        "sequence_level_llm_over_diffusion": asdict(seq_backward),
        "token_level_diffusion_over_llm": asdict(tok_forward),
        "token_level_llm_over_diffusion": asdict(tok_backward),
        "block_level_diffusion_over_llm": asdict(blk_forward),
        "block_level_llm_over_diffusion": asdict(blk_backward),
        "token_alignment": {
            "aligned_samples": int(token_aligned_samples),
            "misaligned_samples": int(token_misaligned_samples),
            "token_level_available": bool(token_aligned_samples > 0 and tok_forward.n_samples > 0),
            "token_level_reliable": bool(
                token_aligned_samples > 0 and tok_forward.n_samples > 0 and id_mapping_exact_match
            ),
            "note": (
                "Token-level ESS is reliable only when token log-prob arrays are index-aligned and token id mapping is exact. "
                f"id_mapping_exact_match={id_mapping_exact_match}."
            ),
        },
    }


def summarize_block_rows(block_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    diff = np.asarray([row["diff_block_logp"] for row in block_rows], dtype=float)
    llm = np.asarray([row["llm_block_logp"] for row in block_rows], dtype=float)
    gap = np.asarray([row["block_gap"] for row in block_rows], dtype=float)
    summary = {
        "num_block_rows": int(len(block_rows)),
        "mean_diff_block_logp": float(np.mean(diff)) if len(diff) else float("nan"),
        "mean_llm_block_logp": float(np.mean(llm)) if len(llm) else float("nan"),
        "mean_block_gap": float(np.mean(gap)) if len(gap) else float("nan"),
        "median_block_gap": float(np.median(gap)) if len(gap) else float("nan"),
        "mean_abs_block_gap": float(np.mean(np.abs(gap))) if len(gap) else float("nan"),
        "pearson_diff_vs_llm": corrcoef_safe(diff, llm),
        "spearman_diff_vs_llm": spearman_safe(diff, llm),
    }
    return summary


def summarize_sample_rows(sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_gap = np.asarray([row["total_gap"] for row in sample_rows], dtype=float)
    diff = np.asarray([row["total_logp_diffusion"] for row in sample_rows], dtype=float)
    llm = np.asarray([row["total_logp_llm"] for row in sample_rows], dtype=float)
    summary = {
        "num_samples": int(len(sample_rows)),
        "mean_total_gap": float(np.mean(total_gap)) if len(total_gap) else float("nan"),
        "median_total_gap": float(np.median(total_gap)) if len(total_gap) else float("nan"),
        "mean_abs_total_gap": float(np.mean(np.abs(total_gap))) if len(total_gap) else float("nan"),
        "pearson_total_diff_vs_llm": corrcoef_safe(diff, llm),
        "spearman_total_diff_vs_llm": spearman_safe(diff, llm),
    }
    return summary


def plot_results(block_df: pd.DataFrame, sample_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not block_df.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(block_df["block_gap"], bins=min(40, max(10, len(block_df) // 4)), color="#1f77b4", alpha=0.85)
        plt.title("Block log-prob gap distribution")
        plt.xlabel("diffusion block logp - llm block logp")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plots_dir / "block_gap_hist.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.scatter(block_df["diff_block_logp"], block_df["llm_block_logp"], s=10, alpha=0.55)
        lims = [
            min(block_df["diff_block_logp"].min(), block_df["llm_block_logp"].min()),
            max(block_df["diff_block_logp"].max(), block_df["llm_block_logp"].max()),
        ]
        plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)
        plt.title("Diffusion vs LLM block logp")
        plt.xlabel("diffusion block logp")
        plt.ylabel("llm block logp")
        plt.tight_layout()
        plt.savefig(plots_dir / "diff_vs_llm_scatter.png", dpi=200)
        plt.close()

        block_index_summary = block_df.groupby("block_index")["block_gap"].agg(["mean", "std", "count"]).reset_index()
        plt.figure(figsize=(9, 5))
        plt.errorbar(
            block_index_summary["block_index"],
            block_index_summary["mean"],
            yerr=block_index_summary["std"].fillna(0.0),
            marker="o",
            capsize=3,
            linewidth=1.5,
        )
        plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
        plt.title("Mean block gap by block index")
        plt.xlabel("block index")
        plt.ylabel("mean gap ± std")
        plt.tight_layout()
        plt.savefig(plots_dir / "gap_by_block_index.png", dpi=200)
        plt.close()

    if not sample_df.empty:
        plt.figure(figsize=(8, 5))
        plt.boxplot(sample_df["total_gap"], vert=True, widths=0.35)
        plt.xticks([1], ["samples"])
        plt.ylabel("total gap")
        plt.title("Sample-level total gap distribution")
        plt.tight_layout()
        plt.savefig(plots_dir / "per_sample_total_gap_box.png", dpi=200)
        plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blockwise diffusion vs LLM shift analysis")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_MAP.keys()), default="gsm8k")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--diffusion_model_path", type=str, required=True)
    parser.add_argument("--diffusion_checkpoint_path", type=str, default="")
    parser.add_argument("--llm_model_path", type=str, default="")
    parser.add_argument("--llm_tokenizer_path", type=str, default="")
    parser.add_argument("--output_dir", type=Path, default=Path("sampling/results"))
    parser.add_argument("--run_name", type=str, default="shift_analysis")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--math500_subset", type=str, choices=["all", "val", "test"], default="all")
    parser.add_argument("--math500_split_file", type=str, default="")
    parser.add_argument("--math500_split_seed", type=int, default=42)
    parser.add_argument("--allow_tokenizer_mismatch", action="store_true")
    parser.add_argument("--strict_vocab_match", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    init_seed(args.sample_seed)

    if not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {args.top_p}")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {args.top_k}")
    if args.top_k == 0:
        args.top_k = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype)

    output_dir = args.output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading diffusion tokenizer from {args.diffusion_model_path}")
    diffusion_tokenizer = load_tokenizer(args.diffusion_model_path)
    if args.mask_id is None:
        if diffusion_tokenizer.mask_token_id is not None:
            mask_id = int(diffusion_tokenizer.mask_token_id)
        else:
            mask_id = 126336
    else:
        mask_id = args.mask_id

    print(f"Loading diffusion model from {args.diffusion_model_path}")
    diffusion_model = load_diffusion_model(
        args.diffusion_model_path,
        args.diffusion_checkpoint_path,
        dtype=dtype,
        device=device,
    )

    llm_model_path = args.llm_model_path or args.diffusion_model_path
    llm_tokenizer_path = args.llm_tokenizer_path or llm_model_path
    print(f"Loading LLM tokenizer from {llm_tokenizer_path}")
    llm_tokenizer = load_tokenizer(llm_tokenizer_path)

    vocab_report = compare_vocabularies(diffusion_tokenizer, llm_tokenizer)
    print(
        "Vocabulary check: "
        f"token_set_exact_match={vocab_report['token_set_exact_match']}, "
        f"id_mapping_exact_match={vocab_report['id_mapping_exact_match']}, "
        f"shared={vocab_report['shared_token_count']}"
    )
    if args.strict_vocab_match and not vocab_report["id_mapping_exact_match"]:
        raise RuntimeError(
            "Vocabulary mismatch detected and --strict_vocab_match is enabled. "
            "Please use tokenizer-compatible models or disable strict check."
        )

    print(f"Loading LLM model from {llm_model_path}")
    llm_model = load_llm_model(llm_model_path, dtype=dtype, device=device)

    if (llm_tokenizer_path != args.diffusion_model_path) and not args.allow_tokenizer_mismatch:
        warnings.warn(
            "LLM tokenizer differs from the diffusion tokenizer. The script will still compare blocks by text spans, "
            "but you should verify that this comparison matches your experimental intent."
        )

    dataset = build_dataset(diffusion_tokenizer, args)
    if args.num_samples > 0 and args.num_samples < len(dataset):
        sample_count = args.num_samples
    else:
        sample_count = len(dataset)

    if sample_count == 0:
        raise RuntimeError("No samples selected for analysis.")

    sample_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []

    for sample_index in tqdm(range(sample_count), desc="Processing samples"):
        prompt, question, answer = dataset[sample_index]
        dataset_index = int(dataset.subsample[sample_index]) if hasattr(dataset, "subsample") else sample_index

        prompt_inputs = diffusion_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs.get("attention_mask")
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device)

        if args.max_prompt_length is not None and prompt_ids.size(1) > args.max_prompt_length:
            prompt_ids = prompt_ids[:, -args.max_prompt_length :]
            if prompt_mask is not None:
                prompt_mask = prompt_mask[:, -args.max_prompt_length :]

        model_prompt_text = prompt_to_model_text(
            diffusion_tokenizer,
            prompt_ids[0],
            prompt_mask[0] if prompt_mask is not None else None,
        )

        generated_ids, diffusion_block_logps, diffusion_block_token_logps = generate_blockwise_with_scores(
            model=diffusion_model,
            prompt_ids=prompt_ids,
            gen_length=args.gen_length,
            block_size=args.block_size,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            mask_id=mask_id,
            top_k=args.top_k,
            top_p=args.top_p,
            dtype=dtype,
        )

        completion_ids = generated_ids[0, prompt_ids.shape[1] :]
        completion_text = diffusion_tokenizer.decode(completion_ids.detach().cpu().tolist(), skip_special_tokens=False)
        completion_tokens = diffusion_tokenizer.convert_ids_to_tokens(completion_ids.detach().cpu().tolist())

        block_char_spans = block_char_spans_from_diffusion_tokenizer(
            diffusion_tokenizer,
            completion_ids.detach().cpu().tolist(),
            args.block_size,
        )

        llm_completion_token_logps, llm_block_token_logps, _ = score_with_llm(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt_text=model_prompt_text,
            completion_text=completion_text,
            block_char_spans=block_char_spans,
            device=device,
        )

        # if len(llm_completion_token_logps) != len(completion_ids):
        #     warnings.warn(
        #         f"LLM token count ({len(llm_completion_token_logps)}) and diffusion completion token count ({len(completion_ids)}) differ for sample {sample_index}."
        #     )

        block_records: List[BlockRecord] = []
        for block_index, (char_span, diff_block_logp, diff_token_logps, llm_token_logps) in enumerate(
            zip(block_char_spans, diffusion_block_logps, diffusion_block_token_logps, llm_block_token_logps)
        ):
            llm_block_logp = float(np.sum(llm_token_logps))
            block_gap = float(diff_block_logp - llm_block_logp)
            token_start = block_index * args.block_size
            token_end = min(token_start + args.block_size, len(completion_ids))
            block_records.append(
                BlockRecord(
                    block_index=block_index,
                    token_start=token_start,
                    token_end=token_end,
                    char_start=int(char_span[0]),
                    char_end=int(char_span[1]),
                    token_ids=[int(x) for x in completion_ids[token_start:token_end].tolist()],
                    token_logps_diffusion=[float(x) for x in diff_token_logps],
                    token_logps_llm=[float(x) for x in llm_token_logps],
                    block_logp_diffusion=float(diff_block_logp),
                    block_logp_llm=llm_block_logp,
                    block_gap=block_gap,
                )
            )

            block_rows.append(
                {
                    "sample_index": sample_index,
                    "dataset_index": dataset_index,
                    "block_index": block_index,
                    "token_start": token_start,
                    "token_end": token_end,
                    "char_start": int(char_span[0]),
                    "char_end": int(char_span[1]),
                    "diff_block_logp": float(diff_block_logp),
                    "llm_block_logp": llm_block_logp,
                    "block_gap": block_gap,
                    "num_tokens": token_end - token_start,
                }
            )

        total_diffusion = float(np.sum(diffusion_block_logps))
        total_llm = float(np.sum([row.block_logp_llm for row in block_records]))
        total_gap = float(total_diffusion - total_llm)
        block_gaps = [row.block_gap for row in block_records]

        diffusion_token_logps = [float("nan")] * len(completion_ids)
        for block_record in block_records:
            for offset, token_logp in enumerate(block_record.token_logps_diffusion):
                diffusion_token_logps[block_record.token_start + offset] = token_logp

        sample_record = SampleRecord(
            sample_index=sample_index,
            dataset_index=dataset_index,
            question=question,
            original_prompt=prompt,
            model_prompt_text=model_prompt_text,
            completion_text=completion_text,
            ground_truth=answer,
            completion_ids=[int(x) for x in completion_ids.tolist()],
            completion_tokens=completion_tokens,
            completion_token_logps_diffusion=diffusion_token_logps,
            completion_token_logps_llm=[float(x) for x in llm_completion_token_logps],
            blocks=block_records,
            total_logp_diffusion=total_diffusion,
            total_logp_llm=total_llm,
            total_gap=total_gap,
            mean_block_gap=float(np.mean(block_gaps)) if block_gaps else float("nan"),
            mean_abs_block_gap=float(np.mean(np.abs(block_gaps))) if block_gaps else float("nan"),
        )

        sample_rows.append(asdict(sample_record))

        # print(
        #     f"[{sample_index + 1}/{sample_count}] sample={dataset_index} total_gap={total_gap:.4f} "
        #     f"mean_block_gap={sample_record.mean_block_gap:.4f}"
        # )

    sample_jsonl = output_dir / "details.jsonl"
    sample_csv = output_dir / "sample_summary.csv"
    block_csv = output_dir / "block_summary.csv"
    summary_json = output_dir / "run_summary.json"

    write_jsonl(sample_jsonl, sample_rows)

    sample_df = pd.DataFrame(
        [
            {
                "sample_index": row["sample_index"],
                "dataset_index": row["dataset_index"],
                "question": row["question"],
                "total_logp_diffusion": row["total_logp_diffusion"],
                "total_logp_llm": row["total_logp_llm"],
                "total_gap": row["total_gap"],
                "mean_block_gap": row["mean_block_gap"],
                "mean_abs_block_gap": row["mean_abs_block_gap"],
                "num_blocks": len(row["blocks"]),
            }
            for row in sample_rows
        ]
    )
    sample_df.to_csv(sample_csv, index=False)

    block_df = pd.DataFrame(block_rows)
    block_df.to_csv(block_csv, index=False)

    block_summary = summarize_block_rows(block_rows)
    sample_summary = summarize_sample_rows(sample_rows)
    ess_summary = summarize_ess(sample_rows, block_rows, vocab_report=vocab_report)

    summary = {
        "config": {
            "dataset": args.dataset,
            "dataset_split": "train",
            "num_samples": sample_count,
            "sample_seed": args.sample_seed,
            "few_shot": args.few_shot,
            "diffusion_model_path": args.diffusion_model_path,
            "diffusion_checkpoint_path": args.diffusion_checkpoint_path,
            "llm_model_path": llm_model_path,
            "llm_tokenizer_path": llm_tokenizer_path,
            "gen_length": args.gen_length,
            "block_size": args.block_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "mask_id": mask_id,
            "dtype": args.dtype,
            "max_prompt_length": args.max_prompt_length,
        },
        "vocab_check": vocab_report,
        "sample_summary": sample_summary,
        "block_summary": block_summary,
        "ess": ess_summary,
        "outputs": {
            "details_jsonl": str(sample_jsonl),
            "sample_csv": str(sample_csv),
            "block_csv": str(block_csv),
            "plots_dir": str(output_dir / "plots"),
        },
    }

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=json_default)

    plot_results(block_df, sample_df, output_dir)
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()