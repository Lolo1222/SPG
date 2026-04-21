#!/usr/bin/env python3
"""Analyze distribution shift between two autoregressive LLMs with ESS metrics.

Workflow:
1) Sample completions from model A
2) Score the same sampled tokens under model A and model B
3) Compute token-level and sequence-level ESS in both directions:
   - a_over_b: w = pi_A / pi_B
   - b_over_a: w = pi_B / pi_A
"""

from __future__ import annotations

from tqdm import tqdm
import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    elif tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token


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
    cnt = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
            cnt += 1
    return cnt


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
        raise NotImplementedError("Countdown train split loader is not provided in this repo.")

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
    else:
        dataset.subsample = np.arange(len(train_records))
    return dataset


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
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


def apply_sampling_policy(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
) -> torch.Tensor:
    x = logits.to(torch.float32)
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    x = x / temperature
    x = top_k_top_p_filtering(x, top_k=top_k, top_p=top_p)
    return x


@torch.no_grad()
def generate_from_model_a(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    do_sample: bool,
    eos_token_id: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return completion token ids and their raw log probs under model A.

    Sampling can still use temperature/top-k/top-p policy, but logged probabilities
    are always computed from raw (unfiltered) model logits.
    """
    device = prompt_ids.device
    seq = prompt_ids.clone()
    logps: List[float] = []
    generated: List[int] = []

    for _ in range(max_new_tokens):
        out = model(input_ids=seq)
        raw_step_logits = out.logits[:, -1, :].to(torch.float32)
        raw_step_log_probs = F.log_softmax(raw_step_logits, dim=-1)

        policy_step_logits = apply_sampling_policy(
            raw_step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        policy_step_log_probs = F.log_softmax(policy_step_logits, dim=-1)

        if do_sample:
            probs = torch.exp(policy_step_log_probs)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(policy_step_log_probs, dim=-1, keepdim=True)

        token_id = int(next_token.item())
        token_logp = float(raw_step_log_probs[0, token_id].item())

        generated.append(token_id)
        logps.append(token_logp)
        seq = torch.cat([seq, next_token.to(device)], dim=-1)

        if eos_token_id is not None and token_id == eos_token_id:
            break

    completion_ids = torch.tensor(generated, dtype=torch.long, device=device)
    logp_a = torch.tensor(logps, dtype=torch.float32, device=device)
    return completion_ids, logp_a


@torch.no_grad()
def score_completion_tokens(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
) -> torch.Tensor:
    """Return raw-model log probs of completion tokens, shape [T]."""
    if completion_ids.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=prompt_ids.device)

    full_ids = torch.cat([prompt_ids, completion_ids.unsqueeze(0)], dim=-1)
    logits = model(input_ids=full_ids).logits[:, :-1, :]

    p = prompt_ids.size(1)
    t = completion_ids.size(0)
    step_logits = logits[:, p - 1 : p - 1 + t, :].to(torch.float32)
    log_probs = F.log_softmax(step_logits, dim=-1)
    token_logp = log_probs.gather(-1, completion_ids.view(1, t, 1)).squeeze(0).squeeze(-1)
    return token_logp.to(torch.float32)


def build_padded_logprob_tensors(
    seq_logp_a: List[torch.Tensor],
    seq_logp_b: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(seq_logp_a)
    max_t = max((x.numel() for x in seq_logp_a), default=0)

    logp_a = torch.full((n, max_t), float("nan"), dtype=torch.float32)
    logp_b = torch.full((n, max_t), float("nan"), dtype=torch.float32)
    mask = torch.zeros((n, max_t), dtype=torch.bool)

    for i, (a, b) in enumerate(zip(seq_logp_a, seq_logp_b)):
        if a.numel() != b.numel():
            raise ValueError(f"Length mismatch at sample {i}: A={a.numel()}, B={b.numel()}")
        t = a.numel()
        if t == 0:
            continue
        logp_a[i, :t] = a.detach().cpu()
        logp_b[i, :t] = b.detach().cpu()
        mask[i, :t] = True

    return logp_a, logp_b, mask


def _ess_from_logw(log_w: torch.Tensor) -> float:
    if log_w.numel() == 0:
        return float("nan")
    lw = log_w - torch.logsumexp(log_w, dim=0)
    wn = torch.exp(lw)
    return float((1.0 / (wn * wn).sum()).item())


def compute_token_level_ess_from_logprobs(
    logp_a: torch.Tensor,
    logp_b: torch.Tensor,
    attention_mask: torch.Tensor,
    direction: str,
) -> ESSResult:
    finite_mask = attention_mask & torch.isfinite(logp_a) & torch.isfinite(logp_b)
    if direction == "a_over_b":
        log_w = logp_a - logp_b
    elif direction == "b_over_a":
        log_w = logp_b - logp_a
    else:
        raise ValueError(f"Unknown direction: {direction}")

    log_w_flat = log_w[finite_mask]
    n = int(log_w_flat.numel())
    if n == 0:
        return ESSResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    ess = _ess_from_logw(log_w_flat)
    return ESSResult(
        ess=ess,
        ess_ratio=float(ess / n),
        log_weights_mean=float(log_w_flat.mean().item()),
        log_weights_std=float(log_w_flat.std().item()),
        n_samples=n,
    )


def compute_sequence_level_ess_from_logprobs(
    logp_a: torch.Tensor,
    logp_b: torch.Tensor,
    attention_mask: torch.Tensor,
    direction: str,
) -> ESSResult:
    finite_mask = attention_mask & torch.isfinite(logp_a) & torch.isfinite(logp_b)
    valid_len = attention_mask.sum(dim=-1)
    finite_len = finite_mask.sum(dim=-1)
    valid_seq = (valid_len > 0) & (valid_len == finite_len)

    if direction == "a_over_b":
        per_token = logp_a - logp_b
    elif direction == "b_over_a":
        per_token = logp_b - logp_a
    else:
        raise ValueError(f"Unknown direction: {direction}")

    per_token = torch.where(attention_mask, per_token, torch.zeros_like(per_token))
    log_w_seq = per_token.sum(dim=-1)
    log_w_seq = log_w_seq[valid_seq]

    n = int(log_w_seq.numel())
    if n == 0:
        return ESSResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    ess = _ess_from_logw(log_w_seq)
    return ESSResult(
        ess=ess,
        ess_ratio=float(ess / n),
        log_weights_mean=float(log_w_seq.mean().item()),
        log_weights_std=float(log_w_seq.std().item()),
        n_samples=n,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-vs-LLM shift analysis with ESS")
    p.add_argument("--dataset", type=str, choices=list(DATASET_MAP.keys()), default="gsm8k")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--few_shot", type=int, default=0)

    p.add_argument("--model_a_path", type=str, required=True, help="Sampling/source model A")
    p.add_argument("--model_b_path", type=str, required=True, help="Target model B")
    p.add_argument("--tokenizer_path", type=str, default="", help="Optional shared tokenizer path")

    p.add_argument("--output_dir", type=Path, default=Path("sampling/results"))
    p.add_argument("--run_name", type=str, default="llm_llm_shift")

    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--stop_on_eos", action="store_true")

    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--max_prompt_length", type=int, default=None)

    p.add_argument("--strict_vocab_match", action="store_true")
    p.add_argument("--math500_subset", type=str, choices=["all", "val", "test"], default="all")
    p.add_argument("--math500_split_file", type=str, default="")
    p.add_argument("--math500_split_seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    init_seed(args.sample_seed)

    if not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {args.top_p}")
    if args.top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {args.top_k}")
    top_k = None if args.top_k == 0 else args.top_k

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype)

    output_dir = args.output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.tokenizer_path or args.model_a_path
    print(f"Loading tokenizer: {tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    ensure_tokenizer_padding(tok)

    print(f"Loading model A: {args.model_a_path}")
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a_path,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()
    print(f"Loading model B: {args.model_b_path}")
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b_path,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()

    dataset = build_dataset(tok, args)
    sample_count = len(dataset.subsample)
    if sample_count == 0:
        raise RuntimeError("No samples selected.")

    eos_id = tok.eos_token_id if args.stop_on_eos else None

    seq_logp_a: List[torch.Tensor] = []
    seq_logp_b: List[torch.Tensor] = []
    detail_rows: List[Dict[str, Any]] = []

    for i in tqdm(range(sample_count), desc="Processing samples"):
        prompt, question, answer = dataset[i]
        dataset_index = int(dataset.subsample[i])

        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = enc["input_ids"].to(device)
        if args.max_prompt_length is not None and prompt_ids.size(1) > args.max_prompt_length:
            prompt_ids = prompt_ids[:, -args.max_prompt_length :]

        completion_ids, logp_a_gen = generate_from_model_a(
            model=model_a,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            eos_token_id=eos_id,
        )

        logp_b = score_completion_tokens(
            model=model_b,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            temperature=args.temperature,
            top_k=top_k,
            top_p=args.top_p,
        )

        if completion_ids.numel() != logp_b.numel():
            raise RuntimeError("Completion/token logprob length mismatch.")

        seq_logp_a.append(logp_a_gen.detach().cpu())
        seq_logp_b.append(logp_b.detach().cpu())

        if completion_ids.numel() > 0:
            seq_logw_a_over_b = float((logp_a_gen - logp_b).sum().item())
            seq_logw_b_over_a = float((logp_b - logp_a_gen).sum().item())
        else:
            seq_logw_a_over_b = float("nan")
            seq_logw_b_over_a = float("nan")

        detail_rows.append(
            {
                "sample_index": i,
                "dataset_index": dataset_index,
                "question": question,
                "ground_truth": answer,
                "prompt_text": prompt,
                "completion_text": tok.decode(completion_ids.tolist(), skip_special_tokens=False),
                "completion_ids": completion_ids.tolist(),
                "logp_a": logp_a_gen.tolist(),
                "logp_b": logp_b.tolist(),
                "seq_logw_a_over_b": seq_logw_a_over_b,
                "seq_logw_b_over_a": seq_logw_b_over_a,
                "completion_len": int(completion_ids.numel()),
            }
        )

        sum_a = float(logp_a_gen.sum().item()) if completion_ids.numel() else float("nan")
        sum_b = float(logp_b.sum().item()) if completion_ids.numel() else float("nan")
        # print(f"[{i + 1}/{sample_count}] sample={dataset_index} len={completion_ids.numel()} sum_logp_a={sum_a:.4f} sum_logp_b={sum_b:.4f}")

    logp_a_mat, logp_b_mat, attn_mask = build_padded_logprob_tensors(seq_logp_a, seq_logp_b)

    token_ess_a_over_b = compute_token_level_ess_from_logprobs(logp_a_mat, logp_b_mat, attn_mask, "a_over_b")
    token_ess_b_over_a = compute_token_level_ess_from_logprobs(logp_a_mat, logp_b_mat, attn_mask, "b_over_a")
    seq_ess_a_over_b = compute_sequence_level_ess_from_logprobs(logp_a_mat, logp_b_mat, attn_mask, "a_over_b")
    seq_ess_b_over_a = compute_sequence_level_ess_from_logprobs(logp_a_mat, logp_b_mat, attn_mask, "b_over_a")

    details_jsonl = output_dir / "details.jsonl"
    sample_csv = output_dir / "sample_summary.csv"
    run_summary_json = output_dir / "run_summary.json"

    write_jsonl(details_jsonl, detail_rows)

    sample_df = pd.DataFrame(
        [
            {
                "sample_index": r["sample_index"],
                "dataset_index": r["dataset_index"],
                "completion_len": r["completion_len"],
                "seq_logw_a_over_b": r["seq_logw_a_over_b"],
                "seq_logw_b_over_a": r["seq_logw_b_over_a"],
            }
            for r in detail_rows
        ]
    )
    sample_df.to_csv(sample_csv, index=False)

    summary = {
        "config": {
            "dataset": args.dataset,
            "num_samples": sample_count,
            "sample_seed": args.sample_seed,
            "few_shot": args.few_shot,
            "model_a_path": args.model_a_path,
            "model_b_path": args.model_b_path,
            "tokenizer_path": tokenizer_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": top_k,
            "do_sample": args.do_sample,
            "stop_on_eos": args.stop_on_eos,
            "dtype": args.dtype,
            "max_prompt_length": args.max_prompt_length,
        },
        "ess": {
            "token_level_a_over_b": asdict(token_ess_a_over_b),
            "token_level_b_over_a": asdict(token_ess_b_over_a),
            "sequence_level_a_over_b": asdict(seq_ess_a_over_b),
            "sequence_level_b_over_a": asdict(seq_ess_b_over_a),
        },
        "outputs": {
            "details_jsonl": str(details_jsonl),
            "sample_csv": str(sample_csv),
            "run_summary_json": str(run_summary_json),
        },
    }

    with run_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== ESS Summary =====")
    print(f"Token ESS (A/B): {token_ess_a_over_b.ess:.4f} | ratio={token_ess_a_over_b.ess_ratio:.6f}")
    print(f"Token ESS (B/A): {token_ess_b_over_a.ess:.4f} | ratio={token_ess_b_over_a.ess_ratio:.6f}")
    print(f"Seq   ESS (A/B): {seq_ess_a_over_b.ess:.4f} | ratio={seq_ess_a_over_b.ess_ratio:.6f}")
    print(f"Seq   ESS (B/A): {seq_ess_b_over_a.ess:.4f} | ratio={seq_ess_b_over_a.ess_ratio:.6f}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
