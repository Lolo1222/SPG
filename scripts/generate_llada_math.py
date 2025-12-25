#!/usr/bin/env python3
"""
Generate multiple answers from LLaDA on a random subset of the math train set.

Saves per-question generations and computes best-of-N accuracy using a simple answer-extraction
and matching heuristic. Supports specifying GPUs (comma-separated indices) and several generation
parameters including gen_length, temperature, diffusion_steps and decoding strategy (remasking).

Usage example:
  python scripts/generate_llada_math.py \
    --parquet /home/jwliu/midtrain/data/math/8k/train.parquet \
    --model_dir /home/jwliu/dlm/SPG/save_dir/hf_models/LLaDA-8B-Instruct \
    --out_dir /home/jwliu/dlm/SPG/save_dir \
    --num_samples 1000 --num_generations 16 --gen_length 512 --temperature 0.9 \
    --diffusion_steps 256 --decoding_strategy low_confidence --gpus 0

"""
from __future__ import annotations
import os
import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
import sys
import traceback
from pathlib import Path as _Path

# Ensure repo root is on sys.path so we can import eval.generate reliably when running from scripts/
repo_root = _Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# Prefer the repository's iterative generate (diffusion/unmask) used by eval.py for consistency
try:
    from eval.generate import generate as eval_generate
    print(f"Imported eval.generate from {eval_generate.__module__}")
except Exception as e:
    eval_generate = None
    print(f"Could not import eval.generate: {e}")
# Prefer the repository's iterative generate (diffusion/unmask) used by eval.py for consistency
try:
    from eval.generate import generate as eval_generate
except Exception:
    eval_generate = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="path to math train parquet")
    p.add_argument("--model_dir", required=True, help="path to local llada model dir")
    p.add_argument("--out_dir", required=True, help="directory to write outputs (jsonl + summary)")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--num_generations", type=int, default=16)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--diffusion_steps", type=int, default=256)
    p.add_argument("--decoding_strategy", type=str, default="low_confidence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default=None, help="comma-separated GPU ids (e.g. '0' or '0,1')")
    return p.parse_args()


def set_visible_gpus(gpus: Optional[str]):
    if gpus is None:
        return
    # set CUDA_VISIBLE_DEVICES so device indices inside the script map to these devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def load_data(parquet_path: str, num_samples: int, seed: int):
    df = pd.read_parquet(parquet_path)
    # try common column names
    if "question" not in df.columns and "problem" in df.columns:
        df = df.rename(columns={"problem": "question"})
    if "question" not in df.columns:
        raise RuntimeError(f"Could not find a 'question' column in {parquet_path}; columns: {list(df.columns)}")

    df_sample = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)
    return df_sample


def build_prompt(question: str) -> str:
    prefix = (
        "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
        "You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. Respond in the following format:\n<reasoning>\nYour reasoning here\n</reasoning>\n<answer>\n\\boxed{...}\n</answer>\n\n"
    )
    return prefix + question + "<|eot_id|>"


def extract_answer_from_text(text: str) -> Optional[str]:
    """Try to extract a final answer from model text.

    Strategy: 1) look for \boxed{...} 2) look for last math-like number token 3) fallback to last non-empty line
    """
    if not text:
        return None
    # 1) boxed: prefer a boxed numeric answer; ignore boxed placeholders like '...'
    m = re.search(r"\\\\boxed\{([^}]+)\}", text)
    if m:
        candidate = m.group(1).strip()
        # ignore placeholder dots
        if re.fullmatch(r"\.{1,}", candidate):
            pass
        else:
            return candidate

    # 2) find all number-like tokens and take last
    # 2) find all number-like tokens and take last (handles integers, decimals, scientific)
    nums = re.findall(r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text, flags=re.IGNORECASE)
    if nums:
        return nums[-1]

    # 3) take last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1]
    return None


def normalize_answer(a: Optional[str]) -> Optional[str]:
    if a is None:
        return None
    s = a.strip()
    # remove surrounding $ and punctuation
    s = s.strip(" $\n\t")
    # remove at most one trailing period (avoid turning '...' into empty string)
    if s.endswith('.'):
        s = s[:-1].rstrip()
    # if result is empty after normalization, return None to avoid empty-string matches
    if s == "":
        return None
    return s


def is_match(pred: Optional[str], target: Optional[str]) -> bool:
    if pred is None or target is None:
        return False
    p = normalize_answer(pred)
    t = normalize_answer(target)
    if p is None or t is None:
        return False
    if p == t:
        return True
    # try numeric comparison if both look numeric
    try:
        fp = float(p)
        ft = float(t)
        return abs(fp - ft) <= 1e-6
    except Exception:
        pass
    # fallback: check substring
    return p in t or t in p


def main():
    args = parse_args()
    set_visible_gpus(args.gpus)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect DDP launcher and initialize process group
    use_ddp = False
    local_rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ:
        use_ddp = True
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        print(f"DDP mode: local_rank={local_rank}, world_size={world_size}, device={device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data from {args.parquet} ...")
    df = load_data(args.parquet, args.num_samples, args.seed)
    print(f"Selected {len(df)} samples")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # try to use bfloat16 if available, else float16
    dtype = getattr(torch, "bfloat16", torch.float16)

    # attempt to initialize model with device_map auto (will use visible GPUs)
    try:
        if use_ddp:
            # In DDP mode, load into CPU (or requested dtype) then move to local device to avoid device_map
            model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, torch_dtype=dtype)
            model.to(device)
        else:
            model = AutoModel.from_pretrained(
                args.model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
            )
    except Exception as e:
        print("Warning: device_map='auto' load failed, falling back to single-device load:", e)
        model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, torch_dtype=dtype).to(device)

    model.eval()

    # per-rank outputs to avoid concurrent writes when using DDP
    rank_suffix = f"_rank{local_rank}" if use_ddp else ""
    out_jsonl = out_dir / f"llada_math_generations{rank_suffix}.jsonl"
    summary_path = out_dir / f"llada_math_generations_summary{rank_suffix}.json"

    total = 0
    correct_best_of = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for i in range(len(df)):
            # partition samples across ranks deterministically
            if use_ddp and (i % world_size) != local_rank:
                continue
            row = df.iloc[i]
            q = str(row["question"]) if not pd.isna(row["question"]) else ""
            # if dataset has 'answer' or 'target' column, pick one
            target = None
            # common column names for the reference answer in different math datasets
            if "answer" in row.index:
                target = str(row["answer"]) if not pd.isna(row["answer"]) else None
            elif "target" in row.index:
                target = str(row["target"]) if not pd.isna(row["target"]) else None
            elif "solution" in row.index:
                # math8k / similar datasets often store the answer in 'solution'
                target = str(row["solution"]) if not pd.isna(row["solution"]) else None

            prompt = build_prompt(q)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

            # move inputs to the correct device
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            # If eval-style iterative generate is available in the repo, prefer it for consistency with eval.py
            outputs = None
            if eval_generate is not None:
                try:
                    print("Calling eval.generate (iterative unmasking)...")
                    # Determine a valid mask_id to pass to eval.generate
                    # Prefer tokenizer.mask_token_id if available; otherwise use a reserved numeric mask id (not eos)
                    mask_id_val = tokenizer.mask_token_id if getattr(tokenizer, "mask_token_id", None) is not None else 126336
                    print(f"Using mask_id={mask_id_val} for eval.generate")
                    # Prepare prompt batch: eval.generate returns one sequence per prompt. To get multiple
                    # generations per question, repeat the prompt tensor `num_generations` times.
                    prompt_ids = inputs["input_ids"]
                    # To reduce peak GPU memory usage, avoid repeating the prompt into a large batch.
                    # Instead call eval_generate sequentially for each requested generation and
                    # concatenate the results. This trades runtime for much lower peak memory.
                    if args.num_generations > 1:
                        outputs_list = []
                        for _g in range(args.num_generations):
                            try:
                                outg = eval_generate(
                                    model,
                                    prompt_ids,
                                    tokenizer,
                                    steps=args.diffusion_steps,
                                    gen_length=args.gen_length,
                                    block_length=32,
                                    temperature=args.temperature,
                                    cfg_scale=0.0,
                                    remasking=args.decoding_strategy,
                                    mask_id=mask_id_val,
                                )
                                outputs_list.append(outg)
                            finally:
                                # free cache to reduce fragmentation between runs
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                        # concat along batch dimension
                        outputs = torch.cat(outputs_list, dim=0)
                    else:
                        prompt_batch = prompt_ids
                    # The eval.generate implementation calls torch.distributed.get_rank(); in single-process
                    # runs the default process group is not initialized which raises. Monkeypatch get_rank to
                    # return 0 temporarily to allow single-process execution.
                    _orig_get_rank = None
                    if not use_ddp:
                        import torch.distributed as _dist
                        if hasattr(_dist, "get_rank"):
                            _orig_get_rank = _dist.get_rank
                            try:
                                _dist.get_rank = lambda: 0
                            except Exception:
                                _orig_get_rank = None
                    try:
                        if args.num_generations > 1:
                            # outputs already set by sequential calls above
                            pass
                        else:
                            outputs = eval_generate(
                                model,
                                prompt_batch,
                                tokenizer,
                                steps=args.diffusion_steps,
                                gen_length=args.gen_length,
                                block_length=32,
                                temperature=args.temperature,
                                cfg_scale=0.0,
                                remasking=args.decoding_strategy,
                                mask_id=mask_id_val,
                            )
                        print("eval.generate completed successfully")
                    finally:
                        if _orig_get_rank is not None:
                            try:
                                _dist.get_rank = _orig_get_rank
                            except Exception:
                                pass
                except Exception as e:
                    print(f"Warning: eval.generate raised an exception, falling back to model.generate: {e}")
                    traceback.print_exc()
                    outputs = None

            if outputs is None:
                gen_kwargs = dict(
                    do_sample=True,
                    max_new_tokens=args.gen_length,
                    temperature=args.temperature,
                    num_return_sequences=args.num_generations,
                    use_cache=False,
                )

                # pass diffusion / decoding strategy kwargs if supported by model generation API
                gen_kwargs_custom = {
                    "diffusion_steps": args.diffusion_steps,
                    "remasking": args.decoding_strategy,
                }

                try:
                    # optimistic path: model.generate supports custom kwargs
                    outputs = model.generate(**inputs, **gen_kwargs, **gen_kwargs_custom)
                except Exception as e:
                    # some models raise ValueError if unknown kwargs are passed to generate
                    print(f"Warning: model.generate rejected custom kwargs ({e}), retrying without them")
                    outputs = model.generate(**inputs, **gen_kwargs)

            generations = []
            # outputs shape: (num_return_sequences, seq_len)
            for sid in range(outputs.size(0)):
                text = tokenizer.decode(outputs[sid].tolist(), skip_special_tokens=True)
                # strip the prompt prefix if present
                if text.startswith(prompt):
                    gen_text = text[len(prompt) :].strip()
                else:
                    gen_text = text.strip()
                generations.append(gen_text)

            extracted = [extract_answer_from_text(t) for t in generations]

            matches = [is_match(e, target) for e in extracted]
            best = any(matches)
            if best:
                correct_best_of += 1
            total += 1

            rec = {
                "idx": int(i),
                "question": q,
                "target": target,
                "generations": generations,
                "extracted_answers": extracted,
                "matches": matches,
                "best_of_N_correct": bool(best),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {"num_samples": total, "best_of_N_correct": correct_best_of, "best_of_N_accuracy": correct_best_of / max(1, total)}
    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print(f"Wrote per-sample generations to {out_jsonl}")
    print(f"Wrote summary to {summary_path}")
    print("Summary:", summary)

    # If running with DDP, merge per-rank outputs into a single file on rank 0
    if use_ddp:
        dist.barrier()
        if local_rank == 0:
            final_jsonl = out_dir / "llada_math_generations.jsonl"
            final_summary = out_dir / "llada_math_generations_summary.json"
            total_samples = 0
            total_correct = 0
            with open(final_jsonl, "w", encoding="utf-8") as fout_final:
                for r in range(world_size):
                    part = out_dir / f"llada_math_generations_rank{r}.jsonl"
                    if not part.exists():
                        continue
                    with open(part, "r", encoding="utf-8") as fr:
                        for line in fr:
                            fout_final.write(line)
            # aggregate summaries
            for r in range(world_size):
                part_sum = out_dir / f"llada_math_generations_summary_rank{r}.json"
                if not part_sum.exists():
                    continue
                with open(part_sum, "r", encoding="utf-8") as frs:
                    ps = json.load(frs)
                    total_samples += ps.get("num_samples", 0)
                    total_correct += ps.get("best_of_N_correct", 0)
            summary_agg = {"num_samples": total_samples, "best_of_N_correct": total_correct, "best_of_N_accuracy": total_correct / max(1, total_samples)}
            with open(final_summary, "w", encoding="utf-8") as ff:
                json.dump(summary_agg, ff, indent=2)
            print(f"Merged {world_size} rank files into {final_jsonl} and wrote aggregate summary to {final_summary}")
        dist.barrier()


if __name__ == "__main__":
    main()
