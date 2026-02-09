#!/usr/bin/env python3
"""
Generate multiple answers from LLaDA on a random subset of the math train set.

Saves per-question generations and computes best-of-N accuracy using a simple answer-extraction
and matching heuristic. Supports specifying GPUs (comma-separated indices) and several generation
parameters including gen_length, temperature, diffusion_steps and decoding strategy (remasking).

Usage example:

    CUDA_VISIBLE_DEVICES=2,3 \
    /root/miniconda3/envs/spg/bin/python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29502 \
    scripts/generate_llada_sudoku.py \
    --parquet dataset/train_sudoku_split_new.csv \
    --model_dir /root/Models/LLaDA-8B-Instruct \
    --out_dir dataset/sudoku \
    --num_samples 3000 \
    --num_generations 1 \
    --gen_length 256 \
    --temperature 0.3 \
    --diffusion_steps 256 \
    --decoding_strategy low_confidence  

"""

from __future__ import annotations
from tqdm import tqdm
import os
import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Any
import numpy as np
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

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if o is pd.NA:
        return None
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

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

def sudoku_get_puzzle_16(question: str) -> str:
    # EXACTLY follow eval puzzle extraction:
    q = (question or "").strip()
    if len(q) >= 16 and all(c.isdigit() or c == "0" for c in q[:16]):
        return q[:16]
    m = re.search(r"Sudoku puzzle: ([0-9]{16})", q)
    if m:
        return m.group(1)
    return ""  # let caller handle error

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

short_example_1 = "Question:\nSolve the following Sudoku puzzle: 3014002020004130\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 3 0 1 4\nR2: 0 0 2 0\nR3: 2 0 0 0\nR4: 4 1 3 0\n\nFill easy singles:\nR1 missing 2 → R1C2=2.\nR4 missing 2 → R4C4=2.\nBox D (R3-4,C3-4) then needs {1,4}; column4 can only accept 1 → R3C4=1, R3C3=4.\nR3 now missing 3 → R3C2=3.\nColumn1 missing 1 → R2C1=1.\nColumn2 missing 4 → R2C2=4.\nLast cell R2C4=3.\n\nFinal grid:\nR1: 3 2 1 4\nR2: 1 4 2 3\nR3: 2 3 4 1\nR4: 4 1 3 2\n</reasoning>\n<answer>\n3214142323414132\n</answer>"
short_example_2 = "Question:\nSolve the following Sudoku puzzle: 0000100420013142\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 0 0 0 0\nR2: 1 0 0 4\nR3: 2 0 0 1\nR4: 3 1 4 2\n\nFill easy singles:\nCol1 missing 4 → R1C1=4.\nCol4 missing 3 → R1C4=3.\nBox A (R1-2,C1-2) missing {2,3} and R1 now needs {1,2} → R1C2=2, R2C2=3.\nR1C3=1.\nR2 now missing 2 → R2C3=2.\nCol2 missing 4 → R3C2=4, then R3C3=3.\n\nFinal grid:\nR1: 4 2 1 3\nR2: 1 3 2 4\nR3: 2 4 3 1\nR4: 3 1 4 2\n</reasoning>\n<answer>\n4213132424313142\n</answer>"
short_example_3 = "Question:\nSolve the following Sudoku puzzle: 2001403002001420\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 2 0 0 1\nR2: 4 0 3 0\nR3: 0 2 0 0\nR4: 1 4 2 0\n\nFill easy singles:\nR1 missing {3,4}; Col2 can't be 1 so R1C2=3 → R1C3=4.\nR4 missing 3 → R4C4=3.\nCol4 missing {2,4}; R2 must take 2 → R2C4=2 → R2C2=1.\nCol1 missing 3 → R3C1=3.\nCol3 missing 1 → R3C3=1 → R3C4=4.\n\nFinal grid:\nR1: 2 3 4 1\nR2: 4 1 3 2\nR3: 3 2 1 4\nR4: 1 4 2 3\n</reasoning>\n<answer>\n2341413232141423\n</answer>"

def build_prompt(puzzle: str):
    # q16 = sudoku_get_puzzle_16(question)

    system_prompt = (
        f"{SUDOKU_SYSTEM_PROMPT}\n\n"
        f"{short_example_1}\n\n"
        f"{short_example_2}\n\n"
        f"{short_example_3}"
    )

    return [
        {
            "role": "user",
            "content": (
                f"{system_prompt}\n\n"
                f"Question: Solve the following Sudoku puzzle: {puzzle}\n"
                f"Answer:\n"
            ),
        }
    ]

def load_data(parquet_path: str, num_samples: int, seed: int):
    if parquet_path.endswith(".csv"):
        df = pd.read_csv(parquet_path, dtype=str)  
    else:
        df = pd.read_parquet(parquet_path)
    df = df.fillna("")

    # Sudoku CSV: Puzzle,Solution
    # if "question" not in df.columns and "Puzzle" in df.columns:
    #     df = df.rename(columns={"Puzzle": "question"})
    # if "ground_truth" not in df.columns and "Solution" in df.columns:
    #     df = df.rename(columns={"Solution": "ground_truth"})

    df_sample = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)
    return df_sample


def sudoku_extract_solution(raw_generation: str) -> str:
    solution_str = ""
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]
    for pattern in patterns:
        if solution_str:
            break
        m = re.search(pattern, raw_generation, re.DOTALL)
        if m and m.group(1).strip():
            solution_str = m.group(1).strip()

    solution_str = re.sub(r"\s", "", solution_str)

    if not solution_str:
        return ""
    if len(solution_str) < 16:
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        solution_str = solution_str[:16]
    return solution_str


def sudoku_score(question_16: str, gt_16: str, pred_16: str):
    # eval: 只算 puzzle 里为 0 的格子
    puzzle_str = question_16[:16]
    empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
    empty_cells = len(empty_indices)
    if empty_cells == 0:
        return 0, 0, 0.0

    if not pred_16:
        return empty_cells, 0, 0.0

    correct_cells = sum(1 for i in empty_indices if pred_16[i] == gt_16[i])
    acc = correct_cells / empty_cells
    return empty_cells, correct_cells, acc


def sudoku_is_correct(question_16: str, gt_16: str, raw_generation: str) -> bool:
    pred = sudoku_extract_solution(raw_generation)
    empty_cells, correct_cells, acc = sudoku_score(question_16, gt_16, pred)
    return (empty_cells > 0) and (correct_cells == empty_cells)

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
        
        num_gpus = torch.cuda.device_count()

        device_id = local_rank % num_gpus
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        
        # torch.cuda.set_device(local_rank)
        # device = torch.device(f"cuda:{local_rank}")
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
        # for i in range(len(df)):
        for i in tqdm(range(len(df)), disable=(local_rank != 0), desc="Processing samples"):
            # partition samples across ranks deterministically
            if use_ddp and (i % world_size) != local_rank:
                continue
            row = df.iloc[i]
            # print(row)
            # q = str(row["question"]) if not pd.isna(row["question"]) else ""
            puzzle = row["Puzzle"]
            # print(puzzle)
            # if dataset has 'answer' or 'target' column, pick one
            target = None
            # common column names for the reference answer in different math datasets
            if "answer" in row.index:
                target = row["answer"] if not pd.isna(row["answer"]) else None
            elif "target" in row.index:
                target = str(row["target"]) if not pd.isna(row["target"]) else None
            elif "Solution" in row.index:
                # math8k / similar datasets often store the answer in 'Solution'
                target = str(row["Solution"]) if not pd.isna(row["Solution"]) else None

            messages = build_prompt(puzzle)  # list[dict] with role/content

            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": torch.ones_like(input_ids).to(device),
            }

            # If eval-style iterative generate is available in the repo, prefer it for consistency with eval.py
            outputs = None
            if eval_generate is not None:
                try:
                    # print("Calling eval.generate (iterative unmasking)...")
                    # Determine a valid mask_id to pass to eval.generate
                    # Prefer tokenizer.mask_token_id if available; otherwise use a reserved numeric mask id (not eos)
                    mask_id_val = tokenizer.mask_token_id if getattr(tokenizer, "mask_token_id", None) is not None else 126336
                    # print(f"Using mask_id={mask_id_val} for eval.generate")
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
                        # print("eval.generate completed successfully")
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
            prompt_ids_1d = inputs["input_ids"][0]  # shape [L]
            L = prompt_ids_1d.shape[0]

            for sid in range(outputs.size(0)):
                out_ids = outputs[sid]

                # 如果输出确实以 prompt_ids 开头，就切掉 prompt；否则不切
                if out_ids.numel() >= L and torch.equal(out_ids[:L], prompt_ids_1d):
                    completion_ids = out_ids[L:]
                else:
                    completion_ids = out_ids

                gen_text = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=False).strip()
                gen_text = gen_text.replace("<|endoftext|>", "").replace("<|eot_id|>", "").strip()

                generations.append(gen_text)



            gt_16 = re.sub(r"\s", "", str(row.get("Solution", "")).strip())[:16]
            q_16 = sudoku_get_puzzle_16(str(row.get("Puzzle", "")))

            # 如果 puzzle 取不到 16 位，直接判错
            if len(q_16) != 16:
                extracted = ["" for _ in generations]
                matches = [False for _ in generations]
            else:
                extracted = [sudoku_extract_solution(t) for t in generations]
                matches = [sudoku_is_correct(q_16, gt_16, t) for t in generations]
            # per-generation scoring (eval-style)
            if len(q_16) == 16:
                scores = [sudoku_score(q_16, gt_16, sudoku_extract_solution(t)) for t in generations]
                # scores: (empty_cells, correct_cells, acc)
                empty_cells_list = [s[0] for s in scores]
                correct_cells_list = [s[1] for s in scores]
                accuracies = [s[2] for s in scores]
            else:
                empty_cells_list = [0 for _ in generations]
                correct_cells_list = [0 for _ in generations]
                accuracies = [0.0 for _ in generations]


            best = any(matches)  

            if best:
                correct_best_of += 1
            total += 1

            rec = {
                "idx": int(i),
                "Puzzle": puzzle,
                "Solution": row.get("Solution", ""),
                "generations": generations,
                "extracted_answers": extracted,
                "matches": matches,
                "empty_cells": empty_cells_list,
                "correct_cells": correct_cells_list,
                "accuracy": accuracies,
                "best_of_N_correct": bool(best),
            }


            fout.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


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
            final_jsonl = out_dir / "llada_sudoku_generations.jsonl"
            final_summary = out_dir / "llada_sudoku_generations_summary.json"
            total_samples = 0
            total_correct = 0
            with open(final_jsonl, "w", encoding="utf-8") as fout_final:
                for r in range(world_size):
                    part = out_dir / f"llada_sudoku_generations_rank{r}.jsonl"
                    if not part.exists():
                        continue
                    with open(part, "r", encoding="utf-8") as fr:
                        for line in fr:
                            fout_final.write(line)
            # aggregate summaries
            for r in range(world_size):
                part_sum = out_dir / f"llada_sudoku_generations_summary_rank{r}.json"
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
    print(">>> generate_llada_math.py entered main()", flush=True)
    main()
    
