#!/usr/bin/env python3
"""
Generate multiple answers from LLaDA on a random subset of the math train set.

Saves per-question generations and computes best-of-N accuracy using a simple answer-extraction
and matching heuristic. Supports specifying GPUs (comma-separated indices) and several generation
parameters including gen_length, temperature, diffusion_steps and decoding strategy (remasking).

Usage example:

CUDA_VISIBLE_DEVICES=2,5 \
/root/miniconda3/envs/spg/bin/python -m torch.distributed.run \
--nproc_per_node=1 --master_port=29517 \
scripts/generate_llada_countdown.py \
--parquet /root/jiawei/SPG/dataset/countdown/train-00000-of-00001.parquet \
--model_dir /root/Models/LLaDA-8B-Instruct \
--out_dir /root/jiawei/SPG/dataset/countdown \
--num_samples 6500 --num_generations 1 --gen_length 256 --temperature 0.9 \
--diffusion_steps 256 --decoding_strategy low_confidence



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
from tqdm import tqdm


import numpy as np
def load_done_indices(jsonl_path: Path) -> set[int]:
    done: set[int] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "idx" in obj:
                    done.add(int(obj["idx"]))
            except Exception:
                # ignore malformed/partial lines
                continue
    return done

def _json_default(o):
    # numpy scalar -> python scalar
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    # numpy array -> list
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # pandas NA
    try:
        import pandas as pd
        if o is pd.NA:
            return None
    except Exception:
        pass
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


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

    # gsm8k, sudoku, math:有 question / problem
    if "question" not in df.columns and "problem" in df.columns:
        df = df.rename(columns={"problem": "question"})

    if "question" in df.columns:
        df_sample = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)
        return df_sample

    # countdown
    if "nums" in df.columns and "target" in df.columns:
        df = df.copy()
        df["question"] = df.apply(
            lambda r: f"Numbers: {list(r['nums'])}\nTarget: {int(r['target'])}\n"
                      f"Find an equation using each number exactly once to reach the target.",
            axis=1
        )
        df["ground_truth"] = df.apply(lambda r: [list(r["nums"]), int(r["target"])], axis=1)

        df_sample = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)
        return df_sample

    raise RuntimeError(
        f"Unrecognized dataset format in {parquet_path}; columns: {list(df.columns)}"
    )

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()

def build_prompt(nums, target) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Using only the numbers {nums}, create an arithmetic expression that evaluates to exactly {target}. "
        "You must use all numbers from the list, and each number must be used exactly once. "
        "You may use the operations +, -, *, and / as needed. "
        "After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. "
        "For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: "
        "<answer>\n"
        "2*4-3\n"
        "</answer>"
    )

import re
from typing import List, Optional, Tuple, Any

# ---- minimal fallback for eval-style boxed extraction ----
def last_boxed_only_string(s: str) -> str:
    # return the LAST "\boxed{...}" (including "\boxed{...}")
    if not s:
        return ""
    starts = [m.start() for m in re.finditer(r"\\boxed\s*{", s)]
    if not starts:
        return ""
    start = starts[-1]
    brace = s.find("{", start)
    if brace == -1:
        return ""
    depth = 0
    for j in range(brace, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start:j+1]
    return s[start:]

def remove_boxed(s: str) -> str:
    m = re.search(r"\\boxed\s*{(.*)}\s*$", s, re.DOTALL)
    return m.group(1).strip() if m else s.strip()

def countdown_extract_equation(generated_text: str) -> str:
    if not generated_text:
        return ""

    equation = ""
    try:
        equation = remove_boxed(last_boxed_only_string(generated_text))
    except Exception:
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
        else:
            equation = generated_text

    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()

    return equation.strip()


def countdown_validate_equation(equation_str: str, available_numbers: List[int]) -> bool:
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False


def countdown_evaluate_equation(equation_str: str) -> float:
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str.strip(), {"__builtins__": None}, {})
    except Exception:
        return float("Inf")


def countdown_parse_ground_truth(question: str, ground_truth: Any) -> Tuple[List[int], Optional[int]]:
    numbers: List[int] = []
    target: Optional[int] = None

    if isinstance(ground_truth, list) and len(ground_truth) == 2:
        numbers = ground_truth[0]
        target = ground_truth[1]
    else:
        numbers_match = re.search(r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE)
        if numbers_match:
            numbers_str = numbers_match.group(1)
            numbers = [int(num.strip()) for num in numbers_str.split(",") if num.strip() != ""]

        target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
        if target_match:
            target = int(target_match.group(1))

    return numbers, target


def is_match_countdown(generated_text: Optional[str], ground_truth: Any, question: str) -> Tuple[bool, str, Optional[float]]:
    if generated_text is None:
        return (False, "", None)

    numbers, target = countdown_parse_ground_truth(question, ground_truth)

    equation = countdown_extract_equation(generated_text)

    is_valid = countdown_validate_equation(equation, numbers)
    result = None
    is_correct = False

    if is_valid:
        result = countdown_evaluate_equation(equation)
        if target is not None and abs(result - target) < 1e-5:
            is_correct = True

    return (is_correct, equation, result)

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

        # 修正映射：local_rank 循环分配到可见 GPU 上
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
    df = df[df["nums"].apply(len) == 3]
    print(f"Selected {len(df)} samples")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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
    mode = "a" if out_jsonl.exists() else "w"
    with open(out_jsonl, mode, encoding="utf-8") as fout:
        for i in tqdm(range(len(df)), disable=(local_rank != 0), desc="Processing samples"):
            # partition samples across ranks deterministically
            if use_ddp and (i % world_size) != local_rank:
                continue
            row = df.iloc[i]
            q = str(row["question"]) if not pd.isna(row["question"]) else ""
            # if dataset has 'answer' or 'target' column, pick one
            target = None
            # common column names for the reference answer in different math datasets
            if "answer" in row.index:
                target = row["answer"] if not pd.isna(row["answer"]) else None
            elif "target" in row.index:
                target = str(row["target"]) if not pd.isna(row["target"]) else None
            elif "solution" in row.index:
                # math8k / similar datasets often store the answer in 'solution'
                target = str(row["solution"]) if not pd.isna(row["solution"]) else None

            numbers, target = row["ground_truth"]
            prompt = build_prompt(numbers, target)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

            # move inputs to the correct device
            for k, v in inputs.items():
                inputs[k] = v.to(device)

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
                    # print(f"Warning: model.generate rejected custom kwargs ({e}), retrying without them")
                    outputs = model.generate(**inputs, **gen_kwargs)

            generations = []
            generations_full = []

            for sid in range(outputs.size(0)):
                out_ids = outputs[sid]
                full_text = tokenizer.decode(out_ids.tolist(), skip_special_tokens=False).strip()
                full_text = re.sub(r"(?:<\|endoftext\|>\s*)+$", "", full_text).strip()
                full_text = full_text.replace("<|eot_id|>", "").strip()
                generations_full.append(full_text)

                # 用于评测：尽量去掉 prompt（否则 extract_equation 会被 prompt 干扰）
                if prompt in full_text:
                    eval_text = full_text.split(prompt, 1)[1].strip()
                else:
                    eval_text = full_text
                generations.append(eval_text)



            # ground_truth: ideally from dataset column, e.g. row["ground_truth"]
            ground_truth = row["ground_truth"] if "ground_truth" in row.index else None

            matches = []
            extracted_equations = []
            eval_results = []

            total += 1

            rec = {
                "idx": int(i),
                "question": q,
                "target": target,
                "numbers": numbers,
                "generations": generations,
            }

            fout.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


    # If running with DDP, merge per-rank outputs into a single file on rank 0
    if use_ddp:
        dist.barrier()
        if local_rank == 0:
            final_jsonl = out_dir / "llada_countdown_generations.jsonl"
            final_summary = out_dir / "llada_countdown_generations_summary.json"
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
            # for r in range(world_size):
            #     part_sum = out_dir / f"llada_countdown_generations_summary_rank{r}.json"
            #     if not part_sum.exists():
            #         continue
            #     with open(part_sum, "r", encoding="utf-8") as frs:
            #         ps = json.load(frs)
            #         total_samples += ps.get("num_samples", 0)
            #         total_correct += ps.get("best_of_N_correct", 0)
            # summary_agg = {"num_samples": total_samples, "best_of_N_correct": total_correct, "best_of_N_accuracy": total_correct / max(1, total_samples)}
            # with open(final_summary, "w", encoding="utf-8") as ff:
            #     json.dump(summary_agg, ff, indent=2)
            print(f"Merged {world_size} rank files into {final_jsonl} and wrote aggregate summary to {final_summary}")
        dist.barrier()


if __name__ == "__main__":
    print(">>> generate_llada_countdown.py entered main()", flush=True)
    main()
