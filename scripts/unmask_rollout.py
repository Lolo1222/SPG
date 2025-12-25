#!/usr/bin/env python3
"""
Sample one generation from a jsonl dataset, mask tokens according to strategy,
perform iterative unmasking (steps = number of masked tokens) for `rollout_nums` runs,
and evaluate correctness using existing math parsing code.

Usage example:
  python3 scripts/unmask_rollout.py \
    --jsonl deal_data/math01/llada_math_generations.jsonl \
    --model_path /path/to/model \
    --p_gen_mask 0.3 --mask_strategy lowprob --rollout_nums 5
"""
import argparse
import json
import os
import random
import time
import sys
from typing import List

# Allow specifying GPUs early via `--gpus` so we can set CUDA_VISIBLE_DEVICES
# before CUDA / torch initializes. Example: `--gpus 0,1`
def _maybe_set_cuda_visible_devices():
    gpus = None
    # Build a new argv without the --gpus entries so argparse won't see them
    new_argv = []
    i = 0
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--gpus":
            # consume this and the following value (if any)
            if i + 1 < len(sys.argv):
                gpus = sys.argv[i + 1]
            i += 2
            continue
        if a.startswith("--gpus="):
            gpus = a.split("=", 1)[1]
            i += 1
            continue
        new_argv.append(a)
        i += 1

    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"[unmask_rollout] set CUDA_VISIBLE_DEVICES={gpus}")

    # replace sys.argv so argparse doesn't see the --gpus flag
    sys.argv[:] = new_argv


_maybe_set_cuda_visible_devices()

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Ensure repository root is on sys.path so `eval` package can be imported
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print(f"[unmask_rollout] inserted repo root to sys.path: {repo_root}")

from eval.parse_and_get_acc import parse_math_answers


def sample_one_record(jsonl_path: str):
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    if total == 0:
        raise RuntimeError(f"No examples in {jsonl_path}")
    idx = random.randrange(total)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise RuntimeError("unexpected")


def choose_mask_positions(ids: List[int], mask_count: int, special_ids: set, strategy: str, model=None, device=None):
    # ids: list of token ids (no batch dim)
    candidate_positions = [i for i in range(len(ids)) if ids[i] not in special_ids]
    if mask_count <= 0:
        return []
    if mask_count >= len(candidate_positions):
        return candidate_positions

    if strategy == "random":
        return sorted(random.sample(candidate_positions, mask_count))

    if strategy == "tail":
        rev_candidates = [p for p in reversed(candidate_positions)]
        chosen = rev_candidates[:mask_count]
        return sorted(chosen)

    if strategy == "lowprob":
        # need model and device to compute token probs for current tokens
        assert model is not None and device is not None
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]  # (S, V)
            probs = F.softmax(logits, dim=-1)
            token_probs = probs[range(len(ids)), ids]
            pairs = [(float(token_probs[p].item()), p) for p in candidate_positions]
            pairs.sort(key=lambda x: x[0])
            chosen = [p for (_, p) in pairs[:mask_count]]
            return sorted(chosen)

    raise ValueError(f"unknown mask strategy {strategy}")


def iterative_unmask(model, input_ids: torch.Tensor, mask_id: int, steps: int, temperature: float, device):
    # input_ids: (1, S) mutable tensor containing mask_id tokens to be filled
    ids = input_ids.clone()
    for step in range(steps):
        with torch.no_grad():
            outputs = model(input_ids=ids)
            logits = outputs.logits  # (1, S, V)
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)

        is_mask = (ids == mask_id)[0]
        mask_positions = is_mask.nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            break

        pos_list = [int(p.item()) for p in mask_positions.squeeze(-1)]
        top_probs = []
        top_ids = []
        for p in pos_list:
            p_probs = probs[0, p]
            val, idx = torch.max(p_probs, dim=-1)
            top_probs.append(float(val.item()))
            top_ids.append(int(idx.item()))

        # select the lowest-confidence position to fill now
        sel_idx = int(min(range(len(pos_list)), key=lambda i: top_probs[i]))
        sel_pos = pos_list[sel_idx]
        sel_token = top_ids[sel_idx]

        ids[0, sel_pos] = sel_token

    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--p_gen_mask", type=float, default=0.3)
    parser.add_argument("--mask_strategy", type=str, choices=["random", "tail", "lowprob"], default="random")
    parser.add_argument("--rollout_nums", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # choose dtype safely
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model from {args.model_path} to {device} (dtype={torch_dtype})")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch_dtype).to(device)
        except Exception:
            print("Warning: PEFT checkpoint requested but failed to load. Proceeding without.")

    rec = sample_one_record(args.jsonl)
    gens = rec.get("generations") or rec.get("generation") or []
    if isinstance(gens, list) and len(gens) > 0:
        gen_text = random.choice(gens)
    elif isinstance(gens, str):
        gen_text = gens
    else:
        gen_text = rec.get("target", "")

    question = rec.get("question")
    target = rec.get("target")
    extracted_answers = rec.get("extracted_answers", [])
    matches = rec.get("matches", [])

    print("Sampled question:\n", question)
    print("Original generation (truncated):\n", gen_text[:1000])

    enc = tokenizer(gen_text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.eos_token_id

    mask_count = max(1, int(round(args.p_gen_mask * len(ids))))
    print(f"Sequence length {len(ids)}, will mask {mask_count} tokens (p_gen_mask={args.p_gen_mask})")

    mask_positions = choose_mask_positions(ids, mask_count, special_ids, args.mask_strategy,
                                           model=model if args.mask_strategy == "lowprob" else None,
                                           device=device)
    print(f"Mask positions count: {len(mask_positions)}; positions example (first 20): {mask_positions[:20]}")

    masked_ids = ids.copy()
    for p in mask_positions:
        masked_ids[p] = mask_token_id

    rollouts = []
    for r in range(args.rollout_nums):
        input_tensor = torch.tensor([masked_ids], dtype=torch.long, device=device)
        decoded = iterative_unmask(model, input_tensor, mask_token_id, steps=len(mask_positions),
                                   temperature=args.temperature, device=device)
        decoded_text = tokenizer.decode(decoded[0].tolist(), skip_special_tokens=False)

        data = {"generations": [{"question": question, "ground_truth": target, "generations": decoded_text}]}
        total_correct, total_processed, processed_items, _ = parse_math_answers(json_data=data)
        is_correct = processed_items[0]["is_correct"] if processed_items else False
        extracted = processed_items[0].get("extracted_answer") if processed_items else None

        rollouts.append({
            "rollout_index": r,
            "decoded_text": decoded_text,
            "is_correct": is_correct,
            "extracted_answer": extracted,
        })
        print(f"Rollout {r}: is_correct={is_correct}, extracted={extracted}")

    out = {
        "sample_question": question,
        "target": target,
        "extracted_answers": extracted_answers,
        "matches": matches,
        "mask_strategy": args.mask_strategy,
        "p_gen_mask": args.p_gen_mask,
        "mask_positions": mask_positions,
        "rollouts": rollouts,
        "model_path": args.model_path,
        "timestamp": time.time(),
    }
    out_path = f"results/unmask_rollout_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()