#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
import importlib.util

# load parser_helper from repo to reuse last_boxed_only_string, remove_boxed, is_equiv
ph_path = Path(__file__).resolve().parent / "parser_helper.py"
spec = importlib.util.spec_from_file_location("parser_helper", str(ph_path))
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)

INPUT = Path("save_dir/generate_results_math_train/llada_math_generations.jsonl")
OUT_JSON = Path("save_dir/generate_results_math_train/llada_k_acc_parse_math.json")
KS = [1,2,4,8,16]

if not INPUT.exists():
    print("Input not found:", INPUT)
    sys.exit(1)

results = []
acc_counts = {k:0 for k in KS}
total = 0

with INPUT.open() as f:
    for line in f:
        line=line.strip()
        if not line: continue
        rec = json.loads(line)
        total += 1
        idx = rec.get('idx', total)
        question = rec.get('question','')
        ground = rec.get('target') if rec.get('target') is not None else rec.get('ground_truth')
        # convert ground to string if it's numeric
        ground_s = ground
        if isinstance(ground, (int,float)):
            ground_s = str(ground)
        gens = rec.get('generations', [])
        per_gen_correct = []
        per_gen_extracted = []
        for raw in gens:
            parsed = None
            try:
                parsed = ph.remove_boxed(ph.last_boxed_only_string(raw))
            except Exception:
                parsed = None
            if not parsed:
                m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
                if m:
                    parsed = m.group(1).strip()
            # use is_equiv for comparison (same as parse_math_answers)
            correct = False
            if parsed is not None:
                try:
                    correct = ph.is_equiv(parsed, ground_s)
                except Exception:
                    correct = False
            per_gen_correct.append(bool(correct))
            per_gen_extracted.append(parsed)
        # pad to 16
        while len(per_gen_correct) < 16:
            print("Padding generations for idx", idx)
            per_gen_correct.append(False)
            per_gen_extracted.append(None)
        per_question = {f'k{k}': any(per_gen_correct[:k]) for k in KS}
        for k in KS:
            if per_question[f'k{k}']:
                acc_counts[k] += 1
        results.append({
            'idx': idx,
            'question': question if len(question)<=200 else question[:200],
            'ground_truth': ground_s,
            'per_gen_extracted': per_gen_extracted,
            'per_gen_correct': per_gen_correct,
            **per_question
        })

# write out
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON,'w') as jf:
    json.dump({'total_questions': total, 'ks':KS, 'acc_counts':acc_counts, 'results':results}, jf, indent=2)

print('Total', total)
for k in KS:
    print(f'top-{k}: {acc_counts[k]}/{total} = {acc_counts[k]/total*100:.2f}%')
print('Wrote detailed results to', OUT_JSON)
