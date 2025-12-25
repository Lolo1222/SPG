#!/usr/bin/env python3
import json
from pathlib import Path

INPUT = Path("save_dir/generate_results_math_train/llada_math_generations.jsonl")
KS = [1,2,4,8,16]

if not INPUT.exists():
    print("Input not found:", INPUT)
    raise SystemExit(1)

total = 0
acc = {k:0 for k in KS}

with INPUT.open() as f:
    for line in f:
        line=line.strip()
        if not line: continue
        rec = json.loads(line)
        total += 1
        matches = rec.get('matches')
        # if matches field absent, try 'per_gen_correct' or 'extracted_answers'
        if matches is None:
            print('No matches in record, skipping')
            continue
        # ensure length at least 16
        while len(matches) < 16:
            matches.append(False)
        for k in KS:
            if any(matches[:k]):
                acc[k] += 1

print('total', total)
for k in KS:
    print(f'top-{k}: {acc[k]}/{total} = {acc[k]/total*100:.2f}%')
