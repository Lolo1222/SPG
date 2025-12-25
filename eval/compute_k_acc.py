#!/usr/bin/env python3
import json
import os
import sys
import re
from pathlib import Path

# Import helper from repo
import importlib.util
from pathlib import Path

# load parser_helper by path to avoid importing package "eval" which conflicts with eval/eval.py
ph_path = Path(__file__).resolve().parent / "parser_helper.py"
spec = importlib.util.spec_from_file_location("parser_helper", str(ph_path))
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)

INPUT_DEFAULT = "save_dir/generate_results_math_train/llada_math_generations.jsonl"
OUT_CSV = "save_dir/generate_results_math_train/llada_k_acc.csv"
OUT_JSON = "save_dir/generate_results_math_train/llada_k_acc.json"

KS = [1,2,4,8,16]


def extract_answer(raw):
    parsed = None
    try:
        parsed = ph.remove_boxed(ph.last_boxed_only_string(raw))
    except Exception:
        parsed = None
    if not parsed:
        m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if m:
            parsed = m.group(1).strip()
    if parsed is None:
        # fallback: try to take last boxed content via regex
        boxed = re.findall(r"\\boxed{(.*?)}", raw)
        if boxed:
            parsed = boxed[-1].strip()
    return parsed


def main(path):
    path = Path(path)
    if not path.exists():
        print(f"Input file not found: {path}")
        sys.exit(1)

    results = []
    total_questions = 0
    acc_counts = {k:0 for k in KS}

    with path.open('r') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                print('json load error:', e)
                continue
            total_questions += 1
            idx = item.get('idx', total_questions)
            question = item.get('question', '')
            # ground truth might be under 'target' or 'ground_truth'
            ground = item.get('target') if item.get('target') is not None else item.get('ground_truth')
            # normalize ground to string
            ground_s = ground
            if isinstance(ground, (int, float)):
                ground_s = str(ground)
            # generations is expected to be list
            gens = item.get('generations', [])
            per_gen_correct = []
            for raw in gens:
                parsed = extract_answer(raw)
                correct = ph.is_equiv(parsed, ground_s)
                per_gen_correct.append(bool(correct))
            # pad to 16
            while len(per_gen_correct) < 16:
                per_gen_correct.append(False)
            per_question = {f'k{k}': any(per_gen_correct[:k]) for k in KS}
            for k in KS:
                if per_question[f'k{k}']:
                    acc_counts[k] += 1
            results.append({
                'idx': idx,
                'question': question if len(question)<=200 else question[:200],
                'ground_truth': ground_s,
                'per_gen_correct': per_gen_correct,
                **per_question
            })

    # write csv
    import csv
    outdir = Path(OUT_CSV).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w') as csvf:
        writer = csv.writer(csvf)
        header = ['idx','ground_truth'] + [f'gen{i+1}_correct' for i in range(16)] + [f'k{k}' for k in KS]
        writer.writerow(header)
        for r in results:
            row = [r['idx'], r['ground_truth']] + [int(x) for x in r['per_gen_correct']] + [int(r[f'k{k}']) for k in KS]
            writer.writerow(row)

    # write json
    with open(OUT_JSON,'w') as jf:
        json.dump({'total_questions': total_questions, 'ks':KS, 'acc_counts':acc_counts, 'results':results}, jf, indent=2)

    print('Done')
    print('Total questions:', total_questions)
    for k in KS:
        pct = acc_counts[k]/total_questions*100 if total_questions>0 else 0.0
        print(f'top-{k} accuracy: {acc_counts[k]}/{total_questions} = {pct:.2f}%')


if __name__=='__main__':
    path = sys.argv[1] if len(sys.argv)>1 else INPUT_DEFAULT
    main(path)
