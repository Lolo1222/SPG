#!/usr/bin/env python3
import json
from pathlib import Path

orig_path = Path('save_dir/generate_results_math_train/llada_math_generations.jsonl')
parse_path = Path('save_dir/generate_results_math_train/llada_k_acc_parse_math.json')
out_json = Path('save_dir/generate_results_math_train/match_vs_parse_diffs.json')
out_csv = Path('save_dir/generate_results_math_train/match_vs_parse_diffs.csv')

if not orig_path.exists():
    print('Original generations file not found:', orig_path)
    raise SystemExit(1)
if not parse_path.exists():
    print('Parsed results file not found:', parse_path)
    raise SystemExit(1)

# load parse results (map idx -> per_gen_correct list)
with open(parse_path, 'r') as f:
    parse_data = json.load(f)
parse_map = {int(r['idx']): r for r in parse_data['results']}

mismatches = []
summary = {'total_samples':0, 'samples_with_any_mismatch':0, 'total_gen':0, 'mismatched_gen':0}

with open(orig_path, 'r') as fo:
    for line in fo:
        rec = json.loads(line)
        idx = int(rec.get('idx', -1))
        summary['total_samples'] += 1
        gens = rec.get('generations', [])
        matches = rec.get('matches')
        if matches is None:
            # fallback: nothing to compare for this sample
            continue
        # ensure length 16
        while len(matches) < 16:
            matches.append(False)
        # get parse result
        pr = parse_map.get(idx)
        if pr is None:
            # try match by question text fallback
            continue
        parse_correct = pr.get('per_gen_correct', [])
        while len(parse_correct) < 16:
            parse_correct.append(False)
        sample_mismatch = False
        # get ground truth from original record if present
        orig_ground = rec.get('target') if rec.get('target') is not None else rec.get('ground_truth')
        for i in range(16):
            summary['total_gen'] += 1
            orig = bool(matches[i])
            parsed = bool(parse_correct[i])
            if orig != parsed:
                sample_mismatch = True
                summary['mismatched_gen'] += 1
                # get original extracted answer if present in the generation record
                orig_extracted = None
                if isinstance(rec.get('extracted_answers'), list) and i < len(rec.get('extracted_answers')):
                    orig_extracted = rec.get('extracted_answers')[i]

                mismatches.append({
                    'idx': idx,
                    'gen_index': i,
                    'orig_match': orig,
                    'parse_match': parsed,
                    'generation_text': gens[i] if i < len(gens) else None,
                    'parse_extracted': pr['per_gen_extracted'][i] if i < len(pr['per_gen_extracted']) else None,
                    'orig_extracted_matches_field': orig_extracted,
                    'ground_truth_orig': orig_ground,
                    'ground_truth_parse': pr.get('ground_truth') if 'ground_truth' in pr else pr.get('ground_truth')
                })
        if sample_mismatch:
            summary['samples_with_any_mismatch'] += 1

# write json and csv
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, 'w') as jf:
    json.dump({'summary': summary, 'mismatches': mismatches}, jf, indent=2, ensure_ascii=False)

with open(out_csv, 'w', encoding='utf-8') as cf:
    cf.write('idx,gen_index,orig_match,parse_match,parse_extracted,generation_truncated\n')
    for m in mismatches:
        gen_text = (m['generation_text'] or '').replace('\n',' ').replace('\r',' ')
        gen_text = gen_text[:200].replace('"','""')
        pe = (m['parse_extracted'] or '')
        cf.write(f"{m['idx']},{m['gen_index']},{int(m['orig_match'])},{int(m['parse_match'])},\"{pe}\",\"{gen_text}\"\n")

print('Wrote', out_json, 'and', out_csv)
print('Summary:', summary)
