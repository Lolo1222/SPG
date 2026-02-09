import json
from pathlib import Path

src = Path("/root/jiawei/SPG/dataset/llada_gsm8k_generations_7473.jsonl")
dst = Path("dataset/llada_gsm8k_generations_7473_converted.jsonl")
    
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)

        # Build the new record
        new_obj = {
            "problem": obj.get("question", ""),
            "solution": obj.get("target", ""),
            "generation": (obj.get("generations") or [""])[0],
        }

        json.dump(new_obj, fout, ensure_ascii=False)
        fout.write("\n")

print(f"Converted to {dst}")