import json
from pathlib import Path

# src = Path("/root/jiawei/SPG/dataset/llada_countdown_generation_3.jsonl")
src = Path("dataset/sudoku/math_merged.jsonl")
# src = Path("dataset/countdown/llada_countdown_generations_3.jsonl")
dst = Path("dataset/sudoku/llada_math_generations_new_converted.jsonl")
# dst = Path("dataset/countdown/llada_countdown_generation_converted_3.jsonl")
    
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)

        # Build the new record
        new_obj = {
            "Puzzle": str(obj.get("Puzzle", "")),
            "Solution": str(obj.get("Solution", "")),
            "generation": (obj.get("generations") or [""])[0],
        }

        json.dump(new_obj, fout, ensure_ascii=False)
        fout.write("\n")

print(f"Converted to {dst}")