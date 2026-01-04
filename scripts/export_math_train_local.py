# python scripts/export_math_train_local.py --output /home/jwliu/dlm/SPG/dataset/math500_train.jsonl --hf-dataset ankner/math-500 --split train
import os
import sys
import json
import argparse
from datasets import load_dataset

# Ensure project root is on sys.path so `spg` package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the exact SYSTEM_PROMPT used by get_math_questions for consistency
# from spg.data_utils import SYSTEM_PROMPT
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def build_record(problem: str, solution: str):
    """Build a record matching get_math_questions output format."""
    return {
        "problem": problem,
        "solution": solution,
        "generation": solution,
    }


def export_math_train(output_path: str, hf_dataset_id: str = "ankner/math-500", split: str = "train"):
    ds = load_dataset(hf_dataset_id, split=split)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            record = build_record(problem, solution)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export math-500 train split locally in get_math_questions format")
    default_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "math_train_formatted.jsonl"))
    parser.add_argument("--output", type=str, default=default_out, help="Output JSONL path")
    parser.add_argument("--hf-dataset", type=str, default="ankner/math-500", help="HF dataset id")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to export")
    args = parser.parse_args()

    export_math_train(args.output, hf_dataset_id=args.hf_dataset, split=args.split)


if __name__ == "__main__":
    main()
