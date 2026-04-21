#!/usr/bin/env python3
"""Extract records from a combined JSONL by source dataset (MATH/GSM8K train).

This script:
1) Loads the combined JSONL file.
2) Loads MATH(train) and GSM8K(train) from Hugging Face datasets.
3) Matches combined records to either source by normalized problem text.
4) Writes two JSONL outputs with only: problem, shortcot, solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Set, Tuple

from datasets import load_dataset


def normalize_text(text: str) -> str:
    """Normalize text for robust string matching."""
    return " ".join(text.strip().split())


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_problem_sets() -> Tuple[Set[str], Set[str]]:
    """Load normalized problem texts for MATH(train) and GSM8K(train)."""
    math_ds = load_dataset("ankner/math-500", split="train")
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")

    math_problems = {normalize_text(x["problem"]) for x in math_ds if x.get("problem")}
    gsm8k_questions = {
        normalize_text(x["question"]) for x in gsm8k_ds if x.get("question")
    }
    return math_problems, gsm8k_questions


def extract_rows(
    combined_path: Path,
    math_problem_set: Set[str],
    gsm8k_question_set: Set[str],
) -> Tuple[list[Dict], list[Dict], int]:
    math_rows: list[Dict] = []
    gsm8k_rows: list[Dict] = []
    unknown_count = 0

    for item in iter_jsonl(combined_path):
        problem = item.get("problem")
        if not isinstance(problem, str):
            unknown_count += 1
            continue

        key = normalize_text(problem)
        row = {
            "problem": item.get("problem", ""),
            "shortcot": item.get("shortcot", ""),
            "solution": item.get("solution", ""),
        }

        in_math = key in math_problem_set
        in_gsm8k = key in gsm8k_question_set

        # If a record appears in both sets (rare), keep it in both outputs.
        if in_math:
            math_rows.append(row)
        if in_gsm8k:
            gsm8k_rows.append(row)
        if not in_math and not in_gsm8k:
            unknown_count += 1

    return math_rows, gsm8k_rows, unknown_count


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MATH(train) and GSM8K(train) records from a combined JSONL by "
            "matching problem text."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/jwliu/dlm/SPG/rebuttal/dataset/combined_ab_3.jsonl"),
        help="Path to combined JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/jwliu/dlm/SPG/rebuttal/dataset"),
        help="Directory to store extracted JSONL files.",
    )
    parser.add_argument(
        "--math_output",
        type=str,
        default="combined_ab_3_math_train.jsonl",
        help="Output filename for MATH(train)-matched records.",
    )
    parser.add_argument(
        "--gsm8k_output",
        type=str,
        default="combined_ab_3_gsm8k_train.jsonl",
        help="Output filename for GSM8K(train)-matched records.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    math_output_path = args.output_dir / args.math_output
    gsm8k_output_path = args.output_dir / args.gsm8k_output

    print("Loading MATH(train) and GSM8K(train) ...")
    math_set, gsm8k_set = load_problem_sets()
    print(f"Loaded MATH(train) problems: {len(math_set)}")
    print(f"Loaded GSM8K(train) questions: {len(gsm8k_set)}")

    print(f"Reading combined file: {args.input}")
    math_rows, gsm8k_rows, unknown_count = extract_rows(
        args.input, math_set, gsm8k_set
    )

    math_count = write_jsonl(math_output_path, math_rows)
    gsm8k_count = write_jsonl(gsm8k_output_path, gsm8k_rows)

    print(f"Wrote MATH records: {math_count} -> {math_output_path}")
    print(f"Wrote GSM8K records: {gsm8k_count} -> {gsm8k_output_path}")
    print(f"Unmatched records in combined file: {unknown_count}")


if __name__ == "__main__":
    main()
