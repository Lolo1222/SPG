#!/usr/bin/env python3
"""Extract MATH/GSM8K train records from a local file with `query` field.

Input record format (example):
{
  "query": "<problem_text>\nPlease reason step by step, and put your final answer within \\boxed{}.",
  "answer": "..."
}

Output files (JSONL):
- *_math_train.jsonl
- *_gsm8k_train.jsonl

Each output record only contains:
- problem
- answer          (from input file)
- solution        (from source dataset itself: MATH.solution / GSM8K.answer)
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset


QUERY_SUFFIX_TEMPLATES = [
    "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "\nPlease reason step by step, and put your final answer within \\\\boxed{}.",
    "Please reason step by step, and put your final answer within \\boxed{}.",
    "Please reason step by step, and put your final answer within \\\\boxed{}.",
]


def normalize_text(text: str) -> str:
    """Normalize text for matching: trim + collapse whitespace."""
    return " ".join(text.strip().split())


def remove_leading_index(text: str) -> str:
    """Remove leading numbering like '14. ' from problems."""
    return re.sub(r"^\s*\d+\.\s*", "", text)


def remove_problem_prefix(text: str) -> str:
    """Remove common leading prefixes like '14.', 'Problem 10.1.', '(3)' etc."""
    s = text.strip()
    # e.g. "Problem 10.1. ...", "problem 3) ..."
    s = re.sub(r"^\s*problem\s*\d+(?:[\.-]\d+)*[\.)]?\s*", "", s, flags=re.IGNORECASE)
    # e.g. "14. ...", "(3) ...", "3) ...", "10.1 ..."
    s = re.sub(r"^\s*\(?\d+(?:[\.-]\d+)*\)?[\.)]?\s*", "", s)
    return s.strip()


def compact_key(text: str) -> str:
    """Aggressive normalization for fuzzy fallback matching.

    Keep only unicode letters/digits after NFKC + lowercase.
    """
    s = unicodedata.normalize("NFKC", text).lower()
    s = remove_problem_prefix(s)
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def strip_query_suffix(query: str) -> str:
    """Strip known prompt template suffix from query to recover the problem text."""
    result = query
    for t in QUERY_SUFFIX_TEMPLATES:
        if result.endswith(t):
            result = result[: -len(t)]
            break
    return result.strip()


def parse_problem_from_query(query: str) -> str:
    """Recover problem from query robustly."""
    s = strip_query_suffix(query)
    s = remove_leading_index(s)
    s = remove_problem_prefix(s)
    return s.strip()


def read_input_records(path: Path) -> List[Dict]:
    """Read records from .json/.jsonl.

    Supported cases:
    - JSONL: one JSON object per line.
    - JSON: either a dict or a list of dicts.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try parse as a full JSON value first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback: treat as JSONL.
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(rec, dict):
                records.append(rec)
    return records


def build_math_index() -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    """Build exact and compact indices for MATH(train)."""
    ds = load_dataset("ankner/math-500", split="train")
    exact_index: Dict[str, Dict] = {}
    compact_index: Dict[str, List[Dict]] = {}

    for rec in ds:
        prob = rec.get("problem")
        if not isinstance(prob, str):
            continue
        raw = prob.strip()
        keys = [
            normalize_text(raw),
            normalize_text(remove_leading_index(raw)),
            normalize_text(remove_problem_prefix(raw)),
        ]
        for k in keys:
            if k and k not in exact_index:
                exact_index[k] = rec

        ck = compact_key(raw)
        if ck:
            compact_index.setdefault(ck, []).append(rec)
    return exact_index, compact_index


def build_gsm8k_index() -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    """Build exact and compact indices for GSM8K(train)."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    exact_index: Dict[str, Dict] = {}
    compact_index: Dict[str, List[Dict]] = {}

    for rec in ds:
        q = rec.get("question")
        if not isinstance(q, str):
            continue
        raw = q.strip()
        keys = [
            normalize_text(raw),
            normalize_text(remove_leading_index(raw)),
            normalize_text(remove_problem_prefix(raw)),
        ]
        for k in keys:
            if k and k not in exact_index:
                exact_index[k] = rec

        ck = compact_key(raw)
        if ck:
            compact_index.setdefault(ck, []).append(rec)
    return exact_index, compact_index


def get_query_keys(query: str) -> List[str]:
    """Generate multiple exact-match keys from query."""
    p = parse_problem_from_query(query)
    keys = [
        normalize_text(p),
        normalize_text(remove_leading_index(p)),
        normalize_text(remove_problem_prefix(p)),
    ]
    # Deduplicate while preserving order
    uniq: List[str] = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def unique_match_by_compact_key(
    compact_idx: Dict[str, List[Dict]],
    key: str,
) -> Optional[Dict]:
    """Return record only when compact key maps to exactly one candidate."""
    candidates = compact_idx.get(key, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract MATH/GSM8K train data from query-based local file"
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to local input file (.json or .jsonl) containing query/answer fields.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/jwliu/dlm/SPG/rebuttal/dataset"),
        help="Directory to save output JSONL files.",
    )
    p.add_argument(
        "--math_output",
        type=str,
        default="extracted_math_train_from_query.jsonl",
        help="Output filename for MATH-matched samples.",
    )
    p.add_argument(
        "--gsm8k_output",
        type=str,
        default="extracted_gsm8k_train_from_query.jsonl",
        help="Output filename for GSM8K-matched samples.",
    )
    p.add_argument(
        "--debug_unmatched_sample",
        type=int,
        default=20,
        help="How many unmatched query heads to print for debugging.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    math_out = args.output_dir / args.math_output
    gsm_out = args.output_dir / args.gsm8k_output

    print(f"Reading local file: {args.input}")
    records = read_input_records(args.input)
    print(f"Loaded local records: {len(records)}")

    print("Loading MATH(train) and building index...")
    math_exact_index, math_compact_index = build_math_index()
    print(
        f"MATH exact keys: {len(math_exact_index)}, compact keys: {len(math_compact_index)}"
    )

    print("Loading GSM8K(train) and building index...")
    gsm_exact_index, gsm_compact_index = build_gsm8k_index()
    print(
        f"GSM8K exact keys: {len(gsm_exact_index)}, compact keys: {len(gsm_compact_index)}"
    )

    math_rows: List[Dict] = []
    gsm_rows: List[Dict] = []
    unmatched = 0
    missing_query = 0
    math_exact_hit = 0
    math_compact_hit = 0
    gsm_exact_hit = 0
    gsm_compact_hit = 0
    unmatched_heads: List[str] = []

    for rec in records:
        query = rec.get("query")
        if not isinstance(query, str):
            missing_query += 1
            unmatched += 1
            continue

        answer = rec.get("answer", "")
        query_keys = get_query_keys(query)

        matched_math: Optional[Dict] = None
        matched_gsm: Optional[Dict] = None

        # 1) exact matching
        for k in query_keys:
            if matched_math is None and k in math_exact_index:
                matched_math = math_exact_index[k]
                math_exact_hit += 1
            if matched_gsm is None and k in gsm_exact_index:
                matched_gsm = gsm_exact_index[k]
                gsm_exact_hit += 1
            if matched_math is not None and matched_gsm is not None:
                break

        # 2) compact fallback, only accept unique matches to avoid noisy collisions
        q_compact = compact_key(parse_problem_from_query(query))
        if matched_math is None and q_compact:
            m = unique_match_by_compact_key(math_compact_index, q_compact)
            if m is not None:
                matched_math = m
                math_compact_hit += 1
        if matched_gsm is None and q_compact:
            g = unique_match_by_compact_key(gsm_compact_index, q_compact)
            if g is not None:
                matched_gsm = g
                gsm_compact_hit += 1

        if matched_math is not None:
            math_rows.append(
                {
                    "problem": matched_math.get("problem", parse_problem_from_query(query)),
                    "answer": answer,
                    "solution": matched_math.get("solution", ""),
                }
            )
        if matched_gsm is not None:
            gsm_rows.append(
                {
                    "problem": matched_gsm.get("question", parse_problem_from_query(query)),
                    "answer": answer,
                    # GSM8K's source answer field is "answer"
                    "solution": matched_gsm.get("answer", ""),
                }
            )

        if matched_math is None and matched_gsm is None:
            unmatched += 1
            if len(unmatched_heads) < args.debug_unmatched_sample:
                head = parse_problem_from_query(query).split("\n")[0].strip()
                unmatched_heads.append(head[:220])

    math_count = write_jsonl(math_out, math_rows)
    gsm_count = write_jsonl(gsm_out, gsm_rows)

    print(f"Wrote MATH records: {math_count} -> {math_out}")
    print(f"Wrote GSM8K records: {gsm_count} -> {gsm_out}")
    print(f"Unmatched records: {unmatched}")
    print(f"Records missing query field: {missing_query}")
    print(
        "Match breakdown: "
        f"MATH exact={math_exact_hit}, MATH compact={math_compact_hit}, "
        f"GSM8K exact={gsm_exact_hit}, GSM8K compact={gsm_compact_hit}"
    )

    if unmatched_heads:
        print("Sample unmatched query heads:")
        for i, h in enumerate(unmatched_heads, start=1):
            print(f"{i}. {h}")


if __name__ == "__main__":
    main()
