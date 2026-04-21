import argparse
import glob
import json
import os
import re
from collections import defaultdict

from datasets import load_dataset

from parse_and_get_acc import parse_math_answers


def extract_setup_name(filename):
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    return match.group(1) if match else None


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split())


def load_generations_grouped_by_setup(results_dir):
    json_files = sorted(glob.glob(os.path.join(results_dir, "*_generations.json")))
    if not json_files:
        raise FileNotFoundError(f"No '*_generations.json' files found in: {results_dir}")

    grouped = defaultdict(list)
    for path in json_files:
        setup = extract_setup_name(os.path.basename(path))
        if setup is None:
            continue
        with open(path, "r") as f:
            data = json.load(f)
        grouped[setup].extend(data.get("generations", []))

    if not grouped:
        raise ValueError(
            "No valid setup names found from result files. Expected names like '*_0_generations.json'."
        )
    return grouped


def build_dataset_index_maps(dataset_name, dataset_split):
    dataset = load_dataset(dataset_name, split=dataset_split)
    qa_to_indices = defaultdict(list)
    q_to_indices = defaultdict(list)

    for idx, item in enumerate(dataset):
        q = normalize_text(item["problem"])
        a = normalize_text(item["answer"])
        qa_to_indices[(q, a)].append(idx)
        q_to_indices[q].append(idx)

    return qa_to_indices, q_to_indices, len(dataset)


def assign_dataset_indices(processed_items, qa_to_indices, q_to_indices):
    assigned = []
    unmatched = []

    # Work on copies so each setup uses a fresh index pool.
    qa_pool = {k: v.copy() for k, v in qa_to_indices.items()}
    q_pool = {k: v.copy() for k, v in q_to_indices.items()}

    for item in processed_items:
        q = normalize_text(item.get("question", ""))
        a = normalize_text(item.get("ground_truth", ""))

        idx = None
        qa_key = (q, a)
        if qa_key in qa_pool and qa_pool[qa_key]:
            idx = qa_pool[qa_key].pop(0)
            if idx in q_pool.get(q, []):
                q_pool[q].remove(idx)
        elif q in q_pool and q_pool[q]:
            idx = q_pool[q].pop(0)

        if idx is None:
            unmatched.append(item)
            continue

        item_with_idx = dict(item)
        item_with_idx["dataset_index"] = idx
        assigned.append(item_with_idx)

    return assigned, unmatched


def evaluate_on_split(assigned_items, split_indices):
    split_set = set(split_indices)
    selected = [x for x in assigned_items if x["dataset_index"] in split_set]
    total = len(selected)
    correct = sum(1 for x in selected if x.get("is_correct", False))
    acc = (100.0 * correct / total) if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "acc": acc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute MATH-500 val/test accuracies from full-test generation outputs and a split file."
    )
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing *_generations.json files")
    parser.add_argument(
        "--split_file",
        type=str,
        default="dataset/math500_test_split_seed42.json",
        help="JSON file with 'val' and 'test' index lists",
    )
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()

    with open(args.split_file, "r") as f:
        split_data = json.load(f)

    if "val" not in split_data or "test" not in split_data:
        raise KeyError(f"Split file must contain 'val' and 'test' keys: {args.split_file}")

    val_indices = split_data["val"]
    test_indices = split_data["test"]

    grouped = load_generations_grouped_by_setup(args.results_dir)
    qa_map, q_map, dataset_size = build_dataset_index_maps(args.dataset_name, args.dataset_split)

    print(f"Loaded split file: {args.split_file}")
    print(f"Dataset: {args.dataset_name} ({args.dataset_split}), size={dataset_size}")
    print("=" * 100)

    header = "{:<45} {:>8} {:>8} {:>10} {:>8} {:>8} {:>10} {:>10}"
    row = "{:<45} {:>8} {:>8} {:>9.2f}% {:>8} {:>8} {:>9.2f}% {:>10}"
    print(
        header.format(
            "setup",
            "val_c",
            "val_n",
            "val_acc",
            "test_c",
            "test_n",
            "test_acc",
            "unmatched",
        )
    )
    print("-" * 100)

    for setup, generations in sorted(grouped.items()):
        _, _, processed_items, _ = parse_math_answers(json_data={"generations": generations})
        assigned, unmatched = assign_dataset_indices(processed_items, qa_map, q_map)

        val_stats = evaluate_on_split(assigned, val_indices)
        test_stats = evaluate_on_split(assigned, test_indices)

        print(
            row.format(
                setup,
                val_stats["correct"],
                val_stats["total"],
                val_stats["acc"],
                test_stats["correct"],
                test_stats["total"],
                test_stats["acc"],
                len(unmatched),
            )
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
