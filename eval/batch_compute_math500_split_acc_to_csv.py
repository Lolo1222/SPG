import argparse
import csv
import json
import os

from compute_math500_split_acc import (
    assign_dataset_indices,
    build_dataset_index_maps,
    evaluate_on_split,
    load_generations_grouped_by_setup,
)
from parse_and_get_acc import parse_math_answers


def resolve_model_dirs(results_root, model_dirs):
    resolved = []
    for entry in model_dirs:
        if os.path.isabs(entry):
            path = entry
        else:
            path = os.path.join(results_root, entry)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        resolved.append(os.path.abspath(path))
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Batch compute MATH-500 val/test split accuracy for multiple result directories and export CSV."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="save_dir/new_eval_results",
        help="Base directory for model result folders",
    )
    parser.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Model result directories (either folder names under results_root or absolute paths)",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="dataset/math500_test_split_seed42.json",
        help="JSON split file with 'val' and 'test' index arrays",
    )
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="save_dir/new_eval_results/math500_split_acc_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    with open(args.split_file, "r") as f:
        split_data = json.load(f)
    if "val" not in split_data or "test" not in split_data:
        raise KeyError(f"Split file must contain 'val' and 'test': {args.split_file}")

    val_indices = split_data["val"]
    test_indices = split_data["test"]

    qa_map, q_map, dataset_size = build_dataset_index_maps(args.dataset_name, args.dataset_split)
    model_paths = resolve_model_dirs(args.results_root, args.model_dirs)

    rows = []
    for model_path in model_paths:
        grouped = load_generations_grouped_by_setup(model_path)
        model_name = os.path.basename(model_path.rstrip("/"))

        for setup, generations in sorted(grouped.items()):
            _, _, processed_items, _ = parse_math_answers(json_data={"generations": generations})
            assigned, unmatched = assign_dataset_indices(processed_items, qa_map, q_map)

            val_stats = evaluate_on_split(assigned, val_indices)
            test_stats = evaluate_on_split(assigned, test_indices)

            rows.append(
                {
                    "model_dir": model_name,
                    "model_path": model_path,
                    "setup": setup,
                    "dataset_name": args.dataset_name,
                    "dataset_split": args.dataset_split,
                    "dataset_size": dataset_size,
                    "val_correct": val_stats["correct"],
                    "val_total": val_stats["total"],
                    "val_acc": round(val_stats["acc"], 6),
                    "test_correct": test_stats["correct"],
                    "test_total": test_stats["total"],
                    "test_acc": round(test_stats["acc"], 6),
                    "unmatched": len(unmatched),
                }
            )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fieldnames = [
        "model_dir",
        "model_path",
        "setup",
        "dataset_name",
        "dataset_split",
        "dataset_size",
        "val_correct",
        "val_total",
        "val_acc",
        "test_correct",
        "test_total",
        "test_acc",
        "unmatched",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed model directories: {len(model_paths)}")
    print(f"Generated rows: {len(rows)}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
