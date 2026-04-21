import argparse
import json
import os

import numpy as np
from datasets import load_dataset

# python dataset/split_math500_test.py --seed 42 --output dataset/math500_test_split_seed42.json
def build_random_split(total_size, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_size)
    midpoint = total_size // 2
    return {
        "val": perm[:midpoint].tolist(),
        "test": perm[midpoint:].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/math500_test_split_seed42.json",
        help="Path to save split indices JSON",
    )
    args = parser.parse_args()

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    split = build_random_split(len(dataset), seed=args.seed)

    payload = {
        "dataset": "HuggingFaceH4/MATH-500",
        "source_split": "test",
        "seed": args.seed,
        "total_size": len(dataset),
        "val_size": len(split["val"]),
        "test_size": len(split["test"]),
        "val": split["val"],
        "test": split["test"],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved split file to {args.output}")
    print(f"val size: {payload['val_size']}, test size: {payload['test_size']}")


if __name__ == "__main__":
    main()
