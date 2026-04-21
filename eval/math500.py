# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
import json
import os

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
from generate import generate
import random
import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, is_equiv

MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""


class MATH500Dataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MATH500_SYSTEM_PROMPT,
        subsample=-1,
        subset="all",
        split_file=None,
        split_seed=42,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
        self.subset = subset
        self.split_file = split_file
        self.split_seed = split_seed

        if self.subset in ["val", "test"]:
            split_indices = self._get_subset_indices(len(self.dataset), self.subset)
            base_indices = np.asarray(self.subsample)
            self.subsample = base_indices[np.isin(base_indices, split_indices)]
            print(f"Using MATH-500 {self.subset} subset with {len(self.subsample)} examples")
        elif self.subset != "all":
            raise ValueError(f"Unsupported MATH-500 subset: {self.subset}. Use one of: all, val, test")

    def _build_random_split(self, total_size):
        rng = np.random.RandomState(self.split_seed)
        perm = rng.permutation(total_size)
        midpoint = total_size // 2
        return {
            "val": perm[:midpoint].tolist(),
            "test": perm[midpoint:].tolist(),
        }

    def _get_subset_indices(self, total_size, subset):
        if self.split_file is not None and os.path.exists(self.split_file):
            with open(self.split_file, "r") as f:
                split_data = json.load(f)
            if subset not in split_data:
                raise KeyError(f"Subset key '{subset}' not found in split file: {self.split_file}")
            return np.asarray(split_data[subset])

        if self.split_file is not None:
            print(f"Split file not found at {self.split_file}. Falling back to on-the-fly split with seed {self.split_seed}.")
        return np.asarray(self._build_random_split(total_size)[subset])

    def load_test_dataset(self):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    def load_few_shot_examples(self):
        train_data = load_dataset("EleutherAI/hendrycks_math", ("algebra"), split="train")
        few_shot_examples = []
        samples = random.sample(range(len(train_data)), self.num_examples)
        for example in samples:
            few_shot_examples.append(
                {"question": train_data[example]["problem"], "answer": train_data[example]["solution"]}
            )
        return few_shot_examples

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
