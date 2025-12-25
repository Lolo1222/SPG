# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from spg.diffu_grpo_trainer import DiffuGRPOTrainer
from spg.spg_trainer import SPGTrainer
from spg.so_trainer import SOTrainer
from spg.diffu_grpo_config import DiffuGRPOConfig
from spg.reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from spg.data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_sudoku_questions_new,
    set_random_seed,
    get_math_questions,
)
from spg.data_utils import get_math_questions_from_local


# ---------------- Checkpoint / resume utilities -----------------
import glob
import re
import os
import threading
import time


def _find_latest_checkpoint_in_dir(ckpt_dir: str):
    """Find the latest checkpoint directory in `ckpt_dir` matching
    patterns like `checkpoint-<step>`. Returns path or None."""
    if not ckpt_dir:
        return None
    pattern = os.path.join(ckpt_dir, "checkpoint-*")
    matches = glob.glob(pattern)
    if not matches:
        # also accept model_state_dict files
        files = glob.glob(os.path.join(ckpt_dir, "model_state_dict*.pt"))
        if not files:
            return None
        # pick newest by mtime
        return max(files, key=os.path.getmtime)
    # extract step numbers and pick largest
    def step_of(path):
        m = re.search(r"checkpoint-(\d+)", path)
        return int(m.group(1)) if m else -1

    latest = max(matches, key=step_of)
    return latest


def start_periodic_checkpoint_saver(trainer, model, ckpt_dir: str, save_every_steps: int = 100, poll_interval_s: float = 5.0):
    """Start a background thread that saves a checkpoint every `save_every_steps` training steps.

    The thread is a daemon and will exit when the process exits. Only the process that
    considers itself main (via `trainer.is_world_process_zero` when available) will perform saves.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    def _saver_loop():
        next_step = save_every_steps
        saved_steps = set()
        while True:
            try:
                step = getattr(trainer.state, "global_step", 0)
                is_main = getattr(trainer, "is_world_process_zero", None)
                if callable(is_main):
                    is_main = is_main()
                elif is_main is None:
                    # best-effort: if accelerator available, use it
                    accel = getattr(trainer, "accelerator", None)
                    is_main = getattr(accel, "is_main_process", True)

                if is_main and step >= next_step and step not in saved_steps:
                    try:
                        ckpt_path = os.path.join(ckpt_dir, f"checkpoint-{step}")
                        # Prefer a full checkpoint save if available (includes optimizer/scheduler/state)
                        if hasattr(trainer, "save_checkpoint"):
                            try:
                                trainer.save_checkpoint(ckpt_path)
                                print(f"[periodic_ckpt] trainer.save_checkpoint -> {ckpt_path}")
                            except Exception:
                                # fall back
                                if hasattr(trainer, "save_model"):
                                    trainer.save_model(ckpt_path)
                                    print(f"[periodic_ckpt] trainer.save_model -> {ckpt_path}")
                                else:
                                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_state_dict_step_{step}.pt"))
                                    print(f"[periodic_ckpt] saved model_state_dict -> {ckpt_dir}")
                        elif hasattr(trainer, "save_state"):
                            try:
                                trainer.save_state()
                                print(f"[periodic_ckpt] trainer.save_state to {ckpt_dir}")
                            except Exception:
                                if hasattr(trainer, "save_model"):
                                    trainer.save_model(ckpt_path)
                                    print(f"[periodic_ckpt] trainer.save_model -> {ckpt_path}")
                                else:
                                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_state_dict_step_{step}.pt"))
                                    print(f"[periodic_ckpt] saved model_state_dict -> {ckpt_dir}")
                        else:
                            if hasattr(trainer, "save_model"):
                                trainer.save_model(ckpt_path)
                                print(f"[periodic_ckpt] trainer.save_model -> {ckpt_path}")
                            else:
                                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_state_dict_step_{step}.pt"))
                                print(f"[periodic_ckpt] saved model_state_dict -> {ckpt_dir}")
                    except Exception as e:
                        print(f"[periodic_ckpt] failed to save checkpoint at step {step}: {e}")
                    saved_steps.add(step)
                    next_step += save_every_steps
            except Exception:
                pass
            time.sleep(poll_interval_s)

    t = threading.Thread(target=_saver_loop, daemon=True)
    t.start()
    return t

# ------------------------------------------------------------------

# ------------------------------------------------------------------

def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    # elif grpo_config.dataset == "sudoku":
    #     dataset = get_sudoku_questions()
    #     reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "sudoku_new":
        dataset = get_sudoku_questions_new(few_shot=grpo_config.few_shot)
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        # Prefer a local dataset path from grpo_config.local_data_path (if provided).
        local_path = getattr(grpo_config, "local_data_path", None)
        if local_path:
            dataset = get_math_questions_from_local(local_path)
        else:
            dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku", "sudoku_new"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    if grpo_config.trainer == "diffu_grpo":
        # Initialize and run trainer
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "spg":
        trainer = SPGTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "so":
        trainer = SOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    else:
        raise ValueError(f"Invalid trainer: {grpo_config.trainer}")

    print("=== train config ===")
    print(f"max_steps: {grpo_config.max_steps}")
    print(f"num_train_epochs: {grpo_config.num_train_epochs}")
    print(f"per_device_train_batch_size: {grpo_config.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {grpo_config.gradient_accumulation_steps}")
    print(f"num_generations: {grpo_config.num_generations}")
    print(f"dataset length (train_set): {len(train_set)}")
    # 计算预计总迭代数并打印
    batches_per_epoch = (len(train_set) + grpo_config.per_device_train_batch_size * grpo_config.world_size - 1) // (grpo_config.per_device_train_batch_size * getattr(grpo_config, "world_size", 1))
    updates_per_epoch = (batches_per_epoch + grpo_config.gradient_accumulation_steps - 1) // grpo_config.gradient_accumulation_steps
    est_total_updates = updates_per_epoch * grpo_config.num_train_epochs
    print(f"batches_per_epoch: {batches_per_epoch}, updates_per_epoch: {updates_per_epoch}, est_total_updates: {est_total_updates}")

    wandb.init(project="llada_diffu_grpo", config=grpo_config, name=grpo_config.run_name)
    # Setup checkpoint directory and try to resume from latest checkpoint if present
    ckpt_dir = getattr(grpo_config, "output_dir", None) or os.environ.get("CHECKPOINT_DIR", "checkpoints_on_low_mem")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt = _find_latest_checkpoint_in_dir(ckpt_dir)
    if latest_ckpt is not None:
        print(f"[resume] found latest checkpoint: {latest_ckpt}; resuming training from it")
        # If it's a .pt model_state_dict file, load weights and attempt to set trainer.state.global_step
        if latest_ckpt.endswith('.pt'):
            try:
                state_dict = torch.load(latest_ckpt, map_location=device)
                # support both full dict and nested 'model_state_dict' keys
                if isinstance(state_dict, dict) and any(k.startswith('module.') or k in state_dict for k in state_dict):
                    try:
                        model.load_state_dict(state_dict)
                    except RuntimeError:
                        # try common nested structure
                        if 'model_state_dict' in state_dict:
                            model.load_state_dict(state_dict['model_state_dict'])
                        else:
                            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
                else:
                    model.load_state_dict(state_dict)
                # try to infer step from filename
                m = re.search(r"(\d+)", os.path.basename(latest_ckpt))
                inferred_step = int(m.group(1)) if m else 0
                # set resume_from_checkpoint to None but set a flag to set trainer.state later
                resume_from_checkpoint = None
                set_step_after_load = inferred_step
                print(f"[resume] loaded model weights from {latest_ckpt}; inferred step={inferred_step}. Note: optimizer/scheduler state NOT restored.")
            except Exception as e:
                print(f"[resume] failed to load model_state_dict {latest_ckpt}: {e}")
                resume_from_checkpoint = latest_ckpt
                set_step_after_load = None
        else:
            resume_from_checkpoint = latest_ckpt
            set_step_after_load = None
    else:
        resume_from_checkpoint = None
        set_step_after_load = None

    # Determine periodic save interval. Priority: env SAVE_EVERY_STEPS > grpo_config.save_steps (TrainingArguments)
    try:
        if os.environ.get("SAVE_EVERY_STEPS") is not None:
            save_every = int(os.environ.get("SAVE_EVERY_STEPS"))
        else:
            save_every = int(getattr(grpo_config, "save_steps", 100))
    except Exception:
        save_every = 100
    # If we loaded only model weights above and inferred a step, try to update trainer.state BEFORE training
    if resume_from_checkpoint is None and 'set_step_after_load' in locals() and set_step_after_load:
        try:
            trainer.state.global_step = set_step_after_load
            trainer.state.epoch = set_step_after_load / max(1, est_total_updates)
            print(f"[resume] trainer.state.global_step set to {set_step_after_load} (epoch approx {trainer.state.epoch})")
        except Exception as e:
            print(f"[resume] failed to set trainer.state: {e}")

    # If the Trainer already handles checkpointing at the desired frequency, skip our saver to avoid duplicates
    trainer_save_strategy = getattr(grpo_config, "save_strategy", None)
    trainer_save_steps = int(getattr(grpo_config, "save_steps", -1))
    if trainer_save_strategy == "steps" and trainer_save_steps == save_every:
        print(f"[checkpoint] Trainer configured to save every {save_every} steps; skipping periodic saver thread")
    else:
        start_periodic_checkpoint_saver(trainer, model, ckpt_dir, save_every_steps=save_every)

    # Start training, resuming from checkpoint if available
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
