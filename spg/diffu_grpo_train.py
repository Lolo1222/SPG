# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, TrainerCallback
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from maskd_data_utils import generate_masked_sequence, generate_masked_sequence_for_countdown, generate_masked_sequence_for_sudoku_new
from spg.diffu_grpo_trainer import DiffuGRPOTrainer
from spg.spg_trainer import SPGTrainer
from spg.token_spg_trainer import TokenSPGTrainer
# from spg.elbo_grpo_trainer import ElboGRPOTrainer
from spg.swift_grpo_trainer import SWIFTTrainer
from spg.swift_grpo_seq_trainer import SWIFTSeqTrainer
from spg.swift_grpo_seq_fast_trainer import SWIFTSeqFastTrainer
# from spg.swift_grpo_variance_trainer import SWIFTTrainer
from spg.new_unigrpo_trainer import newUniGRPOTrainer
# from spg.elbo_grpo_time_trainer import ElboGRPOTrainer
from spg.elbo_grpo_variance_trainer import ElboGRPOTrainer
from spg.so_grpo_trainer import SOGRPOTrainer
from spg.elbo_rloo_trainer import ElboRLOOTrainer
from spg.newnew_unigrpo_trainer import UniGRPOTrainer
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
from spg.data_utils import get_math_questions_from_local, get_gsm8k_questions_from_local, get_countdown_questions_from_local, get_sudoku_questions_new_from_local


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


def reserve_free_cuda_memory_for_allocator(fraction: float, device: torch.device):
    """Reserve free VRAM in this process's allocator without pinning it forever.

    A tensor that remains alive would reduce the memory available to training.
    Instead, this function allocates one large temporary tensor and deletes it
    without calling ``torch.cuda.empty_cache``. PyTorch keeps the block in its
    caching allocator, so this process can reuse it while other processes see
    it as occupied at the driver level. This is intentionally opt-in: it is a
    cooperative-sharing workaround, not an administrative GPU isolation
    mechanism, and should only be used when the user is allowed to reserve the
    selected GPU.
    """
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        raise ValueError(f"gpu_memory_reserve_fraction must be a number in [0, 1), got {fraction!r}")
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"gpu_memory_reserve_fraction must be in [0, 1), got {fraction}")
    if fraction == 0.0 or not torch.cuda.is_available():
        return 0

    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = int(free_bytes * fraction)
    # Avoid tiny allocations that are unlikely to provide useful protection.
    minimum_bytes = 256 * 1024 * 1024
    if reserve_bytes < minimum_bytes:
        print(
            f"[gpu_reserve] requested fraction={fraction:.3f}, but only {free_bytes / 2**30:.2f} GiB is free; "
            "skipping a sub-256 MiB reservation"
        )
        return 0

    # Fragmentation can make an exact allocation fail even when the driver
    # reports enough aggregate free memory. Back off rather than failing the
    # whole training job, and leave the actual training peak to the normal OOM
    # supervisor in the shell launcher.
    attempted_bytes = reserve_bytes
    reserved_bytes = 0
    while attempted_bytes >= minimum_bytes:
        try:
            reservation = torch.empty((attempted_bytes,), dtype=torch.uint8, device=device)
            reserved_bytes = attempted_bytes
            del reservation
            break
        except torch.cuda.OutOfMemoryError:
            attempted_bytes = int(attempted_bytes * 0.8)
            # PyTorch may have purged an unusable cached block while handling
            # this allocation failure; do not call empty_cache here because a
            # successful reservation must remain visible to other processes.

    if reserved_bytes == 0:
        print("[gpu_reserve] unable to reserve the requested free-memory fraction; continuing without reservation")
        return 0

    torch.cuda.synchronize(device)
    reserved_gib = reserved_bytes / 2**30
    free_after, _ = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) / 2**30
    print(
        f"[gpu_reserve] cached {reserved_gib:.2f} GiB ({fraction:.1%} of {free_bytes / 2**30:.2f} GiB initially free); "
        f"driver-free-after={free_after / 2**30:.2f} GiB, torch-reserved={cached:.2f} GiB"
    )
    return reserved_bytes


class CudaMemoryReservationCallback(TrainerCallback):
    """Top up the allocator after DeepSpeed has initialized its optimizer.

    DeepSpeed ZeRO-1/2 calls ``empty_cache`` during initialization, which
    intentionally clears the early reservation used to protect semi-offline
    preprocessing. ``on_train_begin`` runs after that initialization and before
    the first measured step, so this second reservation persists during the
    actual training loop.
    """

    def __init__(self, fraction: float, device: torch.device):
        self.fraction = fraction
        self.device = device

    def on_train_begin(self, args, state, control, **kwargs):
        reserve_free_cuda_memory_for_allocator(self.fraction, self.device)
        return control

# ------------------------------------------------------------------

# ------------------------------------------------------------------

def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model.config.use_cache = False

    # Reserve before dataset preprocessing as low-confidence semi-offline mask
    # construction also performs large-vocabulary model forwards. The dummy
    # block is inactive cache and remains reusable by Trainer/DeepSpeed later.
    reserve_fraction = getattr(grpo_config, "gpu_memory_reserve_fraction", 0.0)
    reserve_free_cuda_memory_for_allocator(reserve_fraction, device)

    # Load dataset based on configuration
    # If semi-offline used, we will load dataset with generated sequences from local path
    # XXX(Lolo1222): currently only math dataset supports semi-offline mode
    if getattr(grpo_config, "semi_offline_flag", False):
        print(f"=== Using semi-offline dataset from {grpo_config.semi_offline_data_path} ===")
        if grpo_config.dataset == "gsm8k":
            original_dataset = get_gsm8k_questions_from_local(grpo_config.semi_offline_data_path)
            reward_functions = [
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
            ]
        elif grpo_config.dataset == "countdown":
            original_dataset = get_countdown_questions_from_local(grpo_config.semi_offline_data_path)
            reward_functions = [countdown_reward_func]
        elif grpo_config.dataset == "sudoku_new":
            original_dataset = get_sudoku_questions_new_from_local(grpo_config.semi_offline_data_path)
            reward_functions = [sudoku_reward_func]
        elif grpo_config.dataset == "math": 
            original_dataset = get_math_questions_from_local(grpo_config.semi_offline_data_path)
            reward_functions = [
                correctness_reward_func_math,
                boxed_and_answer_tags_format_reward,
            ]            
        else:
            raise ValueError(f"Semi-offline dataset not supported for dataset: {grpo_config.dataset}")

        if "generation" in original_dataset.column_names:
            valid_generation_indices = [
                idx
                for idx, generation in enumerate(original_dataset["generation"])
                if isinstance(generation, str) and generation.strip()
            ]
            removed_generation_count = len(original_dataset) - len(valid_generation_indices)
            if removed_generation_count:
                print(
                    f"[semi_offline] filtering {removed_generation_count} samples with empty/non-string generation "
                    f"before masking ({len(valid_generation_indices)} remain)"
                )
                original_dataset = original_dataset.select(valid_generation_indices)

        max_train_samples = getattr(grpo_config, "max_train_samples", None)
        if max_train_samples is not None:
            if max_train_samples < 1:
                raise ValueError("max_train_samples must be at least 1 when set")
            original_dataset = original_dataset.select(range(min(max_train_samples, len(original_dataset))))

        masking_batch_size = int(getattr(grpo_config, "semi_offline_masking_batch_size", 8))
        if masking_batch_size < 1:
            raise ValueError("semi_offline_masking_batch_size must be at least 1")
        if grpo_config.dataset == "countdown":
            dataset = generate_masked_sequence_for_countdown(original_dataset, tokenizer, model=model, p_question_mask=0, p_gen_mask=grpo_config.semi_offline_ratio, seed=grpo_config.seed, gen_masking_strategy=grpo_config.semi_offline_strategy, batch_size=masking_batch_size)
        elif grpo_config.dataset == "sudoku_new":
            dataset = generate_masked_sequence_for_sudoku_new(original_dataset, tokenizer, model=model, p_question_mask=0, p_gen_mask=grpo_config.semi_offline_ratio, seed=grpo_config.seed, gen_masking_strategy=grpo_config.semi_offline_strategy, batch_size=masking_batch_size)
        else:
            dataset = generate_masked_sequence(original_dataset, tokenizer, model=model, p_question_mask=0, p_gen_mask=grpo_config.semi_offline_ratio, seed=grpo_config.seed, gen_masking_strategy=grpo_config.semi_offline_strategy, batch_size=masking_batch_size)


        
    else:
        print(f"=== Using online dataset: {grpo_config.dataset} ===")
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

        max_train_samples = getattr(grpo_config, "max_train_samples", None)
        if max_train_samples is not None:
            if max_train_samples < 1:
                raise ValueError("max_train_samples must be at least 1 when set")
            dataset = dataset.select(range(min(max_train_samples, len(dataset))))

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku", "sudoku_new"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset


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
    elif grpo_config.trainer == "token_spg":
        trainer = TokenSPGTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "so":
        trainer = SOGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "elbo":
        trainer = ElboGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "swift":
        trainer = SWIFTTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "swift_seq":
        trainer = SWIFTSeqTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "swift_seq_fast":
        trainer = SWIFTSeqFastTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "unigrpo":
        trainer = UniGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "newunigrpo":
        trainer = newUniGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
        )
    elif grpo_config.trainer == "elbo_rloo":
        trainer = ElboRLOOTrainer(
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
    os.environ["WANDB_MODE"] = "offline"
    # A distributed run is one experiment. Creating a separate offline W&B
    # run on every rank adds avoidable I/O and can make multi-GPU runs appear
    # slower without providing additional metrics.
    if trainer.is_world_process_zero():
        wandb.init(project="a800_llada_for_semi", config=grpo_config, name=grpo_config.run_name)
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

    if reserve_fraction:
        trainer.add_callback(CudaMemoryReservationCallback(reserve_fraction, device))

    # Start training, resuming from checkpoint if available
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
