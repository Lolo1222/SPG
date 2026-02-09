import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available
from accelerate.utils import gather, gather_object, set_seed
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import print_prompt_completions_sample
import wandb
from spg.elbo_grpo_trainer import ElboGRPOTrainer
from spg.swift_grpo_trainer import SWIFTTrainer

if is_peft_available():
    from peft import PeftConfig

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class ElboRLOOTrainer(SWIFTTrainer):
# class ElboRLOOTrainer(ElboGRPOTrainer):
    """
    ELBO-based RLOO Trainer for Diffusion Language Models.
    
    Changes from GRPO:
    1. Advantage: Calculated using (Reward_i - Mean_of_others)
    2. Loss: Uses standard REINFORCE loss (no PPO clipping)
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The ElboRLOOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        # RLOO 通常是 On-Policy 的，建议 num_iterations 设为 1。
        # 如果 num_iterations > 1，这里会重复使用旧样本进行多次梯度更新（风险较高，因为没有 Clip 保护）。
        this_itr_idx = self._step % self.args.num_iterations
        mask_seeds = inputs["mask_seeds"]
        
        # 1. Compute current model ELBO (Log Probabilities)
        # 复用父类的 ELBO 计算逻辑
        input_ids_expanded = input_ids.unsqueeze(0)
        # per_token_logps = self._get_per_token_logps(
        #     model, input_ids_expanded, logits_to_keep, [mask_seeds[this_itr_idx]], completion_mask
        # )
        if self.args.semi_offline_flag:
            generation_mask = inputs["generation_mask"]
            early_rollout_token_index = inputs["early_rollout_token_index"]
            per_token_logps = self._get_per_token_logps_for_semi_and_early(
            model, input_ids_expanded, logits_to_keep, [mask_seeds[this_itr_idx]], generation_mask, early_rollout_token_index, completion_mask
        )
        else:
            per_token_logps = self._get_per_token_logps(
            model, input_ids_expanded, logits_to_keep, [mask_seeds[this_itr_idx]], completion_mask
        )        
        per_token_logps = per_token_logps.squeeze(0) # [batch, logits_to_keep]

        # 2. Retrieve Reference ELBOs (For KL)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # 3. Get Advantages (Calculated via RLOO logic in _generate_and_score_completions)
        advantages = inputs["advantages"] # [batch_size]
        
        # 4. Compute RLOO Loss (Standard REINFORCE)
        # Loss = - Advantage * LogProb
        # GRPO 中使用了 importance sampling (coef)，RLOO 不需要 old_logps，直接针对当前 logps 优化
        
        per_token_loss = -1 * advantages.unsqueeze(1) * per_token_logps
        
        # Add KL Penalty
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
            
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def _generate_and_score_completions_for_semi_and_early(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # 1. 调用父类方法获取基础数据（Prompt ID, Completion ID, Rewards 等）
        # 我们必须完全重写这部分，因为父类没有返回原始 rewards，只返回了计算好的 GRPO advantage。
        # 为了避免大量代码复制，建议稍微修改父类让其返回 raw rewards，
        # 但既然不能修改父类，这里只能复制并替换 Advantage 计算部分。
        
        # --- START COPY FROM ElboGRPOTrainer (With RLOO Mod) ---
        device = self.accelerator.device
        masked_generations = [x["generation"] for x in inputs]
        masked_generation_inputs = self.processing_class(
            text=masked_generations,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        masked_generation_inputs = Trainer._prepare_inputs(self, masked_generation_inputs)
        masked_generation_ids, masked_generation_mask = masked_generation_inputs["input_ids"], masked_generation_inputs["attention_mask"]

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        # Configuration for generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.9
        cfg_scale = self.args.cfg_scale            
        # Generation Logic
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            early_rollout_token_index_all = []
            
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_masked_generation_ids = masked_generation_ids[i:end_idx]
                
                # Note: No prompt_mask passed to generate to match DiffuGRPO style if not needed, 
                # but can be added if your model requires it.
                batch_prompt_completion_ids, batch_early_rollout_token_index = self.generate_for_semi_and_early(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    masked_generation=batch_masked_generation_ids,
                    early_stop_rollout_flag=self.args.early_stop_rollout_flag,
                    early_stop_threshold=self.args.early_stop_threshold,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                )
                prompt_completion_ids_all.append(batch_prompt_completion_ids)
                early_rollout_token_index_all.append(batch_early_rollout_token_index)

                del batch_prompt_ids, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
            if self.args.early_stop_rollout_flag:
                assert early_rollout_token_index_all[0] is not None, "early_rollout_token_index_all[0] is None"
                early_rollout_token_index = torch.cat(early_rollout_token_index_all, dim=0)
            else:
                early_rollout_token_index = None
        # 重新构建 mask 等（与原文件一致）
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(1)

        # Generate random seeds for GRPO iterations
        # This aligns with DiffuGRPO: one seed per iteration step
        if self.args.random_masking:
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            mask_seeds = [42] * self.num_iterations

        # ***************** generation_mask *****************
        if masked_generation_ids.size(1) < gen_length:
            pad_len = gen_length - masked_generation_ids.size(1)
            padding = torch.full(
            (masked_generation_ids.size(0), pad_len),
            self.args.mask_id,
            dtype=masked_generation_ids.dtype,
            device=device,
            )
            masked_generation_ids_get_mask_index = torch.cat([masked_generation_ids.to(device), padding], dim=1)
        else:
            masked_generation_ids_get_mask_index = masked_generation_ids.to(device)[:, :gen_length]

        generation_mask = (masked_generation_ids_get_mask_index == self.args.mask_id).to(torch.bool)     
        # ***************** generation_mask *****************

        # Compute ELBOs (Log Probabilities)
        # Using the same mask_seeds to ensure Old and Ref policies are evaluated on the same masks
        # The internal _get_per_token_logps will expand these using num_t budget
        with torch.no_grad():
            if self.num_iterations > 1:
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                old_per_token_logps = self._get_per_token_logps_for_semi_and_early(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, generation_mask=generation_mask, early_rollout_token_index=early_rollout_token_index, completion_mask=completion_mask
                )
            else:
                old_per_token_logps = None
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps_for_semi_and_early(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, generation_mask=generation_mask, early_rollout_token_index=early_rollout_token_index, completion_mask=completion_mask
                    )

        # Reward Calculation
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        # num_generations 对应原代码中的 self.num_generations (通常由 args.num_return_sequences 决定)
        rloo_rewards = rewards.view(-1, self.num_generations)
        batch_size_groups, k = rloo_rewards.shape
        
        # RLOO 要求每组至少生成 2 个样本
        if k < 2:
            raise ValueError("RLOO requires at least 2 generations per prompt (num_return_sequences >= 2).")

        # 2. Calculate Leave-One-Out Baseline
        # Sum of rewards for each prompt group
        sum_rewards = rloo_rewards.sum(dim=1, keepdim=True) # [Batch, 1]
        
        # Baseline_i = (Sum - r_i) / (K - 1)
        # 这里的数学含义是：除去我(i)之外，其他 K-1 个样本的平均值
        loo_baselines = (sum_rewards - rloo_rewards) / (k - 1)
        
        # 3. Calculate Advantage
        advantages = rloo_rewards - loo_baselines
        
        # Flatten back to match the pipeline shape
        advantages = advantages.view(-1)
        
        # Slice for multi-process (distributed training)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Logging
        mode = "eval" if self.control.should_evaluate else "train"

        self._metrics[mode]["reward"].append(rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            if self.accelerator.is_main_process:
                prompts_to_log = gather_object(prompts_text)
                completions_to_log = gather_object(completions_text)
                rewards_to_log = rewards.tolist()
                
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages, # 这里返回的是 RLOO advantages
            "mask_seeds": mask_seeds,
            "generation_mask": generation_mask, # Lolo1222: add generation_mask to inputs
            "early_rollout_token_index": early_rollout_token_index, # Lolo1222: add 
        }

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # 1. 调用父类方法获取基础数据（Prompt ID, Completion ID, Rewards 等）
        # 我们必须完全重写这部分，因为父类没有返回原始 rewards，只返回了计算好的 GRPO advantage。
        # 为了避免大量代码复制，建议稍微修改父类让其返回 raw rewards，
        # 但既然不能修改父类，这里只能复制并替换 Advantage 计算部分。
        
        # --- START COPY FROM ElboGRPOTrainer (With RLOO Mod) ---
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        # Configuration for generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.9
        cfg_scale = self.args.cfg_scale            
        # Generation Logic
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                
                # Note: No prompt_mask passed to generate to match DiffuGRPO style if not needed, 
                # but can be added if your model requires it.
                batch_prompt_completion_ids = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                )
                prompt_completion_ids_all.append(batch_prompt_completion_ids)
                del batch_prompt_ids, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
        
        # 重新构建 mask 等（与原文件一致）
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(1)

        # Generate random seeds for GRPO iterations
        # This aligns with DiffuGRPO: one seed per iteration step
        if self.args.random_masking:
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            mask_seeds = [42] * self.num_iterations

        # Compute ELBOs (Log Probabilities)
        # Using the same mask_seeds to ensure Old and Ref policies are evaluated on the same masks
        # The internal _get_per_token_logps will expand these using num_t budget
        with torch.no_grad():
            if self.num_iterations > 1:
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, completion_mask
                )
            else:
                old_per_token_logps = None
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, completion_mask
                    )

        # Reward Calculation
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        # num_generations 对应原代码中的 self.num_generations (通常由 args.num_return_sequences 决定)
        rloo_rewards = rewards.view(-1, self.num_generations)
        batch_size_groups, k = rloo_rewards.shape
        
        # RLOO 要求每组至少生成 2 个样本
        if k < 2:
            raise ValueError("RLOO requires at least 2 generations per prompt (num_return_sequences >= 2).")

        # 2. Calculate Leave-One-Out Baseline
        # Sum of rewards for each prompt group
        sum_rewards = rloo_rewards.sum(dim=1, keepdim=True) # [Batch, 1]
        
        # Baseline_i = (Sum - r_i) / (K - 1)
        # 这里的数学含义是：除去我(i)之外，其他 K-1 个样本的平均值
        loo_baselines = (sum_rewards - rloo_rewards) / (k - 1)
        
        # 3. Calculate Advantage
        advantages = rloo_rewards - loo_baselines
        
        # Flatten back to match the pipeline shape
        advantages = advantages.view(-1)
        
        # Slice for multi-process (distributed training)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Logging
        mode = "eval" if self.control.should_evaluate else "train"

        self._metrics[mode]["reward"].append(rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            if self.accelerator.is_main_process:
                prompts_to_log = gather_object(prompts_text)
                completions_to_log = gather_object(completions_text)
                rewards_to_log = rewards.tolist()
                
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages, # 这里返回的是 RLOO advantages
            "mask_seeds": mask_seeds
        }