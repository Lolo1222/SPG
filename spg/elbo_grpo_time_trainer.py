# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

# [新增] 引入时间和IO相关库
import time
import json
import os
from contextlib import contextmanager

if is_peft_available():
    from peft import PeftConfig

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class ElboGRPOTrainer(GRPOTrainer):
    """
    ELBO-based Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # [新增] 初始化时间统计字典
        self.phase_time_stats = {
            "total_generation": 0.0,      # 生成阶段耗时
            "total_ref_logp": 0.0,        # Old/Ref LogP 计算耗时
            "total_reward": 0.0,          # Reward 计算耗时
            "total_compute_loss": 0.0     # 训练/ELBO估计耗时
        }

        # Ensure num_t is set (Monte Carlo Budget)
        if not hasattr(self.args, "num_t"):
            warnings.warn("args.num_t is not set. Defaulting to 8 for ELBO estimation budget.")
            self.args.num_t = 8

    # [新增] 计时上下文管理器
    @contextmanager
    def _record_time(self, phase_name: str):
        """记录特定阶段的时间，包含CUDA同步以保证准确性"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        self.phase_time_stats[phase_name] += (end_time - start_time)

    # [新增] 保存统计结果的方法
    def save_time_stats(self):
        """将统计结果保存到 output_dir"""
        if self.accelerator.is_main_process and self.args.output_dir:
            output_path = os.path.join(self.args.output_dir, "time_profiling_stats.json")
            # 添加可读的小时/分钟格式
            readable_stats = self.phase_time_stats.copy()
            for k, v in self.phase_time_stats.items():
                readable_stats[f"{k}_readable"] = f"{v/60:.2f} mins"
            
            with open(output_path, "w") as f:
                json.dump(readable_stats, f, indent=4)
            # 可选：打印到控制台
            # print(f"\n[Time Stats Updated] Saved to {output_path}")

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # [修改] 使用上下文管理器包裹整个 compute_loss 过程
        with self._record_time("total_compute_loss"):
            if return_outputs:
                raise ValueError("The ElboGRPOTrainer does not support returning outputs")

            prompt_ids = inputs["prompt_ids"]
            completion_ids = inputs["completion_ids"]
            completion_mask = inputs["completion_mask"]
            
            # Combine prompt and completion
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            logits_to_keep = completion_ids.size(1)

            # Get the current iteration index
            this_itr_idx = self._step % self.args.num_iterations
            mask_seeds = inputs["mask_seeds"] 
            
            # 1. Compute current model ELBO
            input_ids_expanded = input_ids.unsqueeze(0) 
            
            per_token_logps = self._get_per_token_logps(
                model, input_ids_expanded, logits_to_keep, [mask_seeds[this_itr_idx]], completion_mask
            )
            per_token_logps = per_token_logps.squeeze(0) 

            # 2. Retrieve Reference and Old ELBOs
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            old_per_token_logps = inputs["old_per_token_logps"][this_itr_idx].squeeze(0)

            # 3. Compute KL Divergence
            if self.beta != 0.0:
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )

            # 4. Compute GRPO Loss
            advantages = inputs["advantages"] 
            
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
            
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
                
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

            # Log metrics
            mode = "eval" if self.control.should_evaluate else "train"
            if self.beta != 0.0:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["clip_ratio"].append(
                self.accelerator.gather_for_metrics(clip_ratio).mean().item()
            )

        return loss

    def forward_process(self, batch, prompt_index, mask_id, seed=None, completion_mask=None):
        """
        Generates noisy samples for ELBO estimation.
        Supports generating num_t samples per sequence as per user requirement.
        Code logic adopted from spg_trainer.py.
        """
        set_seed(seed)
        
        # Defaults
        num_t = getattr(self.args, "num_t", 1)
        # Using "random" strategy as default for VRPO (random timesteps)
        forward_type = getattr(self.args, "forward_type", "random")
        
        b, l = batch.shape
        device = batch.device

        if forward_type == "random":
            # Sample num_t different masks per sequence
            gen_length = (l - prompt_index.sum()).item()
            completion_length = completion_mask.sum(-1)
            is_mask = torch.zeros((b, num_t, gen_length), dtype=torch.bool, device=batch.device)
            
            # VRPO Strategy: Allocate sample budget across distinct diffusion timesteps
            # Ideally we want stratified sampling or just random sampling across [0,1]
            # Here we implement the discrete masking logic from spg_trainer
            min_t = getattr(self.args, "min_t", 0)
            max_t = getattr(self.args, "max_t", 1)

            for i in range(b):
                start_mask_num = max(int(completion_length[i] * min_t), 1)
                end_mask_num = min(int(completion_length[i] * max_t), completion_length[i])
                
                # Sample num_t mask counts (representing t)
                mask_num = torch.randint(start_mask_num, end_mask_num + 1, (1, num_t), device=batch.device) # [1, num_t]
                
                # Create masks
                indices = torch.arange(gen_length, device=batch.device).repeat(1, num_t, 1) # [1, num_t, gen_length]
                is_mask[[i], :, :] = indices < mask_num.unsqueeze(2)
                
                # Shuffle mask positions for each t
                for j in range(num_t):
                    is_mask[i, j, :completion_length[i]] = is_mask[i, j, :completion_length[i]][torch.randperm(completion_length[i])]
            
            # Pad prompt part
            is_mask = torch.cat((torch.zeros(b, num_t, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=2)
            
            # Handle completion mask (EOS logic)
            completion_mask_append = torch.cat((torch.ones(b, num_t, prompt_index.sum(), dtype=torch.bool, device=batch.device), completion_mask.unsqueeze(1).repeat(1, num_t, 1)), dim=2).to(torch.bool)
            
            # Handle prompt masking if enabled
            if getattr(self.args, "use_mask_prompt", False):
                t_p = torch.ones(b, num_t, device=batch.device) * self.args.p_mask_prompt
                random_matrix = torch.rand((b, num_t, l), device=batch.device)
                is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(2))
                is_mask = (is_mask_prompt | is_mask) | ~completion_mask_append
            else:
                is_mask = is_mask | ~completion_mask_append

            # Create noisy batch: [b, num_t, l]
            noisy_batch = torch.where(is_mask, mask_id, batch.unsqueeze(1).repeat(1, num_t, 1))
            
            return noisy_batch
        
        else:
            # Fallback for "all" or other simple modes if num_t=1
             raise NotImplementedError(f"forward_type {forward_type} not fully implemented in this snippet, please use 'random'")

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id, prompt_mask=None):
        # Adapted from spg_trainer to handle [batch, num_t, len] input
        multisample = False
        if len(batch.shape) == 3:
            multisample = True
            bsz, num_t, l = batch.shape
            batch = batch.view(-1, l)
            if prompt_mask is not None:
                prompt_len = prompt_mask.shape[-1]
                prompt_mask = prompt_mask.unsqueeze(1).repeat(1, num_t, 1).view(-1, prompt_len)

        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index_expanded = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index_expanded] = mask_id
            batch_in = torch.cat([batch, un_batch])
            if prompt_mask is not None:
                prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
        else:
            batch_in = batch
            
        logits = model(batch_in, attention_mask=prompt_mask).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            
        if multisample:
            logits = logits.view(bsz, num_t, l, -1)
            
        return logits
    
    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds, completion_mask):
        """
        Calculate per-token ELBO log probabilities.
        Return shape: [num_iterations, batch_size, logits_to_keep]
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Output tensor
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True 

        # Loop over GRPO iterations
        for iter_idx, mask_seed in enumerate(mask_seeds):
            current_input = input_ids[iter_idx] # [batch_size, seq_len]
            
            # 1. Generate noisy samples with budget num_t
            # Shape: [batch_size, num_t, seq_len]
            noisy_batch = self.forward_process(
                current_input, prompt_index, self.args.mask_id, seed=mask_seed, completion_mask=completion_mask
            )
            
            # 2. Compute logits
            # Shape: [batch_size, num_t, seq_len, vocab_size]
            logits = self.get_logits(
                model, noisy_batch, prompt_index, self.args.cfg_scale, self.args.mask_id
            )
            
            # 3. Compute Cross Entropy Loss (NLL) for completion tokens
            completion_logits = logits[:, :, -logits_to_keep:, :] # [B, T, L, V]
            
            # Targets: [B, T, L]
            targets = current_input[:, -logits_to_keep:].unsqueeze(1).repeat(1, self.args.num_t, 1)
            
            flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
            flat_targets = targets.reshape(-1)
            
            # reduction='none' to get per-token loss
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            
            # Reshape back to [Batch, Num_T, Logits_To_Keep]
            nll_per_token = loss.view(batch_size, self.args.num_t, logits_to_keep)
            
            # 4. Compute ELBO (Mean over Num_T)
            # LogProb = -NLL
            # This averages the log-probabilities over the MC samples
            avg_logps = -nll_per_token.mean(dim=1) # [Batch, Logits_To_Keep]
            
            per_token_logps[iter_idx] = avg_logps

        return per_token_logps.to(torch.float32)

    def add_gumbel_noise(self, logits, temperature, dtype):
        if temperature == 0.0:
            return logits
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise
    
    def get_num_transfer_tokens(self, mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = base.expand(-1, steps).clone()
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1
        return num_transfer_tokens.to(torch.int64)

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """Generation logic identical to DiffuGRPOTrainer"""
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id
            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length
                block_mask_index = x[:, start_idx:end_idx] == mask_id
                
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise
                            
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            x0_p[:, end_idx:] = -np.inf
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index

            return x
        

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
                
                # [新增] 每次生成新的batch后，更新一次统计文件
                if self.state.global_step > 0:
                    self.save_time_stats()
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(inputs)

        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
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

        # [修改] 阶段 1: 统计生成时间
        with self._record_time("total_generation"):
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                generation_batch_size = self.args.generation_batch_size
                prompt_completion_ids_all = []
                
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    
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

        # Extract completions and Create Masks
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(1)

        if self.args.random_masking:
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            mask_seeds = [42] * self.num_iterations

        # [修改] 阶段 2: 统计 Reference/Old LogP 计算时间
        with self._record_time("total_ref_logp"):
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

                if self.beta == 0.0:
                    ref_per_token_logps = None
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, completion_mask
                        )

        # [修改] 阶段 3: 统计 Reward 计算时间
        with self._record_time("total_reward"):
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

            # Advantage Calculation
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) 

            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            advantages = advantages[process_slice]

        # Logging
        mode = "eval" if self.control.should_evaluate else "train"

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

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
            "advantages": advantages,
            "mask_seeds": mask_seeds
        }