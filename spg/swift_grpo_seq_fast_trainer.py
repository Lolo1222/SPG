# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Union

import torch
import torch.nn.functional as F
import wandb
from accelerate.utils import gather, gather_object
from torch import nn
from transformers import Trainer

from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import unwrap_model_for_generation

from spg.swift_grpo_trainer import SWIFTTrainer
from spg.memory_utils import chunked_token_nll


class SWIFTSeqFastTrainer(SWIFTTrainer):
    """
    Sequence-level SWIFT trainer with a direct fast path.

    Different from the seq adapter version, this class computes and stores
    sequence-level log-probabilities directly for old/ref/current policies,
    avoiding token-level log-probability tensors in the training cache.
    """

    @staticmethod
    def _aggregate_token_logps_to_seq(
        token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate token log-probs to per-sequence log-probs.

        token_logps: [batch, num_t, logits_to_keep]
        completion_mask: [batch, logits_to_keep]
        return: [batch, num_t]
        """
        token_mask = completion_mask.to(token_logps.device, dtype=token_logps.dtype)
        denom = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (token_logps * token_mask.unsqueeze(1)).sum(dim=-1) / denom

    def _get_per_seq_logps(
        self,
        model,
        input_ids: torch.Tensor,
        logits_to_keep: int,
        mask_seeds,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Directly compute per-sequence ELBO log-probabilities.

        Return shape: [num_iterations, batch_size]
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_seq_logps = torch.zeros(num_iterations, batch_size, device=device)

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True

        for iter_idx, mask_seed in enumerate(mask_seeds):
            current_input = input_ids[iter_idx]  # [batch_size, seq_len]

            noisy_batch = self.forward_process(
                current_input,
                prompt_index,
                self.args.mask_id,
                seed=mask_seed,
                completion_mask=completion_mask,
            )

            nll, _ = chunked_token_nll(
                model, self.get_logits, noisy_batch,
                current_input[:, -logits_to_keep:], logits_to_keep,
                getattr(self.args, "logits_micro_batch_size", None), prompt_index,
                self.args.cfg_scale, self.args.mask_id,
            )

            token_logps = -nll.view(batch_size, self.args.num_t, logits_to_keep)
            seq_logps_per_t = self._aggregate_token_logps_to_seq(token_logps, completion_mask)
            per_seq_logps[iter_idx] = seq_logps_per_t.mean(dim=1)

        return per_seq_logps.to(torch.float32)

    def _get_per_seq_logps_for_semi_and_early(
        self,
        model,
        input_ids: torch.Tensor,
        logits_to_keep: int,
        mask_seeds,
        generation_mask: torch.Tensor,
        early_rollout_token_index,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Directly compute per-sequence ELBO log-probs for semi-offline/early-rollout."""
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_seq_logps = torch.zeros(num_iterations, batch_size, device=device)

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True

        for iter_idx, mask_seed in enumerate(mask_seeds):
            current_input = input_ids[iter_idx]  # [batch_size, seq_len]

            noisy_batch = self.forward_process(
                current_input,
                prompt_index,
                self.args.mask_id,
                seed=mask_seed,
                completion_mask=completion_mask,
                generation_mask=generation_mask,
                early_rollout_token_index=early_rollout_token_index,
            )

            nll, _ = chunked_token_nll(
                model, self.get_logits, noisy_batch,
                current_input[:, -logits_to_keep:], logits_to_keep,
                getattr(self.args, "logits_micro_batch_size", None), prompt_index,
                self.args.cfg_scale, self.args.mask_id,
            )

            token_logps = -nll.view(batch_size, self.args.num_t, logits_to_keep)
            seq_logps_per_t = self._aggregate_token_logps_to_seq(token_logps, completion_mask)
            per_seq_logps[iter_idx] = seq_logps_per_t.mean(dim=1)

        return per_seq_logps.to(torch.float32)

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SWIFTSeqFastTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        this_itr_idx = self._step % self.args.num_iterations
        mask_seeds = inputs["mask_seeds"]

        input_ids_expanded = input_ids.unsqueeze(0)  # [1, batch, seq_len]

        if self.args.semi_offline_flag:
            generation_mask = inputs["generation_mask"]
            early_rollout_token_index = inputs["early_rollout_token_index"]
            per_seq_logps = self._get_per_seq_logps_for_semi_and_early(
                model,
                input_ids_expanded,
                logits_to_keep,
                [mask_seeds[this_itr_idx]],
                generation_mask,
                early_rollout_token_index,
                completion_mask,
            )
        else:
            per_seq_logps = self._get_per_seq_logps(
                model,
                input_ids_expanded,
                logits_to_keep,
                [mask_seeds[this_itr_idx]],
                completion_mask,
            )

        per_seq_logps = per_seq_logps.squeeze(0)  # [batch]

        old_per_seq_logps = (
            inputs["old_per_seq_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_seq_logps.detach()
        )

        if self.beta != 0.0:
            ref_per_seq_logps = inputs["ref_per_seq_logps"][this_itr_idx].squeeze(0)
            seq_kl = (
                torch.exp(ref_per_seq_logps - per_seq_logps)
                - (ref_per_seq_logps - per_seq_logps)
                - 1
            )

        advantages = inputs["advantages"]
        coef_1 = torch.exp(per_seq_logps - old_per_seq_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)

        seq_loss1 = coef_1 * advantages
        seq_loss2 = coef_2 * advantages
        seq_loss = -torch.min(seq_loss1, seq_loss2)

        if self.beta != 0.0:
            seq_loss = seq_loss + self.beta * seq_kl

        seq_weight = completion_mask.sum(dim=1).to(seq_loss.dtype)
        loss = (seq_loss * seq_weight).sum() / seq_weight.sum().clamp_min(1.0)

        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(seq_kl.mean()).mean().item()
            )

        is_clipped = (seq_loss1 < seq_loss2).float()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(is_clipped.mean()).mean().item()
        )

        return loss

    def _compute_rewards_advantages_and_log(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prompts,
        prompts_text,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        device,
    ):
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

        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            if self.accelerator.is_main_process:
                prompts_to_log = gather_object(prompts_text)
                completions_to_log = gather_object(completions_text)

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

        return advantages

    def _generate_and_score_completions_for_semi_and_early(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
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
        masked_generation_ids = masked_generation_inputs["input_ids"]

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

        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.9
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            early_rollout_token_index_all = []

            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_masked_generation_ids = masked_generation_ids[i:end_idx]

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
        prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_seq_logps = self._get_per_seq_logps_for_semi_and_early(
                    self.model,
                    prompt_completion_ids_expanded,
                    logits_to_keep,
                    mask_seeds,
                    generation_mask=generation_mask,
                    early_rollout_token_index=early_rollout_token_index,
                    completion_mask=completion_mask,
                )
            else:
                old_per_seq_logps = None

            if self.beta == 0.0:
                ref_per_seq_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_seq_logps = self._get_per_seq_logps_for_semi_and_early(
                        self.model,
                        prompt_completion_ids_expanded,
                        logits_to_keep,
                        mask_seeds,
                        generation_mask=generation_mask,
                        early_rollout_token_index=early_rollout_token_index,
                        completion_mask=completion_mask,
                    )

        advantages = self._compute_rewards_advantages_and_log(
            inputs,
            prompts,
            prompts_text,
            completion_ids,
            completion_mask,
            device,
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_seq_logps": old_per_seq_logps,
            "ref_per_seq_logps": ref_per_seq_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
            "generation_mask": generation_mask,
            "early_rollout_token_index": early_rollout_token_index,
        }

    def _generate_and_score_completions(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
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

        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.9
        cfg_scale = self.args.cfg_scale

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

        prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_seq_logps = self._get_per_seq_logps(
                    self.model,
                    prompt_completion_ids_expanded,
                    logits_to_keep,
                    mask_seeds,
                    completion_mask,
                )
            else:
                old_per_seq_logps = None

            if self.beta == 0.0:
                ref_per_seq_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_seq_logps = self._get_per_seq_logps(
                        self.model,
                        prompt_completion_ids_expanded,
                        logits_to_keep,
                        mask_seeds,
                        completion_mask,
                    )

        advantages = self._compute_rewards_advantages_and_log(
            inputs,
            prompts,
            prompts_text,
            completion_ids,
            completion_mask,
            device,
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_seq_logps": old_per_seq_logps,
            "ref_per_seq_logps": ref_per_seq_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
        }
