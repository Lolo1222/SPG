import torch
import numpy as np
import warnings
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union, Sized
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer
from datasets import Dataset, IterableDataset
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from trl.data_utils import maybe_apply_chat_template, is_conversational
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer
from spg.diffu_grpo_trainer import DiffuGRPOTrainer
from accelerate.utils import gather, gather_object
from trl.import_utils import is_rich_available
from trl.trainer.utils import print_prompt_completions_sample


class SOTrainer(DiffuGRPOTrainer):
    """
    SOTrainer: a small variant of DiffuGRPO adapted to inputs that already contain
    masked responses (prompt + masked_response). The model continues diffusion from
    that input and, when computing loss, only tokens that were originally masked
    in the response are used to form the loss / gradients.

    Assumptions (inferred):
    - Each training example contains both a standard `prompt` (used for rewards)
      and a `masked_response` (string) that has mask tokens (self.args.mask_id)
      where tokens should be generated.
    - If the dataset uses conversational templates, `maybe_apply_chat_template`
      is still used for the `prompt` text (for rewards); the `masked_response`
      is appended as-is after tokenization.

    Provides:
    - A generation routine that accepts an `initial_completion` (the masked
      response token ids) and continues diffusion only over the completion
      positions.
    - A compute_loss that only sums per-token losses over positions that were
      originally masked in the provided `masked_response`.
    """

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        original_mask = inputs.get("original_mask")
        mask_seeds = inputs["mask_seeds"]

        if original_mask is None:
            raise ValueError("SOTrainer expects 'original_mask' in generated inputs")

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)

        # per-token log probs (num_iterations, batch, logits_to_keep)
        per_token_logps = self._get_per_token_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed])

        # Compute KL if requested
        if self.beta != 0.0:
            ref_per_token_logps = inputs.get("ref_per_token_logps")
            if ref_per_token_logps is None:
                raise ValueError("ref_per_token_logps required when beta != 0.0")
            ref_per_token_logps = ref_per_token_logps[this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss: only positions that were originally masked in the response
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Only keep tokens that are within the valid completion (completion_mask)
        # and that were originally masked in the provided masked_response (original_mask)
        effective_mask = (completion_mask.bool() & original_mask.bool()).to(per_token_loss.dtype)
        denom = effective_mask.sum()
        if denom == 0:
            # fallback to using the completion_mask if nothing was originally masked
            effective_mask = completion_mask.bool().to(per_token_loss.dtype)
            denom = effective_mask.sum().clamp(min=1.0)

        loss = (per_token_loss * effective_mask).sum() / denom

        # Logging similar to DiffuGRPOTrainer but restricted to effective_mask
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * effective_mask).sum() / denom
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * effective_mask).sum() / denom
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

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
        initial_completion: Optional[torch.Tensor] = None,
    ):
        """
        Modified generate that accepts an `initial_completion` tensor of shape
        [bs, gen_length] which contains a mixture of token ids and mask ids. The
        generation will continue denoising only over the gen_length positions.
        """
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            # If an initial completion is provided, put it into x after the prompt
            if initial_completion is not None:
                # initial_completion expected shape [bs, gen_length]
                if initial_completion.shape[0] != bs or initial_completion.shape[1] != gen_length:
                    raise ValueError("initial_completion must have shape [bs, gen_length]")
                x[:, prompt.shape[1] :] = initial_completion.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks) if num_blocks > 0 else 0

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
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index

            return x

    def _generate_and_score_completions(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Expect inputs to contain 'prompt' (used for rewards) and 'masked_response'
        (string) which contains mask tokens. Tokenize prompt and masked_response
        separately, then generate starting from prompt + masked_response. Return
        the same fields as DiffuGRPOTrainer plus an `original_mask` boolean that
        marks which completion token positions were masked initially.
        """
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        # Build prompt_texts for reward computation as usual
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # Tokenize base prompts (unmasked)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs.get("attention_mask", None)

        # Tokenize masked_responses provided by dataset
        masked_responses = []
        for ex in inputs:
            if "masked_response" in ex:
                masked_responses.append(ex["masked_response"])
            elif "masked_completion" in ex:
                masked_responses.append(ex["masked_completion"])
            elif "masked_prompt" in ex:
                # if full masked prompt is provided, we still need to split later
                # here treat masked_prompt as only the completion portion is required
                masked_responses.append("")
            else:
                raise ValueError(
                    "SOTrainer requires each example to contain a 'masked_response' field with mask tokens"
                )

        completion_inputs = self.processing_class(
            text=masked_responses,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        completion_ids = completion_inputs["input_ids"].to(device)

        # Replace padding token ids with mask_id so they are treated as masked positions
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_token_id is not None:
            completion_ids = torch.where(completion_ids == pad_token_id, self.args.mask_id, completion_ids)

        # gen_length is the completion length
        gen_length = completion_ids.size(1)

        # Generation (process in batches like other trainers)
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_initial_completion = completion_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx] if prompt_mask is not None else None

                batch_prompt_completion_ids = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=self.args.diffusion_steps,
                    gen_length=gen_length,
                    block_length=self.args.block_length,
                    temperature=self.args.temperature or 0.0,
                    cfg_scale=self.args.cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                    initial_completion=batch_initial_completion,
                )

                prompt_completion_ids_all.append(batch_prompt_completion_ids)

                del batch_prompt_ids, batch_initial_completion, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Split back into prompt and completion
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Compute which completion positions were originally masked
        original_mask = (completion_inputs["input_ids"].to(device) == self.args.mask_id).int()

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Generate mask seeds
        if self.args.random_masking:
            mask_seeds = torch.randint(0, 2 ** 12, (self.num_iterations,), device=device)
        else:
            mask_seeds = [42] * self.num_iterations

        # Compute rewards like parent class
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            with profiling_context(self, getattr(reward_func, "__name__", str(reward_func))):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion", "masked_response"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards and advantages (keep same logic as parent)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Metrics
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(prompts_to_log, completions_to_log, rewards_to_log, self.state.global_step)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": None,
            "ref_per_token_logps": None,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
            "original_mask": original_mask,
        }
