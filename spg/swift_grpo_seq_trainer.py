# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Union

import torch
from trl.extras.profiling import profiling_decorator

from spg.swift_grpo_trainer import SWIFTTrainer


class SWIFTSeqTrainer(SWIFTTrainer):
    """
    Sequence-level SWIFT GRPO trainer.

    This trainer reuses SWIFT generation/reward pipelines but changes policy
    optimization from token-level clipping to sequence-level clipping.
    """

    @staticmethod
    def _aggregate_per_token_logps_to_seq(
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate token log-probabilities into sequence log-probabilities.

        Args:
            per_token_logps: Tensor with shape [num_iterations, batch_size, logits_to_keep]
            completion_mask: Tensor with shape [batch_size, logits_to_keep]

        Returns:
            Tensor with shape [num_iterations, batch_size]
        """
        token_mask = completion_mask.to(per_token_logps.device).unsqueeze(0).to(per_token_logps.dtype)
        denom = token_mask.sum(dim=-1).clamp_min(1.0)
        return (per_token_logps * token_mask).sum(dim=-1) / denom

    def _get_per_seq_logps(
        self,
        model,
        input_ids: torch.Tensor,
        logits_to_keep: int,
        mask_seeds,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_token_logps = super()._get_per_token_logps(
            model,
            input_ids,
            logits_to_keep,
            mask_seeds,
            completion_mask,
        )
        return self._aggregate_per_token_logps_to_seq(per_token_logps, completion_mask)

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
        per_token_logps = super()._get_per_token_logps_for_semi_and_early(
            model,
            input_ids,
            logits_to_keep,
            mask_seeds,
            generation_mask=generation_mask,
            early_rollout_token_index=early_rollout_token_index,
            completion_mask=completion_mask,
        )
        return self._aggregate_per_token_logps_to_seq(per_token_logps, completion_mask)

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SWIFTSeqTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        # Combine prompt and completion for current policy evaluation.
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

    def _generate_and_score_completions(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        outputs = super()._generate_and_score_completions(inputs)
        completion_mask = outputs["completion_mask"]

        old_per_token_logps = outputs.get("old_per_token_logps", None)
        if old_per_token_logps is not None:
            outputs["old_per_seq_logps"] = self._aggregate_per_token_logps_to_seq(
                old_per_token_logps,
                completion_mask,
            )
        else:
            outputs["old_per_seq_logps"] = None

        ref_per_token_logps = outputs.get("ref_per_token_logps", None)
        if ref_per_token_logps is not None:
            outputs["ref_per_seq_logps"] = self._aggregate_per_token_logps_to_seq(
                ref_per_token_logps,
                completion_mask,
            )
        else:
            outputs["ref_per_seq_logps"] = None

        return outputs

    def _generate_and_score_completions_for_semi_and_early(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        outputs = super()._generate_and_score_completions_for_semi_and_early(inputs)
        completion_mask = outputs["completion_mask"]

        old_per_token_logps = outputs.get("old_per_token_logps", None)
        if old_per_token_logps is not None:
            outputs["old_per_seq_logps"] = self._aggregate_per_token_logps_to_seq(
                old_per_token_logps,
                completion_mask,
            )
        else:
            outputs["old_per_seq_logps"] = None

        ref_per_token_logps = outputs.get("ref_per_token_logps", None)
        if ref_per_token_logps is not None:
            outputs["ref_per_seq_logps"] = self._aggregate_per_token_logps_to_seq(
                ref_per_token_logps,
                completion_mask,
            )
        else:
            outputs["ref_per_seq_logps"] = None

        return outputs
