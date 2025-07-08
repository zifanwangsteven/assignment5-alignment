import torch
from typing import Literal
from cs336_alignment.helpers import (
    compute_policy_gradient_loss,
    masked_mean,
)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask) # (batch_size, sequence_length) -> (batch_size)
    loss = loss.mean() # (batch_size) -> (1)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, metadata