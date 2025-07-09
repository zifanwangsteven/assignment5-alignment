import os
import json
import torch
from tqdm import tqdm
import random
import numpy as np
import collections
from typing import List, Callable, Literal
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

def tokenize_prompt_and_output(prompt_strs:List[str], 
                               output_strs:List[str], 
                               tokenizer:PreTrainedTokenizer,
                               device:torch.device
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for
    the response tokens and 0 for other tokens (prompt or padding).
    Args:
        prompt_strs: list[str]      # List of prompt strings.
        output_strs: list[str]      # List of output strings.
        tokenizer: PreTrainedTokenizer  # Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]:
            input_ids      # shape (batch_size, max_len-1)
            labels         # shifted input ids, shape (batch_size, max_len-1)
            response_mask  # mask on response tokens, shape (batch_size, max_len-1)
    """
    batch_size = len(prompt_strs)
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompt_strs]
    output_tokens = [tokenizer.encode(output) for output in output_strs]
    total_lens = [len(prompt) + len(output) for prompt, output in zip(prompt_tokens, output_tokens)]
    max_len = max(total_lens)

    input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, device=device)
    labels = torch.full((batch_size, max_len), tokenizer.pad_token_id, device=device)
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i in range(batch_size):
        full_tokens = prompt_tokens[i] + output_tokens[i]
        input_ids[i, :len(full_tokens)] = torch.tensor(full_tokens, device=device, dtype=torch.long)
        labels[i, :len(full_tokens)-1] = torch.tensor(full_tokens[1:], device=device, dtype=torch.long)
        response_mask[i, len(prompt_tokens[i])-1:len(full_tokens)-1] = True
    input_ids = input_ids[:, :max_len-1]
    labels = labels[:, :max_len-1]
    response_mask = response_mask[:, :max_len-1]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits:torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the logits.
    Args:
        logits: torch.Tensor, shape (batch_size, seq_len, num_tokens)
    Returns:
        torch.Tensor, shape (batch_size, seq_len)
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
        and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
        response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
        tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
        compute_entropy.
    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
        log pθ(xt |x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
        for each position (present only if return_token_entropy=True).
    """
    output = model(input_ids)
    logits = output.logits
    log_probs = torch.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)
    idx = labels.unsqueeze(-1)
    conditional_log_probs = torch.gather(log_probs, dim=-1, index=idx).squeeze(-1) # (batch_size, seq_len)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": conditional_log_probs, "token_entropy": token_entropy}
    else:
        return {"log_probs": conditional_log_probs}
    

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    processed_tensor = tensor * mask
    sum = torch.sum(processed_tensor, dim=dim)
    return sum / normalize_constant

def evaluate_vllm(
        evaluate_model:LLM,
        reward_fn:Callable[[str, str], dict[str, float]],
        prompts:List[str],
        answers:List[str],
        eval_sampling_params:SamplingParams,
        step:int=0,
        save_dir:str=None
) -> List[dict]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    # Use tqdm for a progress bar
    outputs = evaluate_model.generate(prompts=prompts, sampling_params=eval_sampling_params)
    
    print("Evaluating model outputs...")
    for output, prompt, answer in tqdm(zip(outputs, prompts, answers), total=len(prompts)):
        response = output.outputs[0].text
        # Correctly call the reward function with the model's response and the ground truth answer
        reward = reward_fn(response, answer)
        results.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": answer,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"]
            }
        )

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"eval_generations_step_{step}.json")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {save_path}")

    return results


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    step: int,
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    train_device: torch.device,
    log_dir: str = None,
    eval_batch_size: int = 6,
) -> List[dict]:
    """
    Log generations from a model and save to disk.
    """

    base_results = evaluate_vllm(
        evaluate_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        step=step,
        eval_sampling_params=eval_sampling_params,
        save_dir=log_dir,            
    )
    prompts = [result["prompt"] for result in base_results]
    outputs = [result["response"] for result in base_results]
    format_rewards = [result["format_reward"] for result in base_results]
    answer_rewards = [result["answer_reward"] for result in base_results]
    rewards = [result["reward"] for result in base_results]
    ground_truths = [result["ground_truth"] for result in base_results]

    tokenized_results = tokenize_prompt_and_output(prompts, outputs, tokenizer, train_device)
    
    # Batch process the log_prob calculation to avoid OOM
    log_probs = []
    token_entropies = []
    
    for i in tqdm(range(0, len(prompts), eval_batch_size)):
        batch_input_ids = tokenized_results["input_ids"][i:i + eval_batch_size]
        batch_labels = tokenized_results["labels"][i:i + eval_batch_size]
        
        with torch.no_grad():
            log_probs_batch = get_response_log_probs(
                model, 
                batch_input_ids, 
                batch_labels, 
                return_token_entropy=True
            )
        log_probs.append(log_probs_batch["log_probs"])
        token_entropies.append(log_probs_batch["token_entropy"])

    final_log_probs = torch.cat(log_probs, dim=0).to(train_device)
    final_token_entropy = torch.cat(token_entropies, dim=0).to(train_device)
    
    log_probs = {"log_probs": final_log_probs, "token_entropy": final_token_entropy}
    rewards_tensor = torch.tensor(rewards, device=train_device)
    rewards_positive_mask = rewards_tensor > 0
    rewards_negative_mask = rewards_tensor <= 0

    avg_token_entropy = 1.0/torch.sum(tokenized_results["response_mask"]) * masked_normalize(log_probs["token_entropy"], tokenized_results["response_mask"], normalize_constant=1.0, dim=None)  
    response_lens = tokenized_results["response_mask"].sum(dim=-1) # (batch_size, seq_len) -> (batch_size,)
    avg_response_len = response_lens.float().mean() # (batch_size,) -> (1,)
    correct_lens = response_lens[rewards_positive_mask] # (B,) -> (b,)
    avg_correct_len = correct_lens.float().mean() # (b,) -> (1,)
    incorrect_lens = response_lens[rewards_negative_mask] # (B,) -> (b,)
    avg_incorrect_len = incorrect_lens.float().mean() # (b,) -> (1,)
    
    format_accuracy = sum(format_rewards) / len(format_rewards)
    answer_accuracy = sum(answer_rewards) / len(answer_rewards)
    accuracy = sum(rewards) / len(rewards)
    
    if log_dir is not None:
        save_path = os.path.join(log_dir, f"eval_metrics_step_{step}.json")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        # Prepare the full results dictionary for saving
        full_results_to_save = {
            "step": step,
            "metrics": {
                "format_accuracy": format_accuracy,
                "answer_accuracy": answer_accuracy,
                "accuracy": accuracy,
                "avg_token_entropy": avg_token_entropy.item(),
                "avg_response_len": avg_response_len.item(),
                "avg_correct_len": avg_correct_len.item(),
                "avg_incorrect_len": avg_incorrect_len.item(),
            },
        }

        with open(save_path, 'w') as f:
            json.dump(full_results_to_save, f, indent=2)
        print(f"Results saved to {save_path}")

    return {
        "step": step,
        "prompts": prompts,
        "outputs": outputs,
        "ground_truths": ground_truths,
        "format_rewards": format_rewards,
        "answer_rewards": answer_rewards,
        "rewards": rewards,
        "format_accuracy": format_accuracy,
        "answer_accuracy": answer_accuracy,
        "accuracy": accuracy,
        "avg_token_entropy": avg_token_entropy,
        "avg_response_len": avg_response_len,
        "avg_correct_len": avg_correct_len,
        "avg_incorrect_len": avg_incorrect_len,
    }

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    vllm_set_random_seed(seed)
    # patch: patch is a tool in the Python standard library unittest.mock, used to temporarily "patch" (monkey-patch) the implementation of a specific object or function, commonly used as a decorator or context manager.
    # world_size_patch: patch all calls to torch.distributed.get_world_size() to return 1, telling vLLM that it is currently running in a single-card environment and not to start multi-process communication.
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)

    # profiling_patch: patch all calls to vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling to return None, telling vLLM that it is currently not running in profiling mode and not to start multi-process communication.
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return llm
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    # get the model instance of vLLM
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    
    # load the policy model weights into the vLLM model instance
    llm_model.load_weights(state_dict.items())


def compute_group_normalized_rewards(
        reward_fn:Callable[[str, str], dict[str, float]],
        rollout_responses:List[str],
        repeated_ground_truths:List[str],
        group_size:int,
        advantage_epsilon:float=1e-5,
        normalize_std:bool=True,
        device:torch.device=torch.device("cpu")
) -> List[float]:
        """
        Compute rewards for each group of rollout responses, normalized by the group size.
        Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.
        Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
        """
        rewards = [reward_fn(rollout_response, ground_truth) for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths)]
        raw_rewards = [reward["reward"] for reward in rewards]
        raw_rewards_tensor = torch.tensor(raw_rewards, device=device)
        raw_rewards_tensor = raw_rewards_tensor.reshape(-1, group_size)
        mean = raw_rewards_tensor.mean(dim=-1, keepdim=True)
        std = raw_rewards_tensor.std(dim=-1, keepdim=True)
        if normalize_std:
            advantages = (raw_rewards_tensor - mean) / (std + advantage_epsilon)
        else:
            advantages = raw_rewards_tensor - mean
        advantages = advantages.reshape(-1)
        metadata = {}
        return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages:torch.Tensor,
        policy_log_probs:torch.Tensor,
) -> torch.Tensor:  
    """
    Compute the naive policy gradient loss.
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the naive policy gradient loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
        advantages:torch.Tensor,
        policy_log_probs:torch.Tensor,
        old_log_probs:torch.Tensor,
        cliprange:float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), scalar
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.
        cliprange: float The clip range for the importance ratio.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the GRPO-Clip loss.
    """
    importance_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_importance_ratio = torch.clamp(importance_ratio, 1 - cliprange, 1 + cliprange)
    clipped_loss = clipped_importance_ratio * advantages
    unclipped_loss = importance_ratio * advantages
    loss = -torch.min(clipped_loss, unclipped_loss)
    metadata = {"clipped": clipped_loss < unclipped_loss}
    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
      policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
      loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
      raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
      advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
      old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
      cliprange Required for "grpo_clip"; scalar ε used for clipping.

    Returns:
      tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    metadata = {}
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor, 
    mask: torch.Tensor, 
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.
    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over the elements with mask value 1.
        dim: int | None, the dimension to compute the mean over.
    Returns:
        torch.Tensor, the mean of the tensor along the dimension.
    """
    processed_tensor = tensor * mask
    sum = torch.sum(processed_tensor, dim=dim)
    return sum / mask.sum(dim=dim)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False