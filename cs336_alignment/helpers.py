import os
import json
import torch
from tqdm import tqdm
import collections
from typing import List, Callable
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
        save_path = os.path.join(save_dir, "MATH_results.json")
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
        save_path = os.path.join(log_dir, f"step_{step}.json")
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
    # patch: patch 是 Python 标准库 unittest.mock 里的一个工具，用来临时"打补丁"（monkey-patch）替换某个对象或函数的实现，常用形式有装饰器或上下文管理器。
    # world_size_patch：把所有对 torch.distributed.get_world_size() 的调用都"骗"成返回 1，即告诉 vLLM 当前只在单卡环境下跑，不要启动多进程通信。
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)

    # profiling_patch：把所有对 vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling 的调用都"骗"成返回 None，即告诉 vLLM 当前没有在 profiling 模式下跑，不要启动多进程通信。
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
    # 获取 vLLM 的模型实例
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    
    # 加载策略模型的权重到 vLLM 的模型实例中
    llm_model.load_weights(state_dict.items())