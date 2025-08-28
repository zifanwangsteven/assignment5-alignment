from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from torch.nn.functional import softmax
from einops import rearrange
from vllm import LLM, SamplingParams
from typing import Callable
import json
import logging
import os

logger = logging.getLogger(__name__)


def load_prompts(path: str) -> list[dict]:
    data = []
    with open(path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def load_prompt_template(path: str) -> str:
    with open(path, "r") as file:
        prompt = file.read()
    return prompt


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    prompt_tokenized = tokenizer.batch_encode_plus(prompt_strs, padding=False)[
        "input_ids"
    ]
    output_tokenized = tokenizer.batch_encode_plus(output_strs, padding=False)[
        "input_ids"
    ]
    input_ids = []
    labels = []
    response_mask = []

    max_len = max(
        len(prompt) + len(token)
        for prompt, token in zip(prompt_tokenized, output_tokenized)
    )
    for prompt, output in zip(prompt_tokenized, output_tokenized):
        curr_len = len(prompt) + len(output)
        input_ids.append(
            (prompt + output + [tokenizer.pad_token_id] * (max_len - curr_len))[:-1]
        )
        labels.append(
            (prompt[1:] + output + [tokenizer.pad_token_id] * (max_len - curr_len + 1))[
                :-1
            ]
        )
        response_mask.append(
            (
                [0] * (len(prompt) - 1)
                + [1] * len(output)
                + [0] * (max_len - curr_len + 1)
            )[:-1]
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "response_mask": torch.tensor(response_mask, dtype=torch.long, device=device),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    return (
        torch.softmax(logits, -1)
        * (torch.logsumexp(logits, dim=-1, keepdim=True) - logits)
    ).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_tokrn_entropy: bool = False,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    output = dict()
    model.to(device)
    with torch.no_grad():
        output_logits = model.forward(input_ids).logits  # batch seq vocab
        if return_tokrn_entropy:
            output["token_entropy"] = compute_entropy(output_logits)
        output_probs = softmax(output_logits, dim=-1)
        output["log_probs"] = torch.log(
            torch.gather(
                output_probs,
                dim=-1,
                index=rearrange(labels, "batch seq -> batch seq 1"),
            )
        ).squeeze()
    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = (
        masked_normalize(
            -policy_log_probs, response_mask, normalize_constant, dim=-1
        ).mean()
        / gradient_accumulation_steps
    )
    # sum across all tokens for a sample
    loss.backward()
    return loss, dict()


def evaluate_vllm(
    name: str,
    evaluate_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams,
    step: int = 0,
    save_dir: str = None,
) -> list[dict]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    # Use tqdm for a progress bar
    outputs = evaluate_model.generate(prompts, eval_sampling_params)

    logger.info("Evaluating model outputs...")
    for output, prompt, answer in zip(outputs, prompts, answers):
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
                "reward": reward["reward"],
            }
        )

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"{name}_eval_generations_step_{step}.json")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {save_path}")

    return results


def log_generations(
    name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    step: int,
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams,
    train_device: torch.device,
    log_dir: str = None,
    eval_batch_size: int = 6,
) -> list[dict]:
    """
    Log generations from a model and save to disk.
    """

    base_results = evaluate_vllm(
        name=name,
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

    tokenized_results = tokenize_prompt_and_output(
        prompts, outputs, tokenizer, train_device
    )

    # Batch process the log_prob calculation to avoid OOM
    log_probs = []
    token_entropies = []

    for i in range(0, len(prompts)):
        batch_input_ids = tokenized_results["input_ids"][i : i + eval_batch_size]
        batch_labels = tokenized_results["labels"][i : i + eval_batch_size]

        with torch.no_grad():
            log_probs_batch = get_response_log_probs(
                model, batch_input_ids, batch_labels, return_token_entropy=True
            )
        log_probs.append(log_probs_batch["log_probs"])
        token_entropies.append(log_probs_batch["token_entropy"])

    final_log_probs = torch.cat(log_probs, dim=0).to(train_device)
    final_token_entropy = torch.cat(token_entropies, dim=0).to(train_device)

    log_probs = {"log_probs": final_log_probs, "token_entropy": final_token_entropy}
    rewards_tensor = torch.tensor(rewards, device=train_device)
    rewards_positive_mask = rewards_tensor > 0
    rewards_negative_mask = rewards_tensor <= 0

    avg_token_entropy = (
        1.0
        / torch.sum(tokenized_results["response_mask"])
        * masked_normalize(
            log_probs["token_entropy"],
            tokenized_results["response_mask"],
            normalize_constant=1.0,
            dim=None,
        )
    )
    response_lens = tokenized_results["response_mask"].sum(
        dim=-1
    )  # (batch_size, seq_len) -> (batch_size,)
    avg_response_len = response_lens.float().mean()  # (batch_size,) -> (1,)
    correct_lens = response_lens[rewards_positive_mask]  # (B,) -> (b,)
    avg_correct_len = correct_lens.float().mean()  # (b,) -> (1,)
    incorrect_lens = response_lens[rewards_negative_mask]  # (B,) -> (b,)
    avg_incorrect_len = incorrect_lens.float().mean()  # (b,) -> (1,)

    format_accuracy = sum(format_rewards) / len(format_rewards)
    answer_accuracy = sum(answer_rewards) / len(answer_rewards)
    accuracy = sum(rewards) / len(rewards)

    if log_dir is not None:
        save_path = os.path.join(log_dir, f"{name}_eval_metrics_step_{step}.json")
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

        with open(save_path, "w") as f:
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
