from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from torch.nn.functional import softmax
from einops import rearrange
from vllm import LLM, SamplingParams
from typing import Callable
import json
import logging

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
    assert(len(prompt_strs) == len(output_strs))
    prompt_tokenized = tokenizer.batch_encode_plus(prompt_strs, padding=False)["input_ids"]
    output_tokenized = tokenizer.batch_encode_plus(output_strs, padding=False)["input_ids"]
    input_ids = []
    labels = []
    response_mask = []

    max_len = max(len(prompt) + len(token) for prompt, token in zip(prompt_tokenized, output_tokenized))
    for prompt, output in zip(prompt_tokenized, output_tokenized):
        curr_len = len(prompt) + len(output)
        input_ids.append((prompt + output + [tokenizer.pad_token_id] * (max_len-curr_len))[:-1])
        labels.append((prompt[1:] + output + [tokenizer.pad_token_id] * (max_len-curr_len+1))[:-1])
        response_mask.append(([0] * (len(prompt)-1) + [1] * len(output) + [0] * (max_len-curr_len+1))[:-1])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "response_mask": torch.tensor(response_mask, dtype=torch.long, device=device)
    }

def compute_entropy(
    logits: torch.Tensor
) -> torch.Tensor:
    return (torch.softmax(logits, -1) * (torch.logsumexp(logits, dim=-1, keepdim=True)-logits)).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_tokrn_entropy: bool=False,
    device: str | None = None
) -> dict[str, torch.Tensor]:
    output = dict()
    model.to(device)
    with torch.no_grad():
        output_logits = model.forward(input_ids).logits # batch seq vocab
        if return_tokrn_entropy:
            output["token_entropy"] = compute_entropy(output_logits)
        output_probs = softmax(output_logits, dim=-1)
        output["log_probs"] = torch.log(torch.gather(output_probs, dim=-1, index=rearrange(labels, "batch seq -> batch seq 1"))).squeeze()
    return output

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant
    

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = masked_normalize(-policy_log_probs, response_mask, normalize_constant, dim=-1).mean() / gradient_accumulation_steps
    # sum across all tokens for a sample
    loss.backward()
    return loss, dict()

def log_generations(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_sampling_params: SamplingParams,
    prompts: list[str],
    answers: list[str],
):
    """
    It’s always good practice to do some in-the-loop logging that involves generation from your model, 
    and reasoning SFT/RL is no exception. Write a function log_generations that will prompt your model 
    to generate responses for some given prompts (e.g., sampled from the validation set).
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for output, expected in zip(outputs, answers):
        response = output.outputs[0].text
        response_logits = output.outputs[0].logits
        reward = reward_fn(response, expected)
        logger.info({
            "prompt": output.prompt,
            "ground_truth": expected,
            "response": response,
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
            "reward": reward["reward"],
            "avg_entropy": compute_entropy(response_logits).mean()
        })





