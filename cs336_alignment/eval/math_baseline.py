import torch
import os
import json
from vllm import LLM, SamplingParams
from typing import Callable, List


def evaluate_vllm(
        evaluate_model:LLM,
        reward_fn:Callable[[str, str], dict[str, float]],
        prompts:List[str],
        answers:List[str],
        eval_sampling_params:SamplingParams,
        save_path:str=None
)-> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    outputs = evaluate_model.generate(prompts=prompts, sampling_params=eval_sampling_params)
    for prompt, output, answer in zip(prompts, answers, outputs):
        responce = output.outputs[0].text
        reward = reward_fn(prompt, answer)
        results.append(
            {
                "prompt":prompt,
                "responce": responce,
                "ground_truth":answer,
                "format_reward":reward["format_reward"],
                "answer_reward":reward["answer_reward"],
                "reward":reward["reward"]
            }
        )
    if save_path is not None:
         with open(save_path, 'a') as f:
            # save json list of dicts
            json.dump(results, f, indent = 2)
    return results