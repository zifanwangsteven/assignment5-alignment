from vllm import LLM, SamplingParams
from typing import Callable
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import torch
import pandas as pd
import json


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

def load_model(model: str, device: str = "cuda:0", gpu_memory_utilization: float=0.85) -> LLM:
    llm = LLM(
        model=model,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    return llm

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams,
    save_path: str | None = None
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for output, expected in zip(outputs, answers):
        response = output.outputs[0].text
        reward = reward_fn(response, expected)
        results.append({
            "prompt": output.prompt,
            "ground_truth": expected,
            "response": response,
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
            "reward": reward["reward"]
        })

    if save_path:
        with open(save_path, "w") as file:
            json.dump(results, file)

def run_evaluation():
    model = load_model("models/Qwen2.5-Math-1.5B")
    raw_prompts = load_prompts("data/MATH/validation.jsonl")
    prompt_template = load_prompt_template("cs336_alignment/prompts/r1_zero.prompt")

    prompts = []
    answers = []
    for prompt in raw_prompts:
        prompts.append(prompt_template.format(question=prompt["problem"]))
        answers.append(prompt["answer"])
    
    evaluate_vllm(model, 
        reward_fn=r1_zero_reward_fn, 
        prompts=prompts, 
        answers=answers, 
        eval_sampling_params=SamplingParams(
            temperature=1.0,
            max_tokens=1024,
            stop="</answer>",
            include_stop_str_in_output=True
        ),
        save_path="eval/zero_shot_math.json"
    )

def process_eval_results():
    with open("eval/zero_shot_math.json") as file:
        output = json.load(file)
    df = pd.DataFrame(output)

    print("Both correct", df[(df.answer_reward == 1) & (df.format_reward == 1)].shape)
    print("Correct format", df[(df.answer_reward == 0) & (df.format_reward == 1)].shape)
    print("Wrong format", df[(df.format_reward == 0)].shape)
    print(f"Zero shot accuracy: {df.reward.sum() / df.shape[0]:.4f}")

if __name__ == "__main__":
    process_eval_results()