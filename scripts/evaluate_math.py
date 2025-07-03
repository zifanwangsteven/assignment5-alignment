import torch
import os
import json
import collections
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from typing import Callable, List
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

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

def compute_metrics(results: List[dict], save_dir: str = None):
    """
    Compute evaluation metrics and save analysis to file.
    """
    category_counts = collections.Counter()
    format_only_examples = []
    incorrect_examples = []

    for res in results:
        if res["format_reward"] == 1.0 and res["answer_reward"] == 1.0:
            category_counts["correct"] += 1
        elif res["format_reward"] == 1.0 and res["answer_reward"] == 0.0:
            category_counts["format_only"] += 1
            if len(format_only_examples) < 10:
                format_only_examples.append(res)
        else: # format_reward == 0.0
            category_counts["incorrect"] += 1
            if len(incorrect_examples) < 10:
                incorrect_examples.append(res)

    # Calculate accuracy
    accuracy = category_counts['correct'] / len(results) if results else 0
    
    # Prepare analysis output
    analysis_output = []
    analysis_output.append(f"Total examples: {len(results)}")
    analysis_output.append(f"(1) Correct (Format=1, Answer=1): {category_counts['correct']}")
    analysis_output.append(f"(2) Format Only (Format=1, Answer=0): {category_counts['format_only']}")
    analysis_output.append(f"(3) Incorrect Format (Format=0): {category_counts['incorrect']}")
    
    analysis_output.append("\n--- Examples of Format Reward 0 ---")
    for i, ex in enumerate(incorrect_examples):
        analysis_output.append(f"\nExample {i+1}:")
        analysis_output.append(f"Response: {ex['response']}")
        analysis_output.append(f"Ground Truth: {ex['ground_truth']}")

    analysis_output.append("\n--- Examples of Format Reward 1, Answer Reward 0 ---")
    for i, ex in enumerate(format_only_examples):
        analysis_output.append(f"\nExample {i+1}:")
        analysis_output.append(f"Response: {ex['response']}")
        analysis_output.append(f"Ground Truth: {ex['ground_truth']}")
        
    analysis_output.append(f"\n--- Final Performance ---")
    analysis_output.append(f"Zero-shot accuracy on validation set: {accuracy:.4f} ({accuracy:.2%})")
    
    # Save analysis to file
    if save_dir is not None:
        save_path = os.path.join(save_dir, "MATH_analysis.txt")
        with open(save_path, 'w') as f:
            for line in analysis_output:
                f.write(line + '\n')
        print(f"Analysis saved to {save_path}")
    
    return {
        "accuracy": accuracy,
        "category_counts": category_counts,
        "format_only_examples": format_only_examples,
        "incorrect_examples": incorrect_examples
    }

def load_MATH(data_path="data/MATH/validation.jsonl"):
    """
    Load MATH dataset problems and solutions from JSON files.
    
    Args:
        data_path: Path to the directory containing MATH JSON files
    
    Returns:
        Tuple of (questions, answers) lists
    """
    questions = []
    answers = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading MATH data"):
            try:
                data = json.loads(line)
                if 'problem' in data and 'answer' in data:
                    questions.append(data['problem'])
                    answers.append(data['answer'])
                else:
                    print(f"Warning: Missing 'problem' or 'answer' in {data_path}")
                    
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
    print(f"Loaded {len(questions)} questions and {len(answers)} answers")
    return questions, answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--MATH_data_path", type=str, default="./data/MATH/validation.jsonl")
    parser.add_argument("--prompt_template_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save evaluation results and analysis")
    args = parser.parse_args()


    print("Loading model...")
    model = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    print("Model loaded.")

    
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    questions, answers = load_MATH(args.MATH_data_path)

    with open(args.prompt_template_path, "r") as f:
        prompt_template = f.read()
    prompts = [prompt_template.format(question=question) for question in questions]

    print("Evaluating model...")
    results = evaluate_vllm(
        evaluate_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        eval_sampling_params=eval_sampling_params,
        save_dir=args.save_dir
    )
    metrics = compute_metrics(results, args.save_dir)
    print("Evaluation complete.")