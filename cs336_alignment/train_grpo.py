import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, HfArgumentParser
from torch.optim import AdamW
from vllm import LLM, SamplingParams
from typing import Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.helpers import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    masked_mean,
    init_vllm,
    load_policy_into_vllm_instance,
    set_seed,
    log_generations
)
import json
from cs336_alignment.evaluate_math import load_MATH_eval
from tqdm import tqdm
import random
import wandb
import os
from cs336_alignment.grpo_config import GRPOConfig
from trl import TrlParser

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


def train_grpo(
   configs: dict
) -> None:
    
    set_seed(configs.seed)
    # Setup device
    train_device = torch.device(configs.train_device)
    print(f"Using training device: {train_device}")
    train_batch_size = configs.train_batch_size
    gradient_accumulation_steps = configs.gradient_accumulation_steps
    rollout_batch_size = configs.rollout_batch_size

    assert configs.train_batch_size % configs.gradient_accumulation_steps == 0, (
        f"train_batch_size must be divisible by gradient_accumulation_steps")
    micro_train_batch_size = configs.train_batch_size // configs.gradient_accumulation_steps
    assert configs.rollout_batch_size % configs.group_size == 0, (
        f"rollout_batch_size must be divisible by group_size")
    n_prompts_per_rollout_batch = configs.rollout_batch_size // configs.group_size
    
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(configs.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_path)

    
    
    print(f"Initializing vLLM model on device: {configs.eval_device}")
    vllm_model = init_vllm(configs.model_path, configs.eval_device, configs.seed)

    
    print("Loading train dataset and eval dataset...")
    train_dataset = []
    with open(configs.data_train_path, "r") as file:
        for line in file:
            train_dataset.append(json.loads(line))

    eval_questions, eval_answers = load_MATH_eval(configs.data_eval_path)
    eval_questions = eval_questions[:1000]
    eval_answers = eval_answers[:1000]
    # Load prompt template
    with open(configs.prompt_template_path, "r") as f:
        prompt_template = f.read()
    eval_prompts = [prompt_template.format(question=question) for question in eval_questions]
    
    num_training_steps = configs.n_grpo_steps * configs.n_grpo_iterations * rollout_batch_size // train_batch_size
    optimizer = AdamW(model.parameters(), lr=configs.lr, weight_decay=0.0, betas=(0.9, 0.95))
    lr_scheduler = get_scheduler(
        name=configs.lr_scheduler, 
        optimizer=optimizer, 
        num_warmup_steps=0.1*num_training_steps, 
        num_training_steps=num_training_steps, 
        scheduler_specific_kwargs=configs.lr_scheduler_kwargs
    )
    
    if configs.reward_type == "r1_zero":
        reward_fn = r1_zero_reward_fn
    elif configs.reward_type == "question_only":
        reward_fn = question_only_reward_fn
    else:
        raise ValueError(f"Invalid reward type: {configs.reward_type}")
    
    off_policy = configs.n_grpo_iterations > 1
    print(f"Starting GRPO training with off-policy={off_policy}...")


    step = 0
    for grpo_step in tqdm(range(configs.n_grpo_steps), desc="GRPO steps"):
        sampling_params = SamplingParams(temperature=1.0, 
                                         top_p=1.0, 
                                         min_tokens=4,
                                         max_tokens=1024, 
                                         stop=["</answer>"], 
                                         include_stop_str_in_output=True, 
                                         n=configs.group_size, 
                                         seed=configs.seed,
                                         )
        rollout_dataset = random.sample(train_dataset, n_prompts_per_rollout_batch)
        rollout_prompts = [prompt_template.format(question=data["problem"]) for data in rollout_dataset]
        rollout_answers = [data["answer"] for data in rollout_dataset]

        rollout_outputs = vllm_model.generate(rollout_prompts, sampling_params=sampling_params)

        prompts=[]
        responses=[]
        repeated_answers=[]
        for prompt, output, answer in zip(rollout_prompts, rollout_outputs, rollout_answers):
            for i in range(configs.group_size):
                prompts.append(prompt)
                responses.append(output.outputs[i].text)
                repeated_answers.append(answer)
        
        tokenized_results = tokenize_prompt_and_output(prompts, responses, tokenizer, train_device)
        input_ids, labels, response_mask = tokenized_results["input_ids"], tokenized_results["labels"], tokenized_results["response_mask"]
        
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=repeated_answers,
            group_size=configs.group_size,
            normalize_std=configs.normalize_std,
            device=train_device
        )

        num_train_steps_per_iteration = rollout_batch_size // train_batch_size
        old_log_probs_train = []
        if off_policy:
            with torch.no_grad():
                for train_step in range(num_train_steps_per_iteration):
                    for train_microstep in range(gradient_accumulation_steps):
                        start_idx = train_step*train_batch_size + train_microstep*micro_train_batch_size
                        end_idx = start_idx + micro_train_batch_size
                        input_ids_micro, labels_micro, response_mask_micro = input_ids[start_idx:end_idx], labels[start_idx:end_idx], response_mask[start_idx:end_idx]
                        log_probs_dict = get_response_log_probs(model, input_ids_micro, labels_micro, return_token_entropy=True)
                        log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                        old_log_probs_train.append(log_probs)
                        assert log_probs.shape[0] == micro_train_batch_size
                old_log_probs_train = torch.cat(old_log_probs_train)

        model.train()
        for grpo_iteration in range(configs.n_grpo_iterations):
            for train_step in range(num_train_steps_per_iteration):
                accumulated_loss = 0.0
                for train_microstep in range(gradient_accumulation_steps):
                    start_idx = train_step*train_batch_size + train_microstep*micro_train_batch_size
                    end_idx = start_idx + micro_train_batch_size
                    input_ids_micro, labels_micro, response_mask_micro = input_ids[start_idx:end_idx], labels[start_idx:end_idx], response_mask[start_idx:end_idx]
                    log_probs_dict = get_response_log_probs(model, input_ids_micro, labels_micro, return_token_entropy=True)
                    policy_log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                    
                    if off_policy:
                        use_old_log_probs = old_log_probs_train[start_idx:end_idx]
                    else:
                        use_old_log_probs = policy_log_probs.detach()
                    
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask_micro,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=configs.loss_type,
                        raw_rewards=raw_rewards[start_idx:end_idx],
                        advantages=advantages[start_idx:end_idx].reshape(-1,1),
                        old_log_probs=use_old_log_probs,
                        cliprange=configs.cliprange
                    )
                    accumulated_loss += loss.item()
                print(f"Step {step}, Training loss = {accumulated_loss}", flush=True)
                step += 1
                # wandb.log({"train/loss": accumulated_loss}, step=step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        if (grpo_step + 1) % 20 == 0:
            model.eval()
            # Load policy into vLLM instance
            load_policy_into_vllm_instance(model, vllm_model)

            eval_sampling_params = SamplingParams(temperature=1.0, 
                                                top_p=1.0, 
                                                min_tokens=4,
                                                max_tokens=1024, 
                                                stop=["</answer>"], 
                                                include_stop_str_in_output=True,
                                                )

            eval_results = log_generations(
                name="grpo",
                model=model,
                tokenizer=tokenizer,
                llm=vllm_model,
                step=step,
                prompts=eval_prompts,
                answers=eval_answers,
                eval_sampling_params=eval_sampling_params,
                train_device=train_device,
                log_dir=configs.log_dir
            )
            
            print(f"Step {step}, Evaluation Accuracy: {eval_results['accuracy']:.4f}, Avg Response Len: {eval_results['avg_response_len']:.4f}")
            # wandb.log({"eval/accuracy": eval_results['accuracy'], 
            #            "eval/format_accuracy": eval_results['format_accuracy'], 
            #            "eval/answer_accuracy": eval_results['answer_accuracy'], 
            #            "eval/avg_response_len": eval_results['avg_response_len'], 
            #            "eval/avg_correct_len": eval_results['avg_correct_len'], 
            #            "eval/avg_incorrect_len": eval_results['avg_incorrect_len']}, 
            #            step=step)

            # Save model checkpoint at the end of each epoch
            if configs.save_dir:
                step_save_path = os.path.join(configs.save_dir, f'checkpoint_grpo_step_{grpo_step}')
                print(f"\nSaving model checkpoint to {step_save_path}")
                model.save_pretrained(step_save_path)
                tokenizer.save_pretrained(step_save_path)

if __name__ == '__main__':
    parser = TrlParser(GRPOConfig)
    (configs,) = parser.parse_args_and_config()
    train_grpo(configs) 