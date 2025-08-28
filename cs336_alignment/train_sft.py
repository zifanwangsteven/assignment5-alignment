from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from utils import (
    load_prompts,
    load_prompt_template,
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from vllm import LLM, SamplingParams
from unittest.mock import patch
import torch
import json
import wandb
import logging

logger = logging.getLogger()


class MathSFT(Dataset):
    """
    Defines a dataset for easy access
    """

    def __init__(self, filepath, use_top_n: int | None = None):
        self.filepath = filepath
        self.data = []
        with open(self.filepath, "r") as file:
            for line in file:
                self.data.append(json.loads(line))
        if use_top_n:
            self.data = self.data[:use_top_n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_func(batch):
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    ground_truth = [item["ground_truth"] for item in batch]

    return prompts, responses, ground_truth


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy
    """

    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def run_sft_experiment(
    model_path: str = "models/Qwen2.5-Math-1.5B",
    training_data_path: str = "data/MATH/sft.jsonl",
    validation_date_path: str = "data/MATH/validation.jsonl",
    training_device: str = "cuda:0",
    eval_device: str = "cuda:1",
    use_top_n: int | None = None,
    batch_size: int = 16,
    lr: float = 0.001,
    gradient_accumulation_steps: int = 4,
    epochs: int = 5,
    random_seed=40,
):
    wandb.init(
        entity="zifan",
        project="cs336",
        config={
            "use_top_n": use_top_n,
            "lr": lr,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "epochs": epochs,
        },
    )
    wandb.define_metric("train_step")  # the x‐axis for training
    wandb.define_metric("eval_step")  # the x-axis for evaluation

    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")

    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    microbatch_size = batch_size // gradient_accumulation_steps
    train_dataset = MathSFT(training_data_path, use_top_n=use_top_n)
    train_data_loader = DataLoader(
        train_dataset, batch_size=microbatch_size, shuffle=True, collate_fn=collate_func
    )
    train_size = len(train_dataset)

    # Initialize vLLM for evaluation
    logger.info(f"Initialize vLLM on {eval_device}...")
    llm_eval = init_vllm(model_id=model_path, device=eval_device, seed=random_seed)

    prompt_template = load_prompt_template("cs336_alignment/prompts/r1_zero.prompt")
    validation_set = load_prompts(validation_date_path)
    validation_prompts = []
    validation_answers = []
    for prompt in validation_set:
        validation_prompts.append(prompt_template.format(question=prompt["problem"]))
        validation_answers.append(prompt["answer"])

    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "cosine_with_min_lr",
        optimizer,
    )

    for epoch in range(epochs):
        model.train()
        accumulated_loss = 0
        for micro_step, prompt, response, ground_truth in enumerate(train_data_loader):
            flag = False
            input_dict = tokenize_prompt_and_output(
                prompt, response, tokenizer, training_device
            )
            input_ids = input_dict["input_id"]
            labels = input_dict["labels"]
            mask = input_dict["response_mask"]
            model_outputs = get_response_log_probs(
                model, input_ids, labels, training_device
            )

            loss, _ = sft_microbatch_train_step(
                model_outputs,
                mask,
                gradient_accumulation_steps,
                normalize_constant=mask.sum(dim=-1).float().mean(),
            )
            accumulated_loss += loss

            if (micro_step + 1) % gradient_accumulation_steps == 0:
                flag = True
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()
                # Lr update
                lr_scheduler.step()
                optimizer.zero_grad()
                logger.info(
                    "Step %d - loss %f",
                    epoch * train_size + micro_step,
                    accumulated_loss,
                )
                wandb.log(
                    {"train/loss": accumulated_loss},
                    step=epoch * train_size + micro_step,
                )
                accumulated_loss = 0.0

            if not flag:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()
                # Lr update
                lr_scheduler.step()
                optimizer.zero_grad()
                logger.info(
                    "Step %d - loss %f",
                    epoch * train_size + micro_step,
                    accumulated_loss,
                )
                wandb.log(
                    {"train/loss": accumulated_loss},
                    step=epoch * train_size + micro_step,
                )
                accumulated_loss = 0.0
        logger.info("Running evaluation at the end of epoch...")
        load_policy_into_vllm_instance(model, llm_eval)

        eval_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

        eval_results = log_generations(
            name="sft",
            model=model,
            tokenizer=tokenizer,
            llm=llm_eval,
            step=epoch * train_size + micro_step * microbatch_size,
            prompts=validation_prompts,
            answers=validation_answers,
            eval_sampling_params=eval_sampling_params,
            train_device=training_device,
        )

        logger.info(
            f"Step {epoch * train_size + micro_step*microbatch_size}, Evaluation Accuracy: {eval_results['accuracy']:.4f}, Avg Response Len: {eval_results['avg_response_len']:.4f}"
        )
        wandb.log(
            {
                "eval/accuracy": eval_results["accuracy"],
                "eval/format_accuracy": eval_results["format_accuracy"],
                "eval/answer_accuracy": eval_results["answer_accuracy"],
                "eval/avg_response_len": eval_results["avg_response_len"],
                "eval/avg_correct_len": eval_results["avg_correct_len"],
                "eval/avg_incorrect_len": eval_results["avg_incorrect_len"],
            },
            step=epoch * train_size + micro_step * microbatch_size,
        )
