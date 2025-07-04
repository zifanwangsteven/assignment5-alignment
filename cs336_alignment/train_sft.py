import torch
from torch.optim import AdamW
import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader, Dataset
from cs336_alignment.helpers import (
    tokenize_prompt_and_output, 
    get_response_log_probs,
    masked_normalize,
    log_generations,
    init_vllm,
    load_policy_into_vllm_instance
)
from vllm import SamplingParams
from cs336_alignment.evaluate_math import load_MATH
import wandb
from omegaconf import OmegaConf

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        policy_log_probs: torch.Tensor, shape (batch_size, sequence_length), log probabilities of the policy.
        response_mask: torch.Tensor, shape (batch_size, sequence_length), mask of the response tokens.
        gradient_accumulation_steps: int, number of gradient accumulation steps.
        normalize_constant: int | None, constant to divide by for normalization.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss: torch.Tensor, shape (1,), loss for the microbatch.
        metadata: dict[str, torch.Tensor], metadata for the microbatch.
    """
    # Calculate loss per sample
    loss_per_sample = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1)
    
    # Average over the batch dimension
    loss = loss_per_sample.mean()
    loss /= gradient_accumulation_steps
    loss.backward()

    return loss, {"loss": loss_per_sample}

class MATH_SFT_Dataset(Dataset):
    def __init__(self, file_path: str="data/MATH/sft.jsonl"):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    prompts = [item['prompt'] for item in batch]
    outputs = [item['response'] for item in batch]
    answers = [item['ground_truth'] for item in batch]
    return prompts, outputs, answers


def train_sft(configs):

    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="hiro_xrl",
        # Set the wandb project where this run will be logged.
        project="MATH-SFT",
        # Track hyperparameters and run metadata.
        config=configs,
    )

    # Set seed for reproducibility
    torch.manual_seed(configs.seed)
    
    # Setup device
    train_device = torch.device(configs.train_device)
    print(f"Using training device: {train_device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(train_device)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("Loading datasets...")
    train_dataset = MATH_SFT_Dataset(configs.data_train_path)
    eval_questions, eval_answers = load_MATH(configs.data_eval_path)

    micro_batch_size = configs.batch_size // configs.gradient_accumulation_steps
    train_dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn)
    train_data_size = len(train_dataset)
    with open(configs.prompt_template_path, "r") as f:
        prompt_template = f.read()
    eval_prompts = [prompt_template.format(question=question) for question in eval_questions]
    eval_answers = eval_answers

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=configs.lr)
    num_training_steps = configs.num_epochs * len(train_dataloader)
    
    lr_scheduler_kwargs = OmegaConf.to_container(configs.lr_scheduler_kwargs, resolve=True) if hasattr(configs, 'lr_scheduler_kwargs') else {}

    lr_scheduler = get_scheduler(
        name=configs.lr_scheduler, optimizer=optimizer, num_warmup_steps=0.05*num_training_steps, num_training_steps=num_training_steps, scheduler_specific_kwargs=lr_scheduler_kwargs
    )
    
    # Initialize vLLM for evaluation
    print(f"Initialize vLLM on {configs.eval_device}...")
    llm_eval = init_vllm(model_id=configs.model_path, device=configs.eval_device, seed=configs.seed)

    # Training loop
    print("Starting training...")
    for epoch in range(configs.num_epochs):
        model.train()
        flag = False
        accumulated_loss = 0.0
        for micro_step, (prompts, outputs, answers) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            # Tokenize batch
            tokenized_batch = tokenize_prompt_and_output(prompts, outputs, tokenizer, train_device)
            input_ids = tokenized_batch['input_ids']
            labels = tokenized_batch['labels']
            response_mask = tokenized_batch['response_mask']

            # Forward pass to get log probs
            model_outputs = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = model_outputs['log_probs']

            # Microbatch training step
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=configs.gradient_accumulation_steps,
                normalize_constant = response_mask.sum(dim=-1).float().mean()
            )

            accumulated_loss += loss.item()

            if (micro_step + 1) % configs.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                flag = True

                print(f"Step {epoch*train_data_size + micro_step*micro_batch_size}, Training loss = {accumulated_loss}", flush=True)
                wandb.log({"train/loss": accumulated_loss}, step=epoch*train_data_size + micro_step*micro_batch_size)
                accumulated_loss = 0.0

        if flag == False:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            flag = True
            print(f"Step {epoch*train_data_size + micro_step*micro_batch_size}, Training loss = {accumulated_loss}", flush=True)
            wandb.log({"train/loss": accumulated_loss}, step=epoch*train_data_size + micro_step*micro_batch_size)
            accumulated_loss = 0.0
    
        # Evaluation at the end of each epoch
        print(f"\nRunning evaluation at end of epoch {epoch+1}...")
        model.eval()
        
        # Load the current trained policy into the vLLM instance
        load_policy_into_vllm_instance(model, llm_eval)

        eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)

        eval_results = log_generations(
            model=model,
            tokenizer=tokenizer,
            llm=llm_eval,
            step=epoch*train_data_size + micro_step*micro_batch_size,
            prompts=eval_prompts,
            answers=eval_answers,
            eval_sampling_params=eval_sampling_params,
            train_device=train_device,
            log_dir=configs.log_dir
        )
        
        print(f"Step {epoch*train_data_size + micro_step*micro_batch_size}, Evaluation Accuracy: {eval_results['accuracy']:.4f}, Avg Response Len: {eval_results['avg_response_len']:.4f}")
        wandb.log({"eval/accuracy": eval_results['accuracy'], 
                   "eval/format_accuracy": eval_results['format_accuracy'], 
                   "eval/answer_accuracy": eval_results['answer_accuracy'], 
                   "eval/avg_response_len": eval_results['avg_response_len'], 
                   "eval/avg_correct_len": eval_results['avg_correct_len'], 
                   "eval/avg_incorrect_len": eval_results['avg_incorrect_len']}, 
                   step=epoch*train_data_size + micro_step*micro_batch_size)

        # Save model checkpoint at the end of each epoch
        if configs.save_dir:
            epoch_save_path = os.path.join(configs.save_dir, f'checkpoint_epoch_{epoch+1}')
            print(f"\nSaving model checkpoint to {epoch_save_path}")
            model.save_pretrained(epoch_save_path)
            tokenizer.save_pretrained(epoch_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with Supervised Fine-Tuning from a config file.")
    parser.add_argument('--config', type=str, default="cs336_alignment/configs/sft_config.yaml", help='Path to the YAML config file.')
    args = parser.parse_args()
    configs = OmegaConf.load(args.config)
    
    train_sft(configs)


