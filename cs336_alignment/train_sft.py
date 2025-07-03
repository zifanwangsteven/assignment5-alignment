import torch
from torch.optim import AdamW
import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
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
    answers = [item['answer'] for item in batch]
    return prompts, outputs, answers

def train_sft(configs):
    # Set seed for reproducibility
    torch.manual_seed(configs.seed)
    
    # Setup device
    train_device = torch.device(configs.train_device)
    print(f"Using device: {train_device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_path,
        torch_dtype=torch.bfloat16
    ).to(train_device)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("Loading datasets...")
    train_dataset = MATH_SFT_Dataset(configs.data_train_path)
    eval_dataset = MATH_SFT_Dataset(configs.data_eval_path)

    micro_batch_size = configs.batch_size // configs.gradient_accumulation_steps
    train_dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn)
    
    # For evaluation, we can use a larger batch size as there are no gradients
    eval_prompts = [item['prompt'] for item in eval_dataset]
    eval_answers = [item['output'] for item in eval_dataset]

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=configs.lr)
    num_training_steps = configs.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=0.1*num_training_steps, num_training_steps=num_training_steps
    )
    
    # Initialize vLLM for evaluation
    print(f"Initialize vLLM on {configs.eval_device}...")
    llm_eval = init_vllm(model_id=configs.model_path, device=configs.eval_device, seed=configs.seed)

    # Training loop
    print("Starting training...")
    global_step = 0
    for epoch in range(configs.num_epochs):
        model.train()
        flag = False
        for i, (prompts, outputs, answers) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            flag = False
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
            )

            if (i + 1) % configs.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                flag = True
                # Log loss
                print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item() * configs.gradient_accumulation_steps}")

        if flag == False:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            flag = True
            # Log loss
            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item() * configs.gradient_accumulation_steps}")
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
            step=global_step,
            prompts=eval_prompts,
            answers=eval_answers,
            eval_sampling_params=eval_sampling_params,
            train_device=train_device,
            save_dir=configs.save_dir
        )
        
        print(f"Epoch {epoch+1} Evaluation Accuracy: {eval_results['accuracy']:.4f}")

        # Save model checkpoint at the end of each epoch
        if configs.save_dir:
            epoch_save_path = os.path.join(configs.save_dir, f'checkpoint_epoch_{epoch+1}')
            print(f"\nSaving model checkpoint to {epoch_save_path}")
            model.save_pretrained(epoch_save_path)
            tokenizer.save_pretrained(epoch_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with Supervised Fine-Tuning")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--data_train_path', type=str, required=True, help='Path to the training data JSONL file.')
    parser.add_argument('--data_eval_path', type=str, required=True, help='Path to the evaluation data JSONL file.')
    parser.add_argument('--train_device', type=str, default="cuda:0", help='Device to train on.')
    parser.add_argument('--eval_device', type=str, default="cuda:1", help='Device to evaluate on.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints and final model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Total batch size for each optimization step.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients over.')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs.')

    configs = parser.parse_args()
    
    train_sft(configs)


