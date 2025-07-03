import torch
from typing import List
from transformers import PreTrainedTokenizer


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
