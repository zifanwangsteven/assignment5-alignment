from transformers import PreTrainedTokenizer
import torch

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