from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from utils import load_prompts, load_prompt_template, tokenize_prompt_and_output, get_response_log_probs
from vllm import LLM
from unittest.mock import patch
import torch

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    """
    Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy
    """

    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
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
    use_top_n: int | None = None,
    training_steps: int = 5000
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    prompt_template = load_prompt_template("cs336_alignment/prompts/r1_zero.prompt")
    training_set = load_prompts(training_data_path)
    validation_set = load_prompts(validation_date_path)

    training_prompts = []
    training_answers = []
    for prompt in training_set:
        training_prompts.append(prompt["problem"])
        training_answers.append(prompt["answer"])
    
    if use_top_n:
        training_prompts = training_prompts[:use_top_n]
        training_answers = training_answers[:use_top_n]
    
    input_dict = tokenize_prompt_and_output(
        training_prompts,
        training_answers,
        tokenizer,
        "cuda:0"
    )
    input_ids = input_dict["input_id"]
    labels = input_dict["labels"]
    mask = input_dict["response_mask"]

    for step in training_steps:
        output = get_response_log_probs(model, input_ids, labels=labels)
        










    validation_prompts = []
    validation_answers = []
    for prompt in validation_set:
        validation_prompts.append(prompt_template.format(question=prompt["problem"]))
        validation_answers.append(prompt["answer"])
    




