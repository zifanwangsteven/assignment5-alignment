from dataclasses import dataclass, field
from typing import Dict

@dataclass
class GRPOConfig:
    model_path: str = field(
        metadata={"help": "Path to pretrained SFT checkpoint (e.g. checkpoint_sft_…)"}
    )
    data_train_path: str = field(metadata={"help": "Training data file (jsonl)"})
    data_eval_path: str = field(metadata={"help": "Evaluation data file (jsonl)"})
    prompt_template_path: str = field(metadata={"help": "Prompt template file"})
    train_device: str = field(default="cuda:0", metadata={"help": "Device for generation/training"})
    eval_device: str = field(default="cuda:1", metadata={"help": "Device for evaluation"})
    log_dir: str = field(default="./logs", metadata={"help": "Where to write WandB/logs"})
    save_dir: str = field(default="./checkpoints", metadata={"help": "Where to save GRPO checkpoints"})

    n_grpo_steps: int = field(default=200, metadata={"help": "Total number of GRPO update steps"})
    train_batch_size: int = field(default=256, metadata={"help": "Batch size for training (on-policy)"})
    gradient_accumulation_steps: int = field(default=128, metadata={"help": "Gradient accumulation steps"})
    rollout_batch_size: int = field(default=256, metadata={"help": "Batch size for rollout generation"})
    group_size: int = field(default=8, metadata={"help": "Group size for reward normalization"})
    n_grpo_iterations: int = field(default=1, metadata={"help": "Iterations per GRPO step (on-policy=1)"})

    lr: float = field(default=1e-6, metadata={"help": "Initial learning rate"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    lr_scheduler: str = field(
        default="cosine_with_min_lr", metadata={"help": "LR scheduler name"}
    )
    lr_scheduler_kwargs: Dict[str, float] = field(
        default_factory=lambda: {"min_lr_rate": 0.1},
        metadata={"help": "Keyword args for LR scheduler"},
    )

    reward_type: str = field(
        default="r1_zero", metadata={"help": "Which reward function to use"}
    )
    normalize_std: bool = field(
        default=True, metadata={"help": "Whether to divide by std in advantage"}
    )
    loss_type: str = field(
        default="grpo_clip", metadata={"help": "Loss variant (e.g. grpo_clip)"}
    )
    cliprange: float = field(
        default=0.2, metadata={"help": "Clipping range for GRPO loss"}
    )
