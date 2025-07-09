from dataclasses import dataclass, field
from typing import Dict

@dataclass
class SFTConfig:
    model_path: str = field(
        metadata={"help": "Path to pretrained model (e.g., models/Qwen2.5-Math-1.5B)"}
    )
    data_train_path: str = field(
        metadata={"help": "Path to training data file (jsonl)"}
    )
    data_eval_path: str = field(
        metadata={"help": "Path to evaluation data file (jsonl)"}
    )
    prompt_template_path: str = field(
        metadata={"help": "Path to prompt template file"}
    )
    train_device: str = field(
        default="cuda:0", metadata={"help": "Device for training"}
    )
    eval_device: str = field(
        default="cuda:1", metadata={"help": "Device for evaluation"}
    )
    save_dir: str = field(
        default="checkpoints", metadata={"help": "Directory to save checkpoints"}
    )
    log_dir: str = field(
        default="logs", metadata={"help": "Directory for logs"}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed"}
    )
    lr_scheduler: str = field(
        default="cosine_with_min_lr", metadata={"help": "Learning rate scheduler name"}
    )
    lr_scheduler_kwargs: Dict[str, float] = field(
        default_factory=lambda: {"min_lr_rate": 0.1},
        metadata={"help": "Arguments for LR scheduler"}
    )
    lr: float = field(
        default=4.0e-5, metadata={"help": "Learning rate"}
    )
    batch_size: int = field(
        default=16, metadata={"help": "Training batch size"}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Gradient accumulation steps"}
    )
    num_epochs: int = field(
        default=5, metadata={"help": "Number of training epochs"}
    )