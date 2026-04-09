from dataclasses import dataclass, field
from typing import Optional
from trl import ScriptArguments
import trl

@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    silence: bool = field(
        default=False,
        metadata={"help": "Whether to silence verification outputs during training."},
    )

    # Dynamic sampling, overlong filtering, and length related arguments
    dynamic_sampling_scale: int = field(
        default=1,
        metadata={"help": "Scale for dynamic sampling. We will multiply the original batch size by this factor and only keep the top samples."},
    )
    overlong_filter_scale: int = field(
        default=1,
        metadata={"help": "Scale for overlong filtering. We will multiply the original generation number by this factor and filter out the samples that are too long."},
    )
    overlong_punishment_threshold: float = field(
        default=1.0,
        metadata={"help": "Threshold for overlong punishment. If the length of a generation is longer than this ratio of the max length, it will be punished."},
    )

    # For model after qwen3
    enable_thinking: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enable thinking mode. If not specified, no parameter will be passed to tokenizer."},
    )

@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'thinking', and'format'"
        },
    )
    quick_eval_dataset: str = field(
        default=None,
        metadata={"help": "Quick evaluation dataset"},
    )
    quick_eval_dataset_size: int = field(
        default=320,
        metadata={"help": "Number of samples to use from the quick evaluation dataset"},
    )
    distributed_training: bool = field(
        default=False,
        metadata={"help": "Whether to use distributed training."},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    trainable_layers: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of layer name patterns to keep trainable. All other layers will be frozen. Example: ['model.layers.23', 'lm_head']"},
    )