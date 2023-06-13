from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InitializationArguments:
    """
    Initilization config, so we can train the model on a different server
    """


    tokenizer_name: Optional[str] = field(
        default="razhan/BPE-tokenizer", metadata={"help": "Tokenizer attached to model."}
    )
    processor_name: Optional[str] = field(
        default="microsoft/trocr-base-handwritten", metadata={"help": "Processor attached to model."}
    )
    encoder_name: Optional[str] = field(
        default="facebook/deit-base-distilled-patch16-384", metadata={"help": "Encoder attached to model."}
    )
    decoder_name: Optional[str] = field(
        default="razhan/roberta-base-ckb", metadata={"help":"Decoder attached to model."}
    )
    max_length: Optional[int] = field(default=128, metadata={"help": "Sequence lengths used for training."})
    model_name: Optional[str] = field(default="razhan/trocr-ckb", metadata={"help": "Name of the created model."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenize, processor and model to the hub."})


@dataclass
class TrainingArguments:
    """
    Training Arguments
    """

    model_ckpt: Optional[str] = field(
        default="razhan/trocr-ckb", metadata={"help": "Model name or path of model to be trained."}
    )
    output_dir: Optional[str] = field(
        default="trocr-ckb", metadata={"help": "Save dir where model repo is cloned and models updates are saved to."}
    )
    handwritten_dataset: bool = field(default=False, metadata={"help": "Whether to use handwritten dataset."})
    root_dir: Optional[str] = field(
        default="data", metadata={"help": "Root dir where data is stored."}
    )
    csv_path: Optional[str] = field(
        default="data/metadata.csv", metadata={"help": "Path of metadata file."}
    )
    test_split: Optional[float] = field(default=0.05, metadata={"help": "Test size."})
    train_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for training."})
    valid_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for evaluation."})
    weight_decay: Optional[float] = field(default=0.1, metadata={"help": "Value of weight decay."})
    with_tracking: Optional[bool] = field(default=True, metadata={"help": "Whether to use tracking."})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "Learning rate fo training."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Learning rate."})
    num_warmup_steps: Optional[int] = field(
        default=3000, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "Number of gradient accumulation steps."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "Number of training epochs."})
    max_train_steps: Optional[int] = field(default=None, metadata={"help": "Maximum number of training steps."})
    # max_eval_steps: Optional[int] = field(
    #     default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    # )
    max_length: Optional[int] = field(default=128, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=42, metadata={"help": "Training seed."})
    checkpointing_steps: Optional[str] = field(
        default="epoch", metadata={"help": "Checkpointing steps."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "States path if the training should continue from a checkpoint folder."}
    )
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenize, processor and model to the hub."})
    hub_token: Optional[str] = field(default=None, metadata={"help": "Hub token."})