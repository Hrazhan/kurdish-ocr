from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InitializationArguments:
    """
    Initilization config, so we can train the model on a different server
    """


    tokenizer_name: Optional[str] = field(
        default="razhan/roberta-base-ckb", metadata={"help": "Tokenizer attached to model."}
    )
    processor_name: Optional[str] = field(
        
    )
    encoder_name: Optional[str] = field(
        default="facebook/deit-base-distilled-patch16-384", metadata={"help": "Encoder attached to model."}
    )
    decoder_name: Optional[str] = field(
        default="razhan/roberta-base-ckb", help="Decoder attached to model."
    )
    model_name: Optional[str] = field(default="razhan/trocr-ckb", metadata={"help": "Name of the created model."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenize, processor and model to the hub."})
