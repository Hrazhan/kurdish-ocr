import logging
import sys
import os
import pandas as pd

import transformers
import evaluate
import datasets
from arguments import TrainingArguments
from dataset import HandWrittenDataset, KTRDataset

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    HfArgumentParser,
    get_scheduler,
    default_data_collator,
    set_seed,
)

logger = logging.getLogger(__name__)


def create_dataloaders(args):
    if not args.handwritten_dataset:
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(args.csv_path)

        train_df, test_df = train_test_split(
            df, test_size=args.test_split, random_state=args.seed
        )
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_dataset = KTRDataset(
            root_dir=args.root_dir,
            df=train_df,
            processor=processor,
            max_target_length=args.max_length,
        )
        eval_dataset = KTRDataset(
            root_dir=args.root_dir,
            df=test_df,
            processor=processor,
            max_target_length=args.max_length,
        )

        return train_dataset, eval_dataset
    else:
        train_dataset = HandWrittenDataset(
            root_dir=args.root_dir,
            train=True,
            processor=processor,
            max_target_length=args.max_length,
            test_split=args.test_split,
        )
        eval_dataset = HandWrittenDataset(
            root_dir=args.root_dir,
            train=False,
            processor=processor,
            max_target_length=args.max_length,
            test_split=args.test_split,
        )
        return train_dataset, eval_dataset


parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args()


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()

# Sanity check
# if the dataset is not handwritten we must have csv_path of the labels
if not args.handwritten_dataset:
    assert args.csv_path is not None, "Please provide csv_path"

if args.seed is not None:
    set_seed(args.seed)

model = VisionEncoderDecoderModel.from_pretrained(args.model_ckpt)
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

processor = TrOCRProcessor.from_pretrained(args.model_ckpt)

# Load dataset and dataloader
train_dataset, eval_dataset = create_dataloaders(args)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        dataloader_num_workers=8,
        warmup_steps=args.num_warmup_steps,
        fp16=True,
        # bf16=True,  # bfloat16 training
        torch_compile=True,  # optimizations
        optim="adamw_torch_fused",  # improved optimizer
        # logging & evaluation strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # report_to="wandb",
        push_to_hub=args.push_to_hub,
        hub_strategy="end",
        hub_model_id=args.model_ckpt,
    ),
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)


checkpoint = None
if args.resume_from_checkpoint is not None:
    checkpoint = args.resume_from_checkpoint
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics

metrics["train_samples"] = len(train_dataset)

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
