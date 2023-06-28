import logging
import math
import os
from argparse import Namespace
from pathlib import Path
import json
import pandas as pd

import transformers
import evaluate
import torch
from torch.optim import AdamW
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    HfArgumentParser,
    get_scheduler,
    default_data_collator,
    set_seed,
)
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository, create_repo

import datasets
from torch.utils.data.dataloader import DataLoader
from arguments import TrainingArguments
from dataset import HandWrittenDataset, KTRDataset
from tqdm.auto import tqdm

from transformers.utils import check_min_version, get_full_repo_name

check_min_version("4.31.0.dev0")

logger = get_logger(__name__)


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

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=default_data_collator,
            shuffle=True,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.valid_batch_size,
            collate_fn=default_data_collator,
        )
        return train_dataloader, eval_dataloader
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
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.valid_batch_size)
        return train_dataloader, eval_dataloader


def compute_tflops(elapsed_time, accelerator, args):
    # TFLOPs formula (from Equation 3 in Section 5.1 of https://arxiv.org/pdf/2104.04473.pdf).
    config_model = accelerator.unwrap_model(model).config
    checkpoint_factor = 4 if args.gradient_checkpointing else 3
    batch_size = (
        args.train_batch_size
        * accelerator.state.num_processes
        * args.gradient_accumulation_steps
    )
    factor = (
        24
        * checkpoint_factor
        * batch_size
        * args.max_length
        * (
            config_model.encoder.num_hidden_layers
            + config_model.decoder.num_hidden_layers
        )
        * ((config_model.encoder.hidden_size + config_model.decoder.hidden_size) ** 2)
    )
    flops_per_iteration = factor * (
        1.0
        + (
            args.max_length
            / (
                6.0
                * (config_model.encoder.hidden_size + config_model.decoder.hidden_size)
            )
        )
        + (
            processor.tokenizer.vocab_size
            / (
                16.0
                * (
                    config_model.encoder.num_hidden_layers
                    + config_model.decoder.num_hidden_layers
                )
                * (config_model.encoder.hidden_size + config_model.decoder.hidden_size)
            )
        )
    )
    tflops = flops_per_iteration / (
        elapsed_time * accelerator.state.num_processes * (10**12)
    )
    return tflops


parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args()

# Sanity check
# if the dataset is not handwritten we must have csv_path of the labels
if not args.handwritten_dataset:
    assert args.csv_path is not None, "Please provide csv_path"


# Accelerator
accelerator = Accelerator(
    log_with=["wandb", "tensorboard"], project_dir=f"{args.output_dir}/log"
)
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

args = Namespace(**vars(args), **acc_state)
samples_per_step = accelerator.state.num_processes * args.train_batch_size


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)


# Handle the repository creation
if accelerator.is_main_process:
    if args.push_to_hub:
        if args.model_ckpt is None:
            repo_name = get_full_repo_name(
                Path(args.output_dir).name, token=args.hub_token
            )
        else:
            repo_name = args.model_ckpt
        create_repo(repo_name, exist_ok=True, token=args.hub_token)
        hf_repo = Repository(
            args.output_dir, clone_from=repo_name, token=args.hub_token
        )

        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
accelerator.wait_for_everyone()


model = VisionEncoderDecoderModel.from_pretrained(args.model_ckpt)
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

processor = TrOCRProcessor.from_pretrained(args.model_ckpt)

if accelerator.distributed_type == DistributedType.TPU:
    model.tie_weights()


# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(args)


optimizer = AdamW(model.parameters(), lr=args.learning_rate)

# Use the device given by the `accelerator` object.
device = accelerator.device
model.to(device)

overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps
)

if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True


lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
)
accelerator.register_for_checkpointing(lr_scheduler)


def get_lr():
    return optimizer.param_groups[0]["lr"]


model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps
)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

# Figure out how many steps we should save the Accelerator states
checkpointing_steps = args.checkpointing_steps
if checkpointing_steps is not None and checkpointing_steps.isdigit():
    checkpointing_steps = int(checkpointing_steps)

if args.with_tracking:
    experiment_config = vars(args)
    accelerator.init_trackers("trocr-ckb", experiment_config)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

total_batch_size = (
    args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
)

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataloader)}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(
    f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
)
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")


progress_bar = tqdm(
    range(args.max_train_steps), disable=not accelerator.is_local_main_process
)


# Train model
completed_steps = 0
starting_epoch = 0

if args.resume_from_checkpoint:
    if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        # Sorts folders by date modified, most recent checkpoint is the last
        path = dirs[-1]
    # Extract `epoch_{i}` or `step_{i}`
    training_difference = os.path.splitext(path)[0]

    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch
    else:
        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = (
            int(training_difference.replace("step_", ""))
            * args.gradient_accumulation_steps
        )
        starting_epoch = resume_step // len(train_dataloader)
        resume_step -= starting_epoch * len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps


progress_bar.update(completed_steps)

for epoch in range(starting_epoch, args.num_train_epochs):
    model.train()
    if args.with_tracking:
        total_loss = 0

    if (
        args.resume_from_checkpoint
        and epoch == starting_epoch
        and resume_step is not None
    ):
        # We skip the first `n` batches in the dataloader when resuming from a checkpoint
        active_dataloader = accelerator.skip_first_batches(
            train_dataloader, resume_step
        )
    else:
        active_dataloader = train_dataloader
    for step, batch in enumerate(active_dataloader):
        with accelerator.accumulate(model):
            loss = model(**batch).loss
            if args.with_tracking:
                total_loss += loss.detach().float()
            # log_metrics(step, {"samples": step * samples_per_step, "loss_per_step/train": loss.item()})

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps }"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= args.max_train_steps:
            break

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.inference_mode():
            loss = model(**batch).loss
            outputs = model.generate(**batch["pixel_values"])

        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
        label_ids = batch["labels"]
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        predictions, references = accelerator.gather_for_metrics([pred_str, label_str])
        losses.append(
            accelerator.gather_for_metrics(loss.repeat(args.valid_batch_size))
        )

        cer_metric.add_batch(predictions=predictions, references=references)
        wer_metric.add_batch(predictions=predictions, references=references)

    losses = torch.cat(losses)

    eval_loss = torch.mean(losses)
    cer = cer_metric.compute()
    wer = wer_metric.compute()

    logger.info(
        f"epoch {epoch}: cer: {cer:.3f} wer: {wer:.3f} eval_loss: {eval_loss:.5f}"
    )

    if args.with_tracking:
        accelerator.log(
            {
                "train_loss": f"{total_loss.item() / len(train_dataloader):.5f}",
                "eval_loss": f"{eval_loss.item():.5f}",
                "cer": f"{cer:.5f}",
                "wer": f"{wer:.5f}",
                "epoch": epoch,
                "completed_steps": completed_steps,
            },
            step=completed_steps,
        )

    if args.push_to_hub and epoch < args.num_train_epochs - 1:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            processor.save_pretrained(args.output_dir)
            hf_repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}",
                blocking=False,
                auto_lfs_prune=True,
            )

    if args.checkpointing_steps == "epoch":
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)

if args.with_tracking:
    accelerator.end_training()


if args.output_dir is not None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        processor.save_pretrained(args.output_dir)
        if args.push_to_hub:
            hf_repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"wer": wer, "cer": cer}, f)
