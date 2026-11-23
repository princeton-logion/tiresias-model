import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import transformers
import torch
import torch.nn as nn
import click
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, TrainerCallback
from ..src.tiresias import (
    TiresiasConfig,
    TiresiasForMaskedLM,
    DataCollatorForMaskedLM,
    DatasetForMaskedLM,
)


"""
Train helpers
"""
class EMACallback(TrainerCallback):
    """
    Trainer callback for EMA
    
    - creates EMA "shadow" during training
    - switch in EMA weights for eval
    - switch back to train weights post-eval
    """

    def __init__(self, model: nn.Module, decay: float = 0.9, update_every: int = 100):
        self.decay = decay
        self.update_every = update_every
        self.shadow_params = {
            name: param.data.clone().detach().cpu()
            for name, param in model.named_parameters()
        }
        self._backup_params = {}

    def _update_ema(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    current = param.data.detach().cpu()
                    self.shadow_params[name].mul_(self.decay).add_(
                        current, alpha=1.0 - self.decay
                    )

    def _swap_to_ema(self, model: nn.Module):
        self._backup_params = {}
        device = next(model.parameters()).device
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                self._backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name].to(device))

    def _swap_from_ema(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self._backup_params:
                param.data.copy_(self._backup_params[name])
        self._backup_params = {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and state.global_step % self.update_every == 0:
            self._update_ema(model)

    def load_shadow(self, path: str):
        """Restore the EMA shadow saved by a previous run (for checkpoint resume)."""
        self.shadow_params = torch.load(path, map_location="cpu")

    def on_save(self, args, state, control, model=None, **kwargs):
        # store shadow inside chkpt dir to keep paried with chkpt and reload on --resume-from-checkpoint
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        target_dir = ckpt_dir if os.path.isdir(ckpt_dir) else args.output_dir
        torch.save(self.shadow_params, os.path.join(target_dir, "ema_shadow.pt"))


class MaskingScheduleCallback(TrainerCallback):
    """
    Callback for dynamic mask schedule
    """

    def __init__(self, collator, log_interval: int = 100):
        self.collator = collator
        self.log_interval = log_interval
        self.schedule_history = []

    def on_train_begin(self, args, state, control, **kwargs):
        # schedule reaches final_mlm_prob exactly at training end
        if state.max_steps:
            self.collator.total_training_steps = state.max_steps

    def on_step_begin(self, args, state, control, **kwargs):
        self.collator.set_step(state.global_step)

        if state.global_step % self.log_interval == 0:
            info = self.collator.get_schedule_info()
            self.schedule_history.append(info)
            print(f"\nStep: {state.global_step}/{info['total_steps']} ({info['progress']:.1%})"
                  f"\nMLM rate: {info['current_mlm_prob']:.3f}")


class TiresiasTrainer(Trainer):
    """
    Custom trainer

    Necessary for:
        - different train + eval collators (due to mask scheduler)
        - EMA weight swap at eval
    """

    def __init__(self, *args, eval_data_collator=None, ema_callback=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator
        self.ema_callback = ema_callback

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> dict:
        if self.ema_callback is not None:
            self.ema_callback._swap_to_ema(self.model)
            click.echo(f"Swapped EMA weights for eval")

        result = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        if self.ema_callback is not None:
            self.ema_callback._swap_from_ema(self.model)
            click.echo(f"Eval complete. Restored training weights.")

        return result

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_data_collator if self.eval_data_collator is not None else self.data_collator
        from torch.utils.data import DataLoader
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )


def eval_plot(train_losses, train_steps, eval_losses, eval_loss_steps, data_filename, output_dir):
    """
    Plot train-eval losses for post-training evaluation
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Train Loss', marker='x', color='red')
    plt.plot(eval_loss_steps, eval_losses, label='Eval Loss', marker='o', color='blue')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/loss_plot_{data_filename}.png')
    plt.close()


"""
Train
"""
@click.command()
@click.option("--dataset", required=True, type=str, help="Path to dataset")
@click.option("--vocab", required=True, type=str, help="Path to vocab")
@click.option("--output", default="./output", help="Output dir")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--epochs", default=30, type=int, help="Number of training epochs")
@click.option("--batch", default=32, type=int, help="Batch size")
@click.option("--lr", default=2e-4, type=float, help="Learning rate")
@click.option("--hidden_size", default=768, type=int, help="Number of hidden layers")
@click.option("--downsample-factor", default=4, type=int, help="GBST downsampling factor")
@click.option("--lambda-p", default=6.0, type=float, help="Poisson distribution mean span length")
@click.option("--max-span-length", default=10, type=int, help="Max span length in characters")
@click.option("--mlm-prob", default=0.15, type=float, help="MLM probability")
@click.option("--save-strategy", default="steps", type=click.Choice(["no", "epoch", "steps"]))
@click.option("--eval-strategy", default="steps", type=click.Choice(["no", "epoch", "steps"]))
@click.option("--eval-steps", default=1600, type=int, help="Step-wise intervals to eval, log, and save model")
@click.option("--resume-from-checkpoint", default=None, help="Checkpoint path from which to resume")
@click.option("--excluded-token-ids", type=int, multiple=True, default=(0, 1, 2, 3, 4, 5), help="Character IDs excluded from token replacement and masking (default: [PAD]=0, [MASK]=1, [CLS]=2, [SEP]=3, [UNK]=4, \"\\n\"=5)")
@click.option("--mlm-schedule-type", default="cosine",
              type=click.Choice(["constant", "linear", "cosine"]))
@click.option("--initial-mlm-prob", default=0.40, type=float, help="Initial MLM probability")
@click.option("--final-mlm-prob", default=0.15, type=float, help="Final MLM probability")
@click.option("--ema-decay", default=0.9, type=float,
              help="EMA decay factor")
@click.option("--ema-update-every", default=100, type=int,
              help="Apply EMA update every N steps")
@click.option("--weight-decay", default=0.1, type=float,
              help="Weight decay")
@click.option("--warmup-steps", default=4000, type=int,
              help="Learning-rate warmup steps")
@click.option("--position-buckets", default=256, type=int,
              help="Number of log-spaced position buckets per direction")
@click.option("--max-relative-positions", default=2048, type=int,
              help="Max relative distance for position bucketing. Should cover longest expected molecule sequence length")
@click.option("--share-att-key/--no-share-att-key", default=False,
              help="Share content/position projection matrices in attention")
@click.option("--num-final-char-layers", default=3, type=int,
              help="Number of layers in the final character-level deep encoder")
@click.option("--char-position-buckets", default=128, type=int,
              help="Log-spaced position buckets for final character-level deep encoder")
@click.option("--char-max-relative-positions", default=512, type=int,
              help="Max relative distance for final character-level deep encoder")
@click.option("--bf16/--no-bf16", default=True, help="Enable/disable bf16 (default: True)")
def train(dataset, vocab, output, seed, epochs, batch, lr,
         hidden_size, downsample_factor, lambda_p, max_span_length, mlm_prob,
         save_strategy, eval_strategy, eval_steps, resume_from_checkpoint,
         excluded_token_ids, mlm_schedule_type, initial_mlm_prob, final_mlm_prob,
         ema_decay, ema_update_every, weight_decay, warmup_steps,
         position_buckets, max_relative_positions, share_att_key,
         num_final_char_layers, char_position_buckets, char_max_relative_positions,
         bf16):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    with open(vocab, "r", encoding="utf-8") as f:
        char_stoi = json.load(f)
    pad_id = char_stoi["[PAD]"]
    mask_id = char_stoi["[MASK]"]
    vocab_size = len(char_stoi)
    click.echo(f"Vocab: {vocab_size} chars, PAD={pad_id}, MASK={mask_id}")

    datasets = load_from_disk(dataset)
    train_dataset = DatasetForMaskedLM(datasets["train"])
    val_dataset = DatasetForMaskedLM(datasets["validation"])
    
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        logging_steps=eval_steps if eval_strategy == "steps" else None,
        save_steps=eval_steps if save_strategy == "steps" else None,
        logging_dir=os.path.join(output, 'logs'),
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        bf16=bf16,
    )


    n_gpu = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 0)
    effective_batch_size = batch * n_gpu * training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_dataset) // effective_batch_size
    if num_update_steps_per_epoch == 0:
        raise ValueError(
            f"Cannot train: dataset ({len(train_dataset)} rows) is smaller than "
            f"one effective batch ({effective_batch_size})."
        )
    total_training_steps = num_update_steps_per_epoch * epochs

    click.echo(f"Training: {n_gpu} GPUs, batch={batch}, effective_batch={effective_batch_size}, "
               f"steps/epoch={num_update_steps_per_epoch}, total_steps={total_training_steps}, "
               f"weight_decay={weight_decay}")


    train_data_collator = DataCollatorForMaskedLM(
        seed=seed,
        mask_token_id=mask_id,
        pad_token_id=pad_id,
        vocab_size=vocab_size,
        mlm_probability=mlm_prob,
        lambda_param=lambda_p,
        max_span_length=max_span_length,
        excluded_token_ids=excluded_token_ids,
        mlm_schedule_type=mlm_schedule_type,
        initial_mlm_prob=initial_mlm_prob,
        final_mlm_prob=final_mlm_prob,
        total_training_steps=total_training_steps,
    )
    eval_data_collator = DataCollatorForMaskedLM(
        seed=seed,
        mask_token_id=mask_id,
        pad_token_id=pad_id,
        vocab_size=vocab_size,
        mlm_probability=0.15,
        lambda_param=lambda_p,
        max_span_length=max_span_length,
        excluded_token_ids=excluded_token_ids,
        mlm_schedule_type="constant",
        initial_mlm_prob=0.15,
        final_mlm_prob=0.15,
        total_training_steps=None,
    )
    click.echo(f"Data: {len(train_dataset)} train, {len(val_dataset)} val, "
               f"schedule={mlm_schedule_type}")

    config = TiresiasConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=8192,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
        gbst_downsample_factor=downsample_factor,
        max_span_length=max_span_length,
        position_buckets=position_buckets,
        max_relative_positions=max_relative_positions,
        share_att_key=share_att_key,
        num_final_char_layers=num_final_char_layers,
        char_position_buckets=char_position_buckets,
        char_max_relative_positions=char_max_relative_positions,
    )

    model = TiresiasForMaskedLM(config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # hard failure if it's ever silently off
    assert model.is_gradient_checkpointing

    ema_callback = None
    callbacks = [MaskingScheduleCallback(train_data_collator)]

    if ema_decay > 0.0:
        ema_callback = EMACallback(model=model, decay=ema_decay, update_every=ema_update_every)
        callbacks.append(ema_callback)
        click.echo(f"EMA enabled:\n\tdecay={ema_decay}\n\tupdate_every={ema_update_every}")
    else:
        click.echo("EMA disabled")

    trainer = TiresiasTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=train_data_collator, eval_data_collator=eval_data_collator,
        ema_callback=ema_callback, callbacks=callbacks,
    )

    if resume_from_checkpoint:
        if ema_callback is not None:
            shadow_path = os.path.join(resume_from_checkpoint, "ema_shadow.pt")
            if os.path.isfile(shadow_path):
                ema_callback.load_shadow(shadow_path)
                click.echo(f"Loaded EMA shadow from {shadow_path}")
        click.echo(f"Resume training from {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # overlay EMA shadow belonging belongs to checkpoint so saved model is exactly the one that produced the winning eval_loss
    if ema_callback is not None:
        best_ckpt = trainer.state.best_model_checkpoint
        shadow_path = os.path.join(best_ckpt, "ema_shadow.pt") if best_ckpt else None
        if shadow_path and os.path.isfile(shadow_path):
            ema_callback.load_shadow(shadow_path)
            click.echo(f"Applying EMA shadow from best checkpoint: {best_ckpt}")
        else:
            click.echo("No best-checkpoint shadow found; applying latest in-memory EMA shadow")
        ema_callback._swap_to_ema(model)
    trainer.save_model(output)
    click.echo(f"Model saved to {output}")

    log_history = trainer.state.log_history
    train_steps, train_losses = [], []
    eval_loss_steps, eval_losses = [], []
    for log in log_history:
        if "loss" in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])
            eval_loss_steps.append(log["step"])

    eval_plot(train_losses, train_steps, eval_losses, eval_loss_steps,
              os.path.basename(dataset), output)
    click.echo("Training complete")


if __name__ == "__main__":
    train()
