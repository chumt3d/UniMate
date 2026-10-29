"""Train a diffusion model on motions for any skeleton topology.

Entry point (usually via ``accelerate launch``, see scripts/)::

    python -m unimate.training.train --config configs/<config>.json

CLI overrides (via tyro): ``--output_dir``, ``--batch_size``, ``--num_workers``,
``--resume``, ``--stats_path``.
"""

import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import tyro
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from unimate.configs.schema import MainConfig
from unimate.dataset.factory import create_train_dataloader
from unimate.models.factory import create_model
from unimate.training.trainer import DiffusionTrainer
from unimate.training.tracker import TrainingTracker
from unimate.training.exact_resume import (
    ExactDataLoader, input_identity, training_identity, preserve_rng,
    read_checkpoint, restore_checkpoint, save_checkpoint,
)
from unimate.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(file_name=__file__)

# Tiny3D RETRAIN-PLAN.md, "Checkpointed training and first run" (2026-10-29).
RECOVERY_CHECKPOINT_SECONDS = 120


# ===================================================================
# CLI arguments
# ===================================================================

@dataclass
class TrainingArgs:
    """CLI arguments parsed by tyro (override config values)."""
    config: str
    output_dir: Optional[str] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    resume: Optional[str] = None  # path to checkpoint .pt file to resume from
    # Normalization stats to load instead of computing them from the data
    # (e.g. the dataset_stats.npy of the run being fine-tuned). Default: the
    # resumed run's own file with --resume, otherwise computed.
    stats_path: Optional[str] = None
    visualize: bool = False  # opt in to expensive debug sampling/rendering
    stop_after_seconds: Optional[float] = None  # checkpointed invocation allowance
    stop_after_updates: Optional[int] = None  # completed updates in this invocation


# ===================================================================
# Training loop
# ===================================================================

def train_diffusion(args: TrainingArgs, config: MainConfig,
                    accelerator: Accelerator):
    """End-to-end training: setup, loop, checkpoint, visualize."""

    segment_started_at = time.monotonic()
    if (args.stop_after_seconds is not None
            and (not math.isfinite(args.stop_after_seconds) or args.stop_after_seconds <= 0)):
        raise ValueError("--stop_after_seconds must be positive and finite")
    if (args.stop_after_updates is not None
            and (isinstance(args.stop_after_updates, bool)
                 or not isinstance(args.stop_after_updates, int)
                 or args.stop_after_updates <= 0)):
        raise ValueError("--stop_after_updates must be a positive integer")
    is_main = accelerator.is_main_process
    num_workers = (args.num_workers if args.num_workers is not None
                   else config.training.num_workers)
    if accelerator.num_processes != 1 or num_workers != 0:
        raise ValueError("Exact training checkpoints require one process and num_workers=0")
    checkpoint = read_checkpoint(args.resume) if args.resume else None

    # ---- Output directories ----
    output_dir = args.output_dir or config.experiment.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    debug_dir = os.path.join(output_dir, "debug")
    for d in [checkpoint_dir, log_dir, debug_dir]:
        os.makedirs(d, exist_ok=True)

    if checkpoint is None and any(name.endswith(".pt") for name in os.listdir(checkpoint_dir)):
        raise FileExistsError("New training requires an output directory without checkpoints")

    # ---- Reproducibility ----
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)
        logger.info(
            f"Set random seed to {config.training.seed} (device-specific)")

    # ---- Dataset ----
    logger.info("Creating dataset loader...")
    # `is not None`, not `or`: --num_workers 0 (load in the main process) is
    # a valid override that `or` would silently replace with the config value.
    batch_size = (args.batch_size if args.batch_size is not None
                  else config.training.batch_size)
    # Write the overrides back so the run's config.json records what ran.
    config.training.batch_size = batch_size
    config.training.num_workers = num_workers
    inputs = input_identity(config)
    inputs["execution"] = {"device": str(accelerator.device),
                           "mixed_precision": accelerator.mixed_precision,
                           "accumulation": accelerator.gradient_accumulation_steps}
    if checkpoint is not None:
        for key, value in inputs.items():
            if checkpoint["identity"].get(key) != value:
                raise ValueError(f"Resume input identity mismatch: {key}")
    stats_src = _resolve_train_stats_path(args)
    dataloader = create_train_dataloader(
        dataset_config=config.dataset,
        model_config=config.model,
        balanced=config.training.balanced,
        batch_size=batch_size,
        num_workers=num_workers,
        stats_path=stats_src,
        seed=config.training.seed or 0,
    )

    stats_path = os.path.join(output_dir, "dataset_stats.npy")
    # Resumes must validate the original stats before writing any run artifacts.
    identity_stats = stats_src if checkpoint is not None else stats_path
    if checkpoint is None:
        dataloader.dataset.save_dataset_stats(stats_path)
    identity = training_identity(inputs, config, dataloader.dataset, identity_stats)
    if checkpoint is not None and checkpoint["identity"] != identity:
        raise ValueError("Resume resolved config, split, stats or encoder identity mismatch")
    config.to_json(os.path.join(output_dir, "config.json"))
    if checkpoint is not None and os.path.abspath(stats_src) != os.path.abspath(stats_path):
        # Preserve the exact bytes whose identity was checked (no recomputation).
        import shutil
        shutil.copyfile(stats_src, stats_path)
    dataloader = ExactDataLoader(dataloader, accelerator.device,
                                 config.training.seed or 0)

    # ---- Model ----
    logger.info("Creating model and diffusion...")
    model = create_model(
        dataset_config=config.dataset,
        model_config=config.model,
    )

    # ---- Trainer (wraps model + diffusion/flow for loss & visualization) ----
    trainer = DiffusionTrainer(
        config=config,
        model=model,
        data=dataloader,
        checkpoint_dir=checkpoint_dir,
    )

    # ---- Optimizer (the LR schedule follows the tracker, below) ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
    )

    # The exact loader owns its cursor; Accelerate wraps only compute state.
    # Keep the LR scheduler on the existing optimizer-step horizon.
    model, optimizer = accelerator.prepare(model, optimizer)
    trainer.model = model  # point trainer at the wrapped model
    trainer.initialize_ema(accelerator)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir) if is_main else None

    # ---- Training tracker ----
    # Optimizer steps per epoch: this rank's batches over the accumulation
    # (accelerate also steps on an epoch's last, partial group).
    # An epoch budget wins over num_steps, which the schema always fills in.
    epoch_budget = bool(config.training.num_epochs)
    if epoch_budget and config.training.num_steps and is_main:
        logger.info(f"num_epochs={config.training.num_epochs} sets the budget; "
                    f"num_steps={config.training.num_steps} is ignored.")
    tracker = TrainingTracker(
        max_epochs=config.training.num_epochs if epoch_budget else None,
        max_steps=None if epoch_budget else config.training.num_steps,
        steps_per_epoch=math.ceil(
            len(dataloader) / config.training.gradient_accumulation_steps),
    )
    lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.training.warmup_ratio * tracker.total_steps),
        num_training_steps=tracker.total_steps,
        min_lr_rate=config.training.min_lr_ratio,
        num_cycles=0.5,
    )

    # Restore RNG last, after all constructors and checkpoint state loads.
    if checkpoint is not None:
        restore_checkpoint(checkpoint, accelerator, model, optimizer, lr_scheduler,
                           trainer.ema_model, tracker, dataloader, identity)
        del checkpoint

    if is_main:
        logger.info(f"Training for {tracker.get_duration_str()}")

    # ---- Initial visualization (skip if resuming) ----
    if args.visualize and not args.resume:
        trainer.eval()
        if is_main:
            with preserve_rng():
                trainer.visualize_samples(
                    save_dir=debug_dir,
                    prefix="initial_before_training",
                    num_samples=config.sampling.num_samples,
                    cfg_scale=config.sampling.cfg_scale,
                )
            torch.cuda.empty_cache()

    trainer.train()

    # ---- Main training loop ----
    pbar = tqdm(
        range(tracker.total_steps),
        desc=tracker.get_progress_desc(),
        initial=tracker.current_step,
        disable=not is_main,
    )
    grad_norm = None
    last_checkpoint_at = time.monotonic()
    stop_requested = False
    segment_updates = 0

    while not tracker.training_finished() and not stop_requested:
        pbar.set_description(tracker.get_progress_desc())

        # A resumed epoch retains its saved order and next batch.
        dataloader.set_epoch(tracker.current_epoch)

        for batch in dataloader:
            with accelerator.accumulate(model):
                # Forward pass (mixed precision)
                with accelerator.autocast():
                    loss, loss_dict = trainer.compute_loss(batch)

                # NaN guard. The decision is reduced across ranks first, so
                # every rank takes the same branch — a rank-local skip would
                # desync DDP. A bad micro-batch is dropped by skipping its
                # backward: earlier micro-batches' gradients survive and are
                # all-reduced by the group's final backward as usual. Only
                # when the bad batch IS that final one is there no all-reduce,
                # and then the group's (rank-local) gradients must be thrown
                # away rather than stepped on. Backward on a detached zero
                # would raise 'does not require grad', so never do that.
                bad_local = (~torch.isfinite(loss)).to(torch.int32)
                skip_batch = accelerator.reduce(bad_local, reduction="max").item() == 1
                if skip_batch and is_main:
                    logger.warning(
                        f"[Global skip] non-finite loss at step {tracker.current_step}"
                    )

                # Backward + optimizer step
                if not skip_batch:
                    accelerator.backward(loss)

                if accelerator.sync_gradients and skip_batch:
                    # No all-reduce ran for this accumulation group.
                    optimizer.zero_grad(set_to_none=True)
                elif accelerator.sync_gradients:
                    if config.training.max_grad_norm is not None:
                        grad_norm = accelerator.clip_grad_norm_(
                            accelerator.unwrap_model(model).parameters(),
                            config.training.max_grad_norm,
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # EMA update
                    if trainer.use_ema and trainer.ema_model is not None:
                        trainer.ema_model.step(
                            accelerator.unwrap_model(model).parameters()
                        )

                    pbar.update(1)
                    tracker.step()
                    segment_updates += 1

                    # --- Per-effective-step side effects ---
                    step = tracker.current_step

                    # Progress bar postfix
                    if is_main:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            lr=f"{lr_scheduler.get_last_lr()[0]:.6f}",
                        )

                    # Periodic logging
                    if step > 0 and step % config.training.log_interval == 0 and is_main:
                        _log_metrics(writer, trainer, lr_scheduler, loss_dict,
                                     grad_norm, config, step)

                    # Recovery checkpoints have a wall-clock cadence independent
                    # of optional debug evaluation. Check only after useful work.
                    final_epoch_batch = (epoch_budget and dataloader.end_of_dataloader
                                         and tracker.current_epoch + 1 == tracker.max_epochs)
                    milestone = (step % config.training.save_interval == 0
                                 or tracker.training_finished() or final_epoch_batch)
                    recovery_due = time.monotonic() - last_checkpoint_at >= RECOVERY_CHECKPOINT_SECONDS
                    stop_requested = (
                        (args.stop_after_seconds is not None
                         and time.monotonic() - segment_started_at >= args.stop_after_seconds)
                        or (args.stop_after_updates is not None
                            and segment_updates >= args.stop_after_updates)
                    )
                    if recovery_due or milestone or stop_requested:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                        save_checkpoint(checkpoint_path, accelerator, model, optimizer,
                                        lr_scheduler, trainer.ema_model, tracker, dataloader, identity)
                        last_checkpoint_at = time.monotonic()
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if args.visualize and milestone and not stop_requested:
                        _visualize_samples(trainer, config, tracker, debug_dir)

            # Sync all GPUs after the very first step
            if tracker.current_step == 1:
                accelerator.wait_for_everyone()

            if tracker.training_finished() or stop_requested:
                break

        if not tracker.training_finished() and not stop_requested:
            tracker.next_epoch()

    # ---- Cleanup ----
    if is_main:
        writer.close()
        pbar.close()

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training segment stopped after checkpoint." if stop_requested
                else "Training completed.")


# ===================================================================
# Helper functions
# ===================================================================

def _log_metrics(writer, trainer, lr_scheduler, loss_dict, grad_norm,
                 config, step):
    """Write training scalars to TensorBoard."""
    for key, value in loss_dict.items():
        writer.add_scalar(f"train/{key}", value, step)

    if (trainer.use_ema and trainer.ema_model is not None
            and trainer.ema_model.cur_decay_value is not None):
        writer.add_scalar("train/ema_decay",
                          trainer.ema_model.cur_decay_value, step)

    writer.add_scalar("train/learning_rate",
                      lr_scheduler.get_last_lr()[0], step)

    if config.training.max_grad_norm is not None and grad_norm is not None:
        writer.add_scalar("train/grad_norm", grad_norm, step)


def _visualize_samples(trainer, config, tracker, debug_dir):
    """Optional debug evaluation cannot advance the training RNG stream."""
    try:
        with preserve_rng():
            trainer.visualize_samples(
                save_dir=debug_dir,
                prefix=f"step_{tracker.current_step}",
                num_samples=config.sampling.num_samples,
                cfg_scale=config.sampling.cfg_scale,
            )
    except Exception:
        logger.exception(f"Visualization failed at step {tracker.current_step}; checkpoint saved")
    torch.cuda.empty_cache()


def _resumed_run_dir(resume_path):
    """``<run>`` for ``<run>/checkpoints/<ckpt>.pt`` (the layout training
    writes), else the checkpoint's own directory."""
    ckpt_dir = os.path.dirname(os.path.abspath(resume_path))
    return (os.path.dirname(ckpt_dir)
            if os.path.basename(ckpt_dir) == "checkpoints" else ckpt_dir)


def _resolve_train_stats_path(args):
    """Stats file the run must reuse, or ``None`` to compute from the data.

    ``--stats_path`` wins. With ``--resume``, the checkpoint's run directory
    supplies the stats the weights were trained with: ``<run>`` for
    ``<run>/checkpoints/<ckpt>.pt`` (the layout training writes), else the
    checkpoint's own directory. Recomputing them would shift the
    normalization whenever the data changed in between.
    """
    if args.stats_path:
        if not os.path.isfile(args.stats_path):
            raise FileNotFoundError(f"--stats_path {args.stats_path} does not exist")
        return args.stats_path
    if args.resume:
        candidate = os.path.join(_resumed_run_dir(args.resume), "dataset_stats.npy")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Exact resume requires saved stats: {candidate}")
    return None



# ===================================================================
# Entry point
# ===================================================================

def main(args: TrainingArgs):
    """Parse config and launch training."""
    config = MainConfig.from_json(args.config)

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=False,
            dispatch_batches=False,
        ),
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        logger.info("Training Pipeline")
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Mixed Precision: {accelerator.mixed_precision}")
        logger.info(f"  Gradient Accumulation Steps: {accelerator.gradient_accumulation_steps}")

    train_diffusion(args, config, accelerator)


if __name__ == "__main__":
    args = tyro.cli(TrainingArgs)
    main(args)
