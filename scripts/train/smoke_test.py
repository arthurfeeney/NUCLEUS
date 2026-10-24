"""Runs a few steps of a small version of a model's training on a split directory,
without wandb, checkpoints or DataLoader worker processes, so errors in the data, model
or training loop surface in seconds. It then times training steps after a few warmup
steps, and on a GPU reports every operation that makes the CPU wait for the GPU. Any
further arguments override the model's training config and the small model, e.g.
model_cfg.params.processor_blocks=2."""

import argparse
import importlib
import itertools
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from nucleus.testing.reference_attention import neighborhood_attention_for
from nucleus.training.checkpoint import TrainingState
from nucleus.training.logger import MetricLogger
from nucleus.training.loop import LoopSettings, fit
from nucleus.training.run import TrainingModuleBuilder, TrainingSetup, prepare_training, run_training

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
MODELS = ("nucleus2_moe", "nucleus2_moe_divfree_projection", "nucleus2_moe_divfree_curl", "nucleus2_moe_staggered")
# small enough to run on a CPU; a head of 32 channels is the smallest neighborhood
# attention takes, and the patch size is left to the config, as it must divide the grid
SMALL_MODEL = [
    "model_cfg.params.embed_dim=64",
    "model_cfg.params.num_heads=2",
    "model_cfg.params.processor_blocks=1",
    "model_cfg.params.num_experts=2",
    "model_cfg.params.topk=1",
    "model_cfg.params.moe_intermediate_dim=64",
    "data.batch_size=2",
]
# past any step the timing runs, so the timed steps neither log, which reads every
# metric back from the GPU, nor validate
NEVER = 2**62


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path, help="a directory made by boiling-data's scripts/split_dataset.py")
    parser.add_argument("--model", choices=MODELS, default="nucleus2_moe")
    parser.add_argument(
        "--full-model", action="store_true",
        help="train the model size the config sets rather than the small model, e.g. to time it on a GPU",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=3,
        help="untimed steps before the timing, which take torch.compile's compilation and the allocator's growth",
    )
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides of config/train_<model>.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    small_model = [] if args.full_model else SMALL_MODEL
    # the given overrides come last, so they win over the small model
    overrides = ["smoke=true", f"split_dir={args.split_dir.resolve()}", *small_model, *args.overrides]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=f"train_{args.model}", overrides=overrides)
    # the model's own train script, next to this one, so the smoke test trains with
    # the same training module
    build_training_module = importlib.import_module(f"train_{args.model}").build_training_module
    state = run_training(cfg, build_training_module)
    print(f"smoke test passed: {state.step} training steps of {args.model}")
    check_performance(cfg, build_training_module, args.warmup_steps, args.timed_steps)


def check_performance(
    cfg: DictConfig, build_training_module: TrainingModuleBuilder, warmup_steps: int, timed_steps: int
) -> None:
    setup = prepare_training(cfg, build_training_module)
    # one batch, already on the device, repeated, so the timing is the training step
    # alone rather than the data loading too
    batch = next(iter(setup.train_batches)).to(setup.device)
    batches = itertools.repeat(batch)
    with neighborhood_attention_for(setup.device):
        state = _train(setup, batches, warmup_steps)
        seconds_per_step = _time_steps(setup, batches, state, timed_steps)
        syncs = _synchronizing_operations(setup, batches, state)
    samples_per_second = setup.loop_settings.accumulate_grad_batches * batch.input.batch_size / seconds_per_step
    print(
        f"performance: {seconds_per_step * 1e3:.1f} ms per training step, {samples_per_second:.1f} samples per second, "
        f"over {timed_steps} steps on {setup.device} after {warmup_steps} warmup steps"
    )
    _report_syncs(syncs)


def _train(setup: TrainingSetup, batches: Iterator, steps: int, state: Optional[TrainingState] = None) -> TrainingState:
    """Trains steps more optimizer steps from state, without logging or validation."""
    state = state or TrainingState()
    settings = LoopSettings(
        max_steps=state.step + steps,
        accumulate_grad_batches=setup.loop_settings.accumulate_grad_batches,
        val_every_n_steps=NEVER,
        log_every_n_steps=NEVER,
    )
    return fit(setup.module, batches, None, setup.optimization, settings, MetricLogger(), setup.device, state=state)


def _time_steps(setup: TrainingSetup, batches: Iterator, state: TrainingState, steps: int) -> float:
    """Seconds per optimizer step. The GPU runs ahead of the CPU, so it is waited
    for before and after, and the steps are timed as they run on it."""
    _synchronize(setup.device)
    started = time.perf_counter()
    _train(setup, batches, steps, state)
    _synchronize(setup.device)
    return (time.perf_counter() - started) / steps


def _synchronizing_operations(setup: TrainingSetup, batches: Iterator, state: TrainingState) -> List[str]:
    """Where one training step makes the CPU wait for the GPU, as torch's sync debug
    mode warns of it; one entry per synchronization. Empty off a GPU."""
    if setup.device.type != "cuda":
        return []
    with warnings.catch_warnings(record=True) as caught:
        # every synchronization, not only the first at each line
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            _train(setup, batches, 1, state)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return [f"{warning.filename}:{warning.lineno}" for warning in caught if "synchroniz" in str(warning.message)]


def _report_syncs(syncs: List[str]) -> None:
    if not syncs:
        print("synchronizations: none found (they are only detected on a GPU)")
        return
    print(f"synchronizations: {len(syncs)} in one training step, by where they happen:")
    for location, count in Counter(syncs).most_common():
        print(f"  {count:4d}  {location}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
