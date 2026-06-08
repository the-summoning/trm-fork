
import os
from pathlib import Path
import sys
import argparse
import yaml
import copy

from torch.utils.data import DataLoader

from trm.data.balanced_dataset import build_balanced_dataset
from trm.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from trm.evaluation.evaluator import save_trajectories

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
from typing import Any, Dict, Mapping, cast
from contextlib import nullcontext
import torch.backends.cudnn as cudnn
from hydra import initialize, compose
from omegaconf import OmegaConf
from trm.models.ema import EMAHelper
# Import functions and classes from trm library
from trm.training import (
    PretrainConfig,
    init_train_state,
)


# Prefer new TF32 API controls to avoid deprecation warnings and ensure predictable math.
try:
    # Use strict IEEE FP32 by default for matmul to prioritize correctness.
    # Change to 'tf32' if you prefer TF32 acceleration.
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
except Exception:
    # Backend may not be available on CPU-only or some environments.
    pass


def parse_args():
    """Parse CLI arguments for the evaluation runner.

    Returns:
        argparse.Namespace with config path, checkpoint, dataset path, output
        directory, eval outputs to save, batch size override, EMA options,
        eval-only toggle, bf16 toggle, and one-batch mode.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/cfg_pretrain.yaml', help='YAML config file (pydantic fields)')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint file. Local path or HuggingFace format: "username/repo/filename"')
    p.add_argument('--dataset', required=True, help='Path to dataset directory to evaluate (overrides data_paths_test)')
    p.add_argument('--split', required=True, help='Split of dataset')
    p.add_argument('--outdir', default=None, help='Directory to save evaluation preds (overrides checkpoint_path in config)')
    p.add_argument('--eval-save-outputs', nargs='+', default=['inputs','labels','puzzle_identifiers','preds'], help='List of keys to save during evaluation')
    p.add_argument('--global-batch-size', type=int, default=None, help='Global batch size override for evaluation')
    p.add_argument('--max-samples', type=int, default=None, help='Max samples')
    # Defaults: eval-only, bf16, and apply-ema are enabled unless explicitly disabled
    p.add_argument('--apply-ema', action='store_true', default=True, help='Apply EMA weights for evaluation (default: on). Use --no-apply-ema to disable')
    p.add_argument('--ema-shadow', default=None, help='Path to EMA shadow state dict (optional). If provided, it will be loaded into EMAHelper before applying EMA.')
        # repeats/seed-start removed: we evaluate exactly once per invocation
    p.add_argument('--eval-only', action='store_true', default=True, help='Run in eval-only mode (skip optimizer creation). Default: on. Use --no-eval-only to disable')
    p.add_argument('--bf16', action='store_true', default=True, help='Use CUDA autocast with bfloat16 during evaluation (default: on). Use --no-bf16 to disable')
    p.add_argument('--save-trajectories', action='store_true', default=True)
    # Negative toggles for convenience
    p.add_argument('--no-apply-ema', dest='apply_ema', action='store_false', help='Disable EMA application during evaluation')
    p.add_argument('--no-eval-only', dest='eval_only', action='store_false', help='Disable eval-only (will construct optimizer); not recommended')
    p.add_argument('--no-bf16', dest='bf16', action='store_false', help='Disable bfloat16 autocast during evaluation')
    p.add_argument('--device', default='cuda')
    p.add_argument('--k', type=int, default=1, help='K')
    p.add_argument('--traj-dir', type=Path, default=Path('./trajectories'), help='K')
    p.add_argument('--use-balanced-ds', action='store_true', default=False)
    p.add_argument('--max-batches', type=int, default=None) # ?

    return p.parse_args()


def main():
    """Entry point for running evaluation with optional distribution/EMA.

    Steps:
    - Initialize distributed context (if under torchrun)
    - Compose and broadcast config
    - Build dataloader(s), model, and optional EMA copy
    - Run a single evaluation pass with optional bf16 autocast
    - On rank 0, print metrics and per-run Wilson 95% CI for accuracy and exact_accuracy when possible
    """
    args = parse_args()

    # Ensure we skip torch.compile() for evaluation to prevent expensive Inductor compilation
    # and potential long startup times under torchrun. The training code compiles unless this
    # environment variable is present.
    os.environ.setdefault('DISABLE_COMPILE', '1')

    # Distributed init (if running under torchrun)
    RANK = 0
    WORLD_SIZE = 1

    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    # Compose config via Hydra on rank 0 and broadcast

    config_obj = None
    objects = [None]
    if RANK == 0:
    # Derive config directory and base name from args.config
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # Hydra's config_path is relative to the script file location, not cwd
        # Since this script is in scripts/, we need to go up one level to find config/
        config_path = "../config"

    # Compose Hydra config; CLI overrides applied programmatically below
        with initialize(version_base=None, config_path=config_path, job_name="run_eval_only"):
            hydra_cfg = compose(config_name=config_name)

    # Convert to plain dict (resolve interpolations)
        cfg_any = OmegaConf.to_container(hydra_cfg, resolve=True)
        if not isinstance(cfg_any, dict):
            raise RuntimeError('Composed config is not a mapping after OmegaConf.to_container')
        cfg: Dict[str, Any] = dict(cast(Mapping[str, Any], cfg_any))

    # Apply programmatic overrides
        #cfg['data_paths_test'] = [args.dataset]
        cfg['load_checkpoint'] = args.checkpoint
        if args.outdir is not None:
            cfg['checkpoint_path'] = args.outdir
        if args.global_batch_size is not None:
            cfg['global_batch_size'] = args.global_batch_size
        cfg['eval_save_outputs'] = args.eval_save_outputs

    # Print composed config on rank 0
        try:
            print('\nComposed config (after Hydra compose + CLI overrides):')
            print(yaml.safe_dump(cfg, sort_keys=False))
        except Exception:
            print('Warning: failed to pretty-print composed config')

    # Build pydantic PretrainConfig
        config_obj = PretrainConfig(**cfg)
        objects = [config_obj]


    config = objects[0]

    # Ensure config present
    if config is None:
        raise RuntimeError('Failed to load config via broadcast; config is None on this rank')

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)
    # Let cuDNN pick fastest algorithms
    try:
        cudnn.benchmark = True
    except Exception:
        pass

    if args.use_balanced_ds:
        print('Using balanced dataset')

        dataset, metadata = build_balanced_dataset(
            dataset_path=args.dataset, 
            split=args.split, 
            set_name="all", 
            num_examples_per_puzzle=args.k,
            max_samples=args.max_samples
        )
    else:
        print('Using standard PuzzleDataset')

        dataset_config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[args.dataset],
            rank=0,
            num_replicas=1,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=args.batch_size
        )
        
        dataset = PuzzleDataset(dataset_config, split=args.split)
        metadata = dataset.metadata



    eval_loader = DataLoader(
        dataset,
        batch_size=config.global_batch_size,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )

    # Init model & train_state (loads checkpoint on rank 0 inside create_model).
    # Pass is_eval according to CLI flag to skip optimizer construction in evaluation-only runs.
    train_state = init_train_state(config, metadata, rank=RANK, world_size=WORLD_SIZE, is_eval=bool(args.eval_only), device=args.device)

    # Optionally switch to EMA copy if requested by CLI or config
    train_state_eval = train_state
    if args.apply_ema or config.ema:

        if RANK == 0:
            print('Preparing EMA for evaluation...')

        ema_helper = EMAHelper(mu=config.ema_rate)
        # Register model parameters
        ema_helper.register(train_state.model)

        # If user provided an EMA shadow file, load and broadcast it to all ranks
        ema_state = None
        objects = [None]
        if args.ema_shadow is not None:
            if RANK == 0:
                ema_state = torch.load(args.ema_shadow, map_location='cpu')
                objects = [ema_state]


        if objects[0] is not None:
            # Load shadow into helper
            ema_helper.load_state_dict(objects[0])
            if RANK == 0:
                print('Loaded EMA shadow state and applying EMA copy for evaluation.')
            train_state_eval = copy.deepcopy(train_state)
            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
        else:
            # No explicit shadow file provided. If the checkpoint already contains EMA weights (saved by training
            # after swapping to the EMA copy), then load_checkpoint already set those weights when init_train_state ran.
            # We still create a deepcopy for safety to avoid modifying the main train_state model during eval.
            if RANK == 0:
                print('No EMA shadow provided — assuming checkpoint contains EMA weights (if training saved EMA).')
            train_state_eval = copy.deepcopy(train_state)

    # Set checkpoint output directory and ensure it exists
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join('checkpoints', 'eval_run')
    if RANK == 0:
        os.makedirs(config.checkpoint_path, exist_ok=True)

    # deepcopy eval state to avoid side-effects
    ts = copy.deepcopy(train_state_eval)
    ts.model.eval()

    use_cuda = torch.cuda.is_available()
    if args.bf16 and use_cuda:
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    args.traj_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode(), amp_ctx:
        save_trajectories(
            config=config,
            train_state=ts,
            eval_loader=cast(Any, eval_loader),
            rank=RANK,
            device=args.device,
            N_sup=config.arch.halt_max_steps, # type: ignore
            traj_dir=args.traj_dir,
            max_batches=args.max_batches
        )



if __name__ == '__main__':
    main()
