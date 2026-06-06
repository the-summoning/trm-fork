"""Evaluation utilities and main evaluation loop."""

import os
from typing import Optional, List, Any
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np

from trm.training.config import PretrainConfig, TrainState
from trm.utils import load_model_class

# Import from original locations (these haven't moved yet)
from trm.data.puzzle_dataset import PuzzleDatasetMetadata


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    """Create evaluator instances from config.
    
    Args:
        config: Training configuration
        eval_metadata: Evaluation dataset metadata
        
    Returns:
        List of evaluator instances
    """
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "trm.evaluation.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    device: str = 'cuda',
    save_traj: bool = False
):
    """Run evaluation on test set.
    
    This function:
    1. Runs model inference on all test batches
    2. Computes basic metrics (accuracy, loss)
    3. Calls task-specific evaluators for advanced metrics
    4. Saves predictions if requested
    
    Args:
        config: Training configuration
        train_state: Current training state
        eval_loader: DataLoader for test set
        eval_metadata: Metadata about test dataset
        evaluators: List of task-specific evaluators
        rank: Current process rank
        world_size: Total processes
        cpu_group: CPU process group for communication
        
    Returns:
        Dictionary of metrics (on rank 0 only)
    """
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        items = 0
        set_name = 'all'

        trajectories = []
       
        for inputs, labels, puzzle_identifiers in eval_loader:
            items += inputs.shape[0]
            processed_batches += 1
            
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {
                "inputs": inputs.to(device),
                "labels": labels.to(device),
                "puzzle_identifiers": puzzle_identifiers.to(device)
            }
            
            with torch.device(device):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0

            if save_traj:
                B, T, S = carry.inner_carry.z_H.shape
                traj = np.zeros((config.arch.halt_max_steps, B, T, S), dtype=np.float32) # type: ignore

            while inference_steps < 2:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )

                if save_traj:
                    traj[inference_steps] = carry.inner_carry.z_H.float().cpu().numpy()

                inference_steps += 1

                if all_finish:
                    break
            
            if save_traj:
                trajectories.append(traj)

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device=device
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        print(items)

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        if save_traj:
            print('Saved trajectories')

            trajectories = np.concatenate(trajectories, axis=1)  # [STEPS, ALL_SAMPLES, T, S]
            trajectories = np.transpose(trajectories, (1, 0, 2, 3)) # [ALL_SAMPLES × STEPS × T × S]

            np.savez(Path(config.checkpoint_path) / f"y_trajectories.npz", y_trajectories=trajectories) # type: ignore

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def save_trajectories(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    traj_dir: Path,
    rank: int,
    device: str = 'cuda',
    N_sup: int = 16,
):
    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)

        carry = None
        processed_batches = 0
        
        items = 0
        set_name = 'all'
       
        for inputs, labels, puzzle_identifiers in eval_loader:
            items += inputs.shape[0]
            processed_batches += 1
            
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {
                "inputs": inputs.to(device),
                "labels": labels.to(device),
                "puzzle_identifiers": puzzle_identifiers.to(device)
            }
            
            with torch.device(device):
                carry = train_state.model.initial_carry(batch)  # type: ignore


            B, T, S = carry.inner_carry.z_H.shape
            traj = np.zeros((N_sup, B, T, S), dtype=np.float32) # type: ignore

            for step in range(N_sup):
                carry, _, _, _, _ = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )

                traj[step] = carry.inner_carry.z_H.float().cpu().numpy()
            
            # trajectories.append(traj)
            np.savez(traj_dir / f'batch_{processed_batches}.npz', y_trajectories=traj)


        print(items)

        # trajectories = np.concatenate(trajectories, axis=1)  # [STEPS, ALL_SAMPLES, T, S]
        # trajectories = np.transpose(trajectories, (1, 0, 2, 3)) # [ALL_SAMPLES × STEPS × T × S]

