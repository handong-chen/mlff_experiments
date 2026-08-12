"""Resolve reproducible Stage-1 atom budgets from visible GPU memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os


GPU_MEMORY_GIB_ENV = "MLFF_GPU_MEMORY_GIB"


def _memory_tier(memory_gib: float) -> tuple[str, int]:
    """Map reported GiB to nominal GPU memory classes."""

    if memory_gib >= 76.0:
        return "80gb_plus", 0
    if memory_gib >= 38.0:
        return "40_to_79gb", 1
    if memory_gib >= 22.0:
        return "24_to_39gb", 2
    if memory_gib >= 11.0:
        return "12_to_23gb", 3
    return "under_12gb", 4


@dataclass(frozen=True)
class ElasticBatchPlan:
    gpu_memory_gib: float
    tier: str
    full_budget_memory_gib: float
    target_train_batch_atoms: int
    max_train_batch_atoms: int
    max_train_microbatch_atoms: int
    max_train_microbatch_pair_slots: int | None


def resolve_elastic_batch(
    *,
    target_train_batch_atoms: int,
    full_budget_memory_gib: float = 40.0,
    full_budget_pair_slots: int | None = None,
    gpu_memory_gib: float | None = None,
) -> ElasticBatchPlan:
    """Keep one logical batch while adapting its physical subdivisions."""

    target_train_batch_atoms = int(target_train_batch_atoms)
    if target_train_batch_atoms <= 0:
        raise ValueError("target_train_batch_atoms must be > 0")
    full_budget_memory_gib = float(full_budget_memory_gib)
    if not math.isfinite(full_budget_memory_gib) or full_budget_memory_gib <= 0.0:
        raise ValueError("full_budget_memory_gib must be finite and > 0")
    if full_budget_pair_slots is not None:
        full_budget_pair_slots = int(full_budget_pair_slots)
        if full_budget_pair_slots <= 0:
            raise ValueError("full_budget_pair_slots must be > 0")
    if gpu_memory_gib is None:
        override = os.environ.get(GPU_MEMORY_GIB_ENV)
        if override is not None:
            gpu_memory_gib = float(override)
        else:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "elastic batching requires a visible CUDA GPU or "
                    f"{GPU_MEMORY_GIB_ENV}=<GiB>"
                )
            device = torch.cuda.current_device()
            gpu_memory_gib = float(
                torch.cuda.get_device_properties(device).total_memory
            ) / float(1024**3)
    gpu_memory_gib = float(gpu_memory_gib)
    if not math.isfinite(gpu_memory_gib) or gpu_memory_gib <= 0.0:
        raise ValueError("GPU memory must be finite and > 0 GiB")

    tier, memory_rank = _memory_tier(gpu_memory_gib)
    _full_tier, full_budget_rank = _memory_tier(full_budget_memory_gib)
    divisor = 2 ** max(0, memory_rank - full_budget_rank)
    if target_train_batch_atoms % divisor != 0:
        raise ValueError(
            "target_train_batch_atoms must be divisible by the elastic "
            f"microbatch divisor {divisor}"
        )
    return ElasticBatchPlan(
        gpu_memory_gib=gpu_memory_gib,
        tier=tier,
        full_budget_memory_gib=full_budget_memory_gib,
        target_train_batch_atoms=target_train_batch_atoms,
        max_train_batch_atoms=target_train_batch_atoms,
        max_train_microbatch_atoms=target_train_batch_atoms // divisor,
        max_train_microbatch_pair_slots=(
            None
            if full_budget_pair_slots is None
            else full_budget_pair_slots // divisor
        ),
    )


def print_elastic_batch_plan(plan: ElasticBatchPlan) -> None:
    print(
        "[denoise-mlff][elastic-batch] "
        f"gpu_memory_gib={plan.gpu_memory_gib:.3f} "
        f"tier={plan.tier} "
        f"full_budget_memory_gib={plan.full_budget_memory_gib} "
        f"target_train_batch_atoms={plan.target_train_batch_atoms} "
        f"max_train_batch_atoms={plan.max_train_batch_atoms} "
        f"max_train_microbatch_atoms={plan.max_train_microbatch_atoms} "
        f"max_train_microbatch_pair_slots={plan.max_train_microbatch_pair_slots}",
        flush=True,
    )
