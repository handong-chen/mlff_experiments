"""Stage-1 dense Transformer with independent sigmoid gates and cell multiplicity.

The architecture matches the d192/L24 Full-AttnRes control except that every
pair is gated independently and the message is averaged over valid atoms,
instead of using a score-normalized softmax. This makes atom states intensive
without coupling one edge's gate to the scores of any other edges.
Each logical batch samples total supercell multiplicity M with probabilities
P(M=1,2)=(0.7, 0.3) before atom-budget packing.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from ._elastic_batch import print_elastic_batch_plan, resolve_elastic_batch


GRAPH_ROOT = os.path.expanduser("~/data/MPtrj/graph_mmap")
MODEL_ROOT = os.path.expanduser("~/models")
SOURCE_SPLIT_SCHEME = "train98_val1_test1_seed0"
MIN_ENERGY_SPLIT_SCHEME = (
    "min_energy_frame_rms_q90_uniform_train98_val1_test1_seed0"
)
TRAJECTORY_ID_TRANSFORM = "drop_last_dash_component"
TARGET_TRAIN_BATCH_ATOMS = 12_800
FULL_TIER_MICROBATCH_ATOMS = 7_200
FULL_TIER_MICROBATCH_PAIR_SLOTS = 750_000
MAX_ATOMS = 100
MAX_SUPERCELL_MULTIPLICITY = 2
MIN_TRAIN_MICROBATCH_ATOMS = 900
MIN_TRAIN_MICROBATCH_PAIR_SLOTS = 93_750


def _short_code_version(version: str) -> str:
    if version.startswith("local://"):
        return "local"
    if len(version) >= 12 and all(char in "0123456789abcdefABCDEF" for char in version):
        return version[:7]
    return version


def _resolve_code_version_str() -> str:
    try:
        from remote_import.repo_config import (
            get_code_version_from_env,
            prepare_code_version_str,
        )

        return prepare_code_version_str(
            {
                repo: _short_code_version(version)
                for repo, version in get_code_version_from_env().items()
            }
        )
    except Exception:
        return "DEFAULT"


class ConfigProvider:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        fit_config_name = Path(__file__).stem
        physical_batch_plan = resolve_elastic_batch(
            target_train_batch_atoms=FULL_TIER_MICROBATCH_ATOMS,
            full_budget_memory_gib=79.0,
            # Under AMP, Full AttnRes retains growing history stacks at every
            # sublayer. The deeper model therefore needs lower physical caps
            # even though its width and parameter count are smaller.
            full_budget_pair_slots=FULL_TIER_MICROBATCH_PAIR_SLOTS,
        )
        max_physical_graph_atoms = MAX_ATOMS * MAX_SUPERCELL_MULTIPLICITY
        physical_batch_plan = replace(
            physical_batch_plan,
            max_train_microbatch_atoms=max(
                physical_batch_plan.max_train_microbatch_atoms,
                max_physical_graph_atoms,
                MIN_TRAIN_MICROBATCH_ATOMS,
            ),
            max_train_microbatch_pair_slots=max(
                physical_batch_plan.max_train_microbatch_pair_slots or 0,
                max_physical_graph_atoms**2,
                MIN_TRAIN_MICROBATCH_PAIR_SLOTS,
            ),
        )
        batch_plan = replace(
            physical_batch_plan,
            target_train_batch_atoms=TARGET_TRAIN_BATCH_ATOMS,
            max_train_batch_atoms=TARGET_TRAIN_BATCH_ATOMS,
        )
        print_elastic_batch_plan(batch_plan)
        code_version = _resolve_code_version_str()
        output_dir = os.path.join(
            MODEL_ROOT,
            "denoise_mlff",
            "stage1",
            fit_config_name,
            code_version,
        )
        preprocess_summary = os.path.join(
            GRAPH_ROOT,
            "splits",
            MIN_ENERGY_SPLIT_SCHEME,
            "manifest.json",
        )

        return dict(
            model=dict(
                name="remote_import.mlff.model.build_sigmoid_transformer_denoiser",
                params=dict(
                    config=dict(
                        n_elements=119,
                        d_model=192,
                        n_heads=3,
                        n_layers=24,
                        n_rbf=32,
                        rbf_cutoff=10.0,
                        rpe_mode="film",
                        qk_norm=True,
                        attn_res=True,
                        pair_film_backend="triton",
                        architecture_version=(
                            "transformer_filmzero_rpe_sigmoid_mean_v5"
                        ),
                        position_head_hidden=384,
                        position_head_n_layers=2,
                        position_head_init_output_std=1.0e-3,
                        position_head_output_init="scaled",
                    ),
                ),
            ),
            corruption=dict(
                name="remote_import.mlff.data.build_position_corruptor",
                params=dict(
                    config=dict(
                        sigma_position_max="from_preprocess",
                        t_sampling="uniform",
                        species_corruption_rate_max=0.0,
                        position_target="denoise",
                    ),
                ),
            ),
            optimizer=dict(
                name="remote_import.mlff.pipeline.build_optimizer",
                params=dict(
                    name="torch.optim.AdamW",
                    params=dict(lr=1.0e-3, weight_decay=0.01),
                ),
            ),
            lr_scheduler=dict(
                name="remote_import.mlff.pipeline.build_scheduler",
                params=dict(
                    name="exp_warmup",
                    params=dict(
                        # 12.8k atoms/step gives 3/4 as many steps per epoch
                        # as the 9.6k sibling, preserving the epoch schedule.
                        warmup_steps=3_750,
                        decay_steps=75_000,
                        min_lr_ratio=0.01,
                    ),
                    interval="step",
                ),
            ),
            equilibrium_preprocess=dict(
                graph_root=GRAPH_ROOT,
                source_split_scheme=SOURCE_SPLIT_SCHEME,
                output_split_scheme=MIN_ENERGY_SPLIT_SCHEME,
                trajectory_id_transform=TRAJECTORY_ID_TRANSFORM,
                sigma_percentile=90.0,
                sigma_corruption_percentile=90.0,
                sigma_t_sampling="uniform",
                sigma_calibration_seed=1729,
                progress_every=5_000_000,
            ),
            trainer=dict(
                output_dir=output_dir,
                epochs=30,
                seed=42,
                eval_seed=1729,
                grad_clip=1.0,
                grad_accumulation_steps=1,
                log_every_steps=100,
                save_every_epochs=1,
                keep_last_checkpoints=None,
                compile_model=False,
                compile_mode="default",
                compile_dynamic=True,
            ),
            denoise_pretrain=dict(
                graph_root=None,
                graph_roots=(GRAPH_ROOT,),
                split_scheme=MIN_ENERGY_SPLIT_SCHEME,
                preprocess_summary=preprocess_summary,
                train_split="train",
                val_split="val",
                batch_size=8,
                max_atoms=MAX_ATOMS,
                max_train_batch_atoms=batch_plan.max_train_batch_atoms,
                target_train_batch_atoms=batch_plan.target_train_batch_atoms,
                max_train_microbatch_atoms=batch_plan.max_train_microbatch_atoms,
                max_train_microbatch_pair_slots=(
                    batch_plan.max_train_microbatch_pair_slots
                ),
                elastic_batch_memory_gib=batch_plan.gpu_memory_gib,
                elastic_batch_tier=batch_plan.tier,
                train_batch_bucket_size=1024,
                train_supercell_multiplicity_probabilities=(0.7, 0.3),
                num_workers=4,
                pin_memory=True,
                drop_last_train=True,
                persistent_workers=True,
                prefetch_factor=2,
                shard_cache_size=2,
                load_ids=False,
                shuffle_with_replacement=False,
                samples_per_epoch=None,
                sampling_seed=42,
                max_train_batches=None,
                max_val_batches=None,
                random_rotation_train=True,
                gates=dict(
                    max_rmse_over_corruption=1.0,
                    min_effective_neighbor_frac=0.0,
                    min_pred_correction_norm_mean=1.0e-5,
                    enforce=False,
                ),
            ),
        )
