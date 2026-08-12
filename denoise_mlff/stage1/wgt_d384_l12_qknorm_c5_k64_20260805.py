"""Stage-1 MPtrj denoising pretrain for the Windowed Graph Transformer."""

from __future__ import annotations

import os
from pathlib import Path

from ._elastic_batch import print_elastic_batch_plan, resolve_elastic_batch


GRAPH_ROOT = "/scratch/gpfs/BERNEVIG/hdchen/data/MPtrj/graph_mmap"
MODEL_ROOT = "/scratch/gpfs/BERNEVIG/hdchen/models"
SOURCE_SPLIT_SCHEME = "train98_val1_test1_seed0"
MIN_ENERGY_SPLIT_SCHEME = (
    "min_energy_frame_rms_q90_uniform_train98_val1_test1_seed0"
)
TRAJECTORY_ID_TRANSFORM = "drop_last_dash_component"
TARGET_TRAIN_BATCH_ATOMS = 9_600


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
        batch_plan = resolve_elastic_batch(
            target_train_batch_atoms=TARGET_TRAIN_BATCH_ATOMS,
            full_budget_memory_gib=80,
            full_budget_pair_slots=480_000,
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
                name="remote_import.mlff.model.build_wgt_denoiser",
                params=dict(
                    config=dict(
                        n_elements=119,
                        d_model=384,
                        n_heads=6,
                        n_layers=12,
                        n_rbf=32,
                        rbf_cutoff=5.0,
                        max_neighbors=64,
                        envelope_log_prior_scale=1.0,
                        rpe_mode="film",
                        qk_norm=True,
                        attn_weight_drop=0.0,
                        attn_key_drop=0.0,
                        attn_res=False,
                        architecture_version="wgt_windowed_graph_transformer_v1",
                        position_head_hidden=768,
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
                        warmup_steps=3_333,
                        decay_steps=66_667,
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
                max_atoms=100,
                max_train_batch_atoms=batch_plan.max_train_batch_atoms,
                target_train_batch_atoms=batch_plan.target_train_batch_atoms,
                max_train_microbatch_atoms=batch_plan.max_train_microbatch_atoms,
                max_train_microbatch_pair_slots=(
                    batch_plan.max_train_microbatch_pair_slots
                ),
                elastic_batch_memory_gib=batch_plan.gpu_memory_gib,
                elastic_batch_tier=batch_plan.tier,
                train_batch_bucket_size=None,
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
                    min_effective_neighbor_frac=0.25,
                    min_pred_correction_norm_mean=1.0e-5,
                    enforce=False,
                ),
            ),
        )
