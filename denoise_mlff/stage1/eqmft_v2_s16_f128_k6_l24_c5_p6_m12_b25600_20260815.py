"""Stage-1 EqMFT v2 with six L=2 mean-field tokens and multiplicity training.

Every site carries 16 channels and every field token carries 128 channels in
each of the l=0, 1, and 2 sectors.  Four routing heads retain four site value
channels per head and sector.  The local periodic environment uses the fixed
p=6 C2 envelope at 5 A.  Each logical batch samples total supercell
multiplicity M with probabilities P(M=1,2)=(0.7, 0.3).
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
TARGET_TRAIN_BATCH_ATOMS = 25_600
MAX_ATOMS = 100
MAX_SUPERCELL_MULTIPLICITY = 2
REFERENCE_GPU_MEMORY_GIB = 7.656
REFERENCE_MICROBATCH_ATOMS = 500
MICROBATCH_ATOM_GRANULARITY = 100
REFERENCE_MICROBATCH_EDGES = 18_000
MICROBATCH_EDGE_GRANULARITY = 1_000


def _short_code_version(version: str) -> str:
    if version.startswith("local://"):
        return "local"
    if len(version) >= 12 and all(
        char in "0123456789abcdefABCDEF" for char in version
    ):
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
            full_budget_memory_gib=80.0,
        )
        max_physical_graph_atoms = MAX_ATOMS * MAX_SUPERCELL_MULTIPLICITY
        scaled_microbatch_atoms = min(
            TARGET_TRAIN_BATCH_ATOMS,
            max(
                max_physical_graph_atoms,
                int(
                    REFERENCE_MICROBATCH_ATOMS
                    * batch_plan.gpu_memory_gib
                    / REFERENCE_GPU_MEMORY_GIB
                    / MICROBATCH_ATOM_GRANULARITY
                )
                * MICROBATCH_ATOM_GRANULARITY,
            ),
        )
        scaled_microbatch_edges = max(
            MICROBATCH_EDGE_GRANULARITY,
            int(
                REFERENCE_MICROBATCH_EDGES
                * batch_plan.gpu_memory_gib
                / REFERENCE_GPU_MEMORY_GIB
                / MICROBATCH_EDGE_GRANULARITY
            )
            * MICROBATCH_EDGE_GRANULARITY,
        )
        batch_plan = replace(
            batch_plan,
            max_train_microbatch_atoms=scaled_microbatch_atoms,
        )
        print_elastic_batch_plan(batch_plan)
        print(
            "[denoise-mlff][elastic-eqmft] "
            f"max_train_microbatch_edges={scaled_microbatch_edges}",
            flush=True,
        )
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
                name="remote_import.mlff.model.build_eqmft_denoiser",
                params=dict(
                    config=dict(
                        n_elements=119,
                        n_layers=24,
                        n_common_tokens=6,
                        l_max=2,
                        environment_channels=(16, 16, 16),
                        site_channels=(16, 16, 16),
                        field_channels=(128, 128, 128),
                        qk_channels=(8, 8, 8),
                        n_heads=4,
                        atom_embedding_dim=32,
                        n_rbf=32,
                        radial_hidden_dim=256,
                        invariant_hidden_dim=256,
                        attn_res_key_dim=64,
                        normalization_eps=1.0e-8,
                        cutoff_angstrom=5.0,
                        envelope_exponent=6,
                        local_interaction_init_scale=0.1,
                        field_to_site_init_scale=0.0,
                        architecture_version="eqmft_denoiser_cg_meanfield_v2",
                        position_readout_hidden_dim=128,
                        position_readout_n_layers=2,
                        position_readout_init_scale=1.0e-3,
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
                max_train_microbatch_pair_slots=None,
                max_train_microbatch_edges=scaled_microbatch_edges,
                elastic_batch_memory_gib=batch_plan.gpu_memory_gib,
                elastic_batch_tier=batch_plan.tier,
                train_batch_bucket_size=None,
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
                    min_effective_neighbor_frac=None,
                    min_pred_correction_norm_mean=1.0e-5,
                    enforce=False,
                ),
            ),
        )
