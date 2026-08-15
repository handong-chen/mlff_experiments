"""Stage-1 pure EqMFT without symmetry data augmentation.

The reciprocal stream uses 32 fixed Gaussian radial basis functions feeding
learned filters and includes every nonzero reciprocal lattice vector with
|G| < 0.8 inverse Angstrom.  It does not truncate to 32 vectors.  Site and
field widths are 32 and 128 channels in each l=0,1,2 sector, with three shared
mean-field tokens.

The logical optimizer batch is fixed at 256,000 atoms on every GPU so runs can
resume across devices without changing optimization semantics.  Only physical
subdivision caps scale with visible memory.  Exact O(3) equivariance and
cell-repetition invariance make random rotations and multiplicity sampling
redundant, so both augmentations are disabled.
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
TARGET_TRAIN_BATCH_ATOMS = 256_000
MAX_ATOMS = 100
MAX_SUPERCELL_MULTIPLICITY = 1
# Physical-subdivision reference for an 8-GiB-class card.  A real first
# optimizer step used 2.712 GiB allocated at 1,200 atoms / 45,792 edges on a
# 7.656-GiB RTX 3060 Ti.  The larger caps are an aggressive throughput choice,
# not a measured worst-case memory bound.  The same continuous scaling gives
# about 41,700 atoms / 1.671M edges at 40 GiB and 83,500 / 3.343M at 80 GiB.
REFERENCE_GPU_MEMORY_GIB = 7.656
REFERENCE_MICROBATCH_ATOMS = 8_000
MICROBATCH_ATOM_GRANULARITY = 100
REFERENCE_MICROBATCH_EDGES = 320_000
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
                name=(
                    "remote_import.mlff.model."
                    "build_eqmft_mean_field_denoiser"
                ),
                params=dict(
                    config=dict(
                        n_elements=119,
                        n_layers=24,
                        n_common_tokens=3,
                        l_max=2,
                        environment_channels=(16, 16, 16),
                        reciprocal_environment_channels=(16, 16, 16),
                        site_channels=(32, 32, 32),
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
                        n_reciprocal_basis=32,
                        reciprocal_g_max_inv_angstrom=0.8,
                        reciprocal_envelope_exponent=6,
                        field_to_site_init_scale=0.0,
                        architecture_version="eqmft_denoiser_meanfield_v2",
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
                        warmup_steps=375,
                        decay_steps=7_500,
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
                log_every_steps=10,
                save_every_epochs=1,
                keep_last_checkpoints=None,
                float32_matmul_precision="high",
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
                train_supercell_multiplicity_probabilities=(),
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
                random_rotation_train=False,
                gates=dict(
                    max_rmse_over_corruption=1.0,
                    min_effective_neighbor_frac=None,
                    min_pred_correction_norm_mean=1.0e-5,
                    enforce=False,
                ),
            ),
        )
