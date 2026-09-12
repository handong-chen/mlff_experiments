"""Compiled-field and history-memory stage-2 sibling; canonical provider unchanged.

Budget estimates include fixed-capacity field padding, not an A100 capacity test.
The logical optimizer batch is 16,000 atoms, with a rescaled step scheduler.

The untied site64 checkpoint path follows the requested naming convention;
checkpoint existence is not asserted by this provider.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from ..stage1._elastic_batch import print_elastic_batch_plan, resolve_elastic_batch


GRAPH_ROOT = os.path.expanduser("~/data/MPtrj/graph_mmap")
MODEL_ROOT = os.path.expanduser("~/models")
SPLIT_SCHEME = "train98_val1_test1_seed0"
TARGET_TRAIN_BATCH_ATOMS = 16_000
VALIDATION_SAMPLE_CAP = 128
MAX_ATOMS = 100
# Compiled-field + history physical atom capacities.
# At 80GB, with at most 100 atoms per graph, an 8,192-atom cap fits a full
# 16,000-atom logical batch in two nearly full physical calls. This capacity
# is a planning estimate, not a measured A100 fit. A one-call 16,000-atom fit
# is unmeasured; shared fields remain 128-wide, so memory does not simply
# halve relative to site128. Other GPU tiers retain their physical caps
# and accumulate the larger logical batch.
MICROBATCH_ATOMS_BY_TIER = {
    "under_12gb": 640,
    "12_to_23gb": 1536,
    "24_to_39gb": 3072,
    "40_to_79gb": 4096,
    "80gb_plus": 8192,
}
STAGE1_CHECKPOINT = os.path.join(
    MODEL_ROOT,
    "denoise_mlff",
    "stage1",
    "eqmft_fourier_v3_sparse_site64_20260906",
    "mlff=1f8e6f9;workshop=9afaf97",
    "epoch_000020.pt",
)
ENERGY_HEAD_INIT_BIAS = -6.19


def _short_code_version(version: str) -> str:
    if version.startswith("local://"):
        return "local"
    if len(version) >= 12 and all(
        char in "0123456789abcdefABCDEF" for char in version
    ):
        return version[:7]
    return version


def _resolve_code_version_str() -> str:
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


class ConfigProvider:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        fit_config_name = Path(__file__).stem
        batch_plan = resolve_elastic_batch(
            target_train_batch_atoms=TARGET_TRAIN_BATCH_ATOMS,
            full_budget_memory_gib=80.0,
        )
        scaled_microbatch_atoms = max(
            MAX_ATOMS,
            min(
                TARGET_TRAIN_BATCH_ATOMS,
                MICROBATCH_ATOMS_BY_TIER[batch_plan.tier],
            ),
        )
        batch_plan = replace(
            batch_plan,
            max_train_microbatch_atoms=scaled_microbatch_atoms,
            max_train_microbatch_pair_slots=None,
        )
        print_elastic_batch_plan(batch_plan)
        validation_batch_size = {
            "80gb_plus": 32,
            "40_to_79gb": 16,
            "24_to_39gb": 8,
        }.get(batch_plan.tier, 4)
        output_dir = os.path.join(
            MODEL_ROOT,
            "denoise_mlff",
            "stage2",
            fit_config_name,
            _resolve_code_version_str(),
        )

        return dict(
            model=dict(
                name=(
                    "remote_import.mlff.model."
                    "build_eqmft_mean_field_supervised_mlff"
                ),
                params=dict(
                    config=dict(
                        n_elements=119,
                        n_layers=24,
                        layer_weight_sharing_block_size=1,
                        n_common_tokens=3,
                        l_max=2,
                        environment_channels=(16, 16, 16),
                        reciprocal_environment_channels=(16, 16, 16),
                        site_channels=(64, 64, 64),
                        field_channels=(128, 128, 128),
                        qk_channels=(8, 8, 8),
                        n_heads=4,
                        atom_embedding_dim=64,
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
                        dynamic_reciprocal_channels=(16, 8, 4),
                        dynamic_reciprocal_refresh_layers=(0, 4, 8, 12, 16, 20),
                        dynamic_local_refresh_layers=(0, 4, 8, 12, 16, 20),
                        dynamic_n_reciprocal_basis=16,
                        dynamic_reciprocal_g_max_inv_angstrom=1.6,
                        dynamic_reciprocal_envelope_exponent=6,
                        dynamic_reciprocal_init_scale=0.0,
                        dynamic_reciprocal_reference_atomic_volume_angstrom3=(
                            15.0
                        ),
                        dynamic_reciprocal_gate_init=0.1,
                        dynamic_reciprocal_gate_max=1.0,
                        architecture_version=(
                            "eqmft_supervised_conservative_meanfield_"
                            "dynamic_fourier_v3"
                        ),
                        energy_head_hidden_dim=128,
                        energy_head_n_layers=2,
                        energy_head_dropout=0.0,
                        energy_head_init_scale=1.0e-3,
                        energy_head_init_bias=ENERGY_HEAD_INIT_BIAS,
                        linear_reference_energies=None,
                        energy_scale=None,
                        force_mode="conservative",
                        stress_mode="conservative",
                        pretrained_checkpoint=STAGE1_CHECKPOINT,
                        pretrained_extra_token_init_scale=None,
                        init_supervised_checkpoint=None,
                    ),
                ),
            ),
            optimizer=dict(
                name="remote_import.mlff.pipeline.build_optimizer",
                params=dict(
                    name="torch.optim.AdamW",
                    params=dict(lr=3.0e-4, weight_decay=1.0e-3),
                ),
            ),
            lr_scheduler=dict(
                name="remote_import.mlff.pipeline.build_scheduler",
                params=dict(
                    name="exp_warmup",
                    params=dict(
                        # Preserve approximate atom exposure: old steps *
                        # 12,000 / 16,000, rounded to the nearest step.
                        warmup_steps=1_153,
                        decay_steps=23_066,
                        min_lr_ratio=0.01,
                    ),
                    interval="step",
                ),
            ),
            trainer=dict(
                output_dir=output_dir,
                epochs=50,
                seed=42,
                eval_seed=1729,
                grad_clip=0.3,
                grad_accumulation_steps=1,
                log_every_steps=100,
                save_every_epochs=1,
                keep_last_checkpoints=None,
                amp=False,
                compile_model=False,
                compile_mode="default",
                compile_dynamic=True,
                eqmft_compile_fields=True,
                eqmft_history=True,
                eqmft_field_atom_capacity=batch_plan.max_train_microbatch_atoms,
                ema_decay=0.0,
            ),
            supervised=dict(
                graph_root=None,
                graph_roots=(GRAPH_ROOT,),
                split_scheme=SPLIT_SCHEME,
                train_split="train",
                val_split="val",
                batch_size=validation_batch_size,
                max_atoms=MAX_ATOMS,
                max_train_batch_atoms=batch_plan.max_train_batch_atoms,
                target_train_batch_atoms=batch_plan.target_train_batch_atoms,
                max_train_microbatch_atoms=batch_plan.max_train_microbatch_atoms,
                max_train_microbatch_pair_slots=None,
                elastic_batch_memory_gib=batch_plan.gpu_memory_gib,
                elastic_batch_tier=batch_plan.tier,
                train_batch_bucket_size=16_384,
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
                max_val_batches=VALIDATION_SAMPLE_CAP // validation_batch_size,
                validate_every_steps=250,
                random_rotation_train=True,
                loss=dict(
                    energy_weight=0.05,
                    force_weight=0.05,
                    stress_weight=0.125,
                    force_scale_eVA=1.0,
                    stress_scale_eVA3=1.0,
                    energy_scale_eV_per_atom=1.0,
                    energy_loss_type="mae",
                    force_loss_type="l2norm",
                    stress_loss_type="mae",
                    stress_loss_decomp="iso_aniso",
                    stress_aniso_loss_weight=0.37,
                    stress_aniso_loss_type="l2norm",
                    online_force_normalization=False,
                    online_stress_normalization=False,
                ),
                best_metric_name="forces_mae",
                lower_is_better=True,
            ),
        )
