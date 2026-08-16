"""Stage-2 WGT finetune with small direct energy, force, and stress heads.

The periodic-image d192/L24 WGT trunk is finetuned end to end from the immutable
Stage-1 epoch-5 checkpoint. Energy uses a small site head; force and stress add
small sparse pair branches over the same uncapped periodic edges as the trunk.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from ..stage1._elastic_batch import print_elastic_batch_plan, resolve_elastic_batch


GRAPH_ROOT = os.path.expanduser("~/data/MPtrj/graph_mmap")
MODEL_ROOT = os.path.expanduser("~/models")
SPLIT_SCHEME = "train98_val1_test1_seed0"
TARGET_TRAIN_BATCH_ATOMS = 10_240
VALIDATION_SAMPLE_CAP = 128
MAX_ATOMS = 100
REFERENCE_GPU_MEMORY_GIB = 8.0
REFERENCE_MICROBATCH_ATOMS = 1_700
MICROBATCH_ATOM_GRANULARITY = 100
STAGE1_CHECKPOINT = os.path.join(
    MODEL_ROOT,
    "denoise_mlff",
    "stage1",
    "wgt_d192_l24_attnres_c5_b32k_20260811",
    "mlff=46cd4af;workshop=9afaf97",
    "epoch_000005.pt",
)


def _short_code_version(version: str) -> str:
    if version.startswith("local://"):
        return "local"
    if len(version) >= 12 and all(char in "0123456789abcdefABCDEF" for char in version):
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
        scaled_microbatch_atoms = min(
            TARGET_TRAIN_BATCH_ATOMS,
            max(
                MAX_ATOMS,
                int(
                    REFERENCE_MICROBATCH_ATOMS
                    * batch_plan.gpu_memory_gib
                    / REFERENCE_GPU_MEMORY_GIB
                    / MICROBATCH_ATOM_GRANULARITY
                )
                * MICROBATCH_ATOM_GRANULARITY,
            ),
        )
        batch_plan = replace(
            batch_plan,
            max_train_microbatch_atoms=scaled_microbatch_atoms,
        )
        validation_batch_size = {
            "80gb_plus": 32,
            "40_to_79gb": 16,
            "24_to_39gb": 8,
        }.get(batch_plan.tier, 4)
        validation_probe_batches = VALIDATION_SAMPLE_CAP // validation_batch_size
        print_elastic_batch_plan(batch_plan)
        print(
            "[denoise-mlff][elastic-wgt] "
            f"max_train_microbatch_atoms={batch_plan.max_train_microbatch_atoms}",
            flush=True,
        )
        output_dir = os.path.join(
            MODEL_ROOT,
            "denoise_mlff",
            "stage2",
            fit_config_name,
            _resolve_code_version_str(),
        )

        return dict(
            model=dict(
                name="remote_import.mlff.model.build_wgt_supervised_mlff",
                params=dict(
                    config=dict(
                        n_elements=119,
                        d_model=192,
                        n_heads=3,
                        n_layers=24,
                        readout_trunk_n_layers=0,
                        n_rbf=32,
                        rbf_cutoff=5.0,
                        envelope_basis_size=5,
                        attention_logit_cap=8.0,
                        rpe_mode="film",
                        qk_norm=True,
                        attn_res=True,
                        attn_weight_drop=0.0,
                        attn_key_drop=0.0,
                        pair_film_backend="eager",
                        architecture_version=(
                            "wgt_windowed_graph_transformer_supervised_v1"
                        ),
                        energy_head_hidden=64,
                        energy_head_n_layers=2,
                        # Physical edge subdivision varies with GPU memory, so
                        # stochastic head masks would change the computation.
                        energy_head_dropout=0.0,
                        energy_head_init_output_std=0.0,
                        energy_head_output_init="linear_default",
                        energy_mean_bias=-6.19,
                        energy_head_tap=-1,
                        energy_head_mode="site",
                        force_mode="direct",
                        force_head_mode="site_edge",
                        direct_force_postprocess="zero_net",
                        force_head_hidden=224,
                        force_head_n_layers=2,
                        force_head_dropout=0.0,
                        force_head_init_output_std=1.0e-3,
                        force_head_output_init="scaled",
                        stress_mode="direct",
                        stress_head_mode="site_edge",
                        stress_basis_mode="symmetric",
                        stress_head_hidden=70,
                        stress_head_n_layers=2,
                        stress_head_dropout=0.0,
                        stress_head_init_output_std=0.0,
                        stress_head_output_init="scaled",
                        pair_proj_dim=64,
                        pair_head_hidden=32,
                        pair_head_n_layers=2,
                        pair_head_init_output_std=0.0,
                        pair_head_output_init="scaled",
                        stress_pair_final_bias=False,
                        stress_pair_traceless=True,
                        pretrained_checkpoint=STAGE1_CHECKPOINT,
                        freeze_pretrained=False,
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
                        # Match the prior small-direct-head optimizer
                        # computation: 10,240 atoms per update.
                        warmup_steps=2_000,
                        decay_steps=40_000,
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
                grad_clip=1.0,
                grad_accumulation_steps=1,
                log_every_steps=100,
                save_every_epochs=1,
                keep_last_checkpoints=None,
                amp=True,
                float32_matmul_precision="high",
                compile_model=False,
                compile_mode="default",
                compile_dynamic=True,
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
                max_val_batches=validation_probe_batches,
                validate_every_steps=500,
                random_rotation_train=True,
                loss=dict(
                    energy_weight=0.05,
                    force_weight=0.05,
                    stress_weight=0.005,
                    force_scale_eVA=0.5,
                    stress_scale_eVA3=0.02,
                    energy_scale_eV_per_atom=1.0,
                    energy_loss_type="mse",
                    force_loss_type="mae",
                    stress_loss_type="mae",
                    online_force_normalization=False,
                    online_stress_normalization=False,
                ),
                best_metric_name="forces_mae",
                lower_is_better=True,
            ),
        )
