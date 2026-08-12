"""Conservative L12 Transformer finetune with an 11,200-atom global batch.

The pretrained dense Transformer trunk is finetuned end to end from Stage-1
epoch 20. Forces and stress are exact derivatives of the single scalar energy
surface. The 80-GiB physical cap targets 5,600 atoms and 560,000 padded pair
slots, with 100 atoms of whole-structure packing slack. This aims for roughly
one physical batch per rank on two 80-GiB GPUs and two physical batches on one
GPU while preserving the same global optimizer batch.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from ..stage1._elastic_batch import print_elastic_batch_plan, resolve_elastic_batch


GRAPH_ROOT = os.path.expanduser("~/data/MPtrj/graph_mmap")
MODEL_ROOT = os.path.expanduser("~/models")
SPLIT_SCHEME = "train98_val1_test1_seed0"
TARGET_TRAIN_BATCH_ATOMS = 11_200
VALIDATION_SAMPLE_CAP = 128
MAX_ATOMS = 100
FULL_BUDGET_MICROBATCH_ATOMS = 5_700
FULL_BUDGET_PAIR_SLOTS = 560_000
FORTY_TO_79_GIB_MICROBATCH_ATOMS = 2_900
FORTY_TO_79_GIB_MICROBATCH_PAIR_SLOTS = 240_000
TWENTY_FOUR_TO_39_GIB_MICROBATCH_ATOMS = 1_500
TWELVE_TO_23_GIB_MICROBATCH_ATOMS = 800
SUB_7_GIB_MICROBATCH_ATOMS = 128
SUB_7_GIB_MICROBATCH_PAIR_SLOTS = 12_800
UNDER_12_GIB_MICROBATCH_ATOMS = 256
UNDER_12_GIB_MICROBATCH_PAIR_SLOTS = 25_600
STAGE1_CHECKPOINT = os.path.join(
    MODEL_ROOT,
    "denoise_mlff",
    "stage1",
    "transformer_d384_l12_b9600_20260806",
    "mlff=65de3bf;workshop=9afaf97",
    "epoch_000020.pt",
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
            full_budget_memory_gib=79.0,
            full_budget_pair_slots=FULL_BUDGET_PAIR_SLOTS,
        )
        if batch_plan.tier == "80gb_plus":
            # The prior 3,200/320,000 cap peaks near 42 GiB on an 80-GiB A100;
            # 560,000 pair slots target roughly 73.5 GiB. The 5,700 atom cap
            # includes slack so a balanced 5,600-atom rank shard is not split
            # solely at a whole-structure boundary.
            batch_plan = replace(
                batch_plan,
                max_train_microbatch_atoms=FULL_BUDGET_MICROBATCH_ATOMS,
            )
        elif batch_plan.tier == "40_to_79gb":
            # Keep a similar absolute cushion on 40-GiB GPUs instead of using
            # the raw 280,000 pair-slot half of the 80-GiB cap. The atom cap
            # likewise includes boundary slack above the 2,800-atom target.
            batch_plan = replace(
                batch_plan,
                max_train_microbatch_atoms=FORTY_TO_79_GIB_MICROBATCH_ATOMS,
                max_train_microbatch_pair_slots=(
                    FORTY_TO_79_GIB_MICROBATCH_PAIR_SLOTS
                ),
            )
        elif batch_plan.tier == "24_to_39gb":
            # Scale atoms from the 5,600 physical target, not the 11,200
            # logical batch; retain 100 atoms of structure-boundary slack.
            batch_plan = replace(
                batch_plan,
                max_train_microbatch_atoms=(
                    TWENTY_FOUR_TO_39_GIB_MICROBATCH_ATOMS
                ),
            )
        elif batch_plan.tier == "12_to_23gb":
            batch_plan = replace(
                batch_plan,
                max_train_microbatch_atoms=TWELVE_TO_23_GIB_MICROBATCH_ATOMS,
            )
        elif batch_plan.gpu_memory_gib < 7.0:
            batch_plan = replace(
                batch_plan,
                max_train_microbatch_atoms=SUB_7_GIB_MICROBATCH_ATOMS,
                max_train_microbatch_pair_slots=SUB_7_GIB_MICROBATCH_PAIR_SLOTS,
            )
        elif batch_plan.gpu_memory_gib < 11.0:
            batch_plan = replace(
                batch_plan,
                max_train_microbatch_atoms=UNDER_12_GIB_MICROBATCH_ATOMS,
                max_train_microbatch_pair_slots=(
                    UNDER_12_GIB_MICROBATCH_PAIR_SLOTS
                ),
            )
        validation_batch_size = {
            "80gb_plus": 32,
            "40_to_79gb": 16,
            "24_to_39gb": 8,
        }.get(batch_plan.tier, 4)
        validation_probe_batches = VALIDATION_SAMPLE_CAP // validation_batch_size
        print_elastic_batch_plan(batch_plan)
        output_dir = os.path.join(
            MODEL_ROOT,
            "denoise_mlff",
            "stage2",
            fit_config_name,
            _resolve_code_version_str(),
        )

        return dict(
            model=dict(
                name="remote_import.mlff.model.build_transformer_supervised_mlff",
                params=dict(
                    config=dict(
                        n_elements=119,
                        d_model=384,
                        n_heads=6,
                        n_layers=12,
                        readout_trunk_n_layers=0,
                        n_rbf=32,
                        rbf_cutoff=10.0,
                        rpe_mode="film",
                        qk_norm=True,
                        attn_res=True,
                        attn_weight_drop=0.0,
                        attn_key_drop=0.0,
                        pair_film_backend="triton",
                        architecture_version="transformer_filmzero_rpe_v2",
                        energy_head_hidden=64,
                        energy_head_n_layers=2,
                        energy_head_dropout=0.0,
                        energy_head_init_output_std=0.0,
                        energy_head_output_init="linear_default",
                        energy_mean_bias=-6.19,
                        energy_head_tap=-1,
                        energy_head_mode="site_edge",
                        force_mode="conservative",
                        stress_mode="conservative",
                        force_head_mode="site",
                        stress_head_mode="site",
                        stress_basis_mode="symmetric",
                        force_head_init_output_std=1.0e-3,
                        force_head_output_init="scaled",
                        stress_head_init_output_std=0.0,
                        stress_head_output_init="scaled",
                        pair_proj_dim=64,
                        pair_head_hidden=32,
                        pair_head_n_layers=2,
                        pair_head_init_output_std=0.0,
                        pair_head_output_init="scaled",
                        stress_pair_traceless=False,
                        pretrained_checkpoint=STAGE1_CHECKPOINT,
                        freeze_pretrained=False,
                    ),
                ),
            ),
            optimizer=dict(
                name="remote_import.mlff.pipeline.build_optimizer",
                params=dict(
                    name="torch.optim.AdamW",
                    params=dict(lr=1.0e-3, weight_decay=1.0e-3),
                ),
            ),
            lr_scheduler=dict(
                name="remote_import.mlff.pipeline.build_scheduler",
                params=dict(
                    name="exp_warmup",
                    params=dict(
                        # The 11,200-atom budget packs this split into 3,754
                        # steps/epoch versus 13,084 at 3,200 atoms. These values
                        # retain the 0.468-epoch warmup and 9.363-epoch decay.
                        warmup_steps=1_757,
                        decay_steps=35_147,
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
                amp=False,
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
                max_train_microbatch_pair_slots=(
                    batch_plan.max_train_microbatch_pair_slots
                ),
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
                max_val_batches=validation_probe_batches,
                validate_every_steps=500,
                random_rotation_train=True,
                loss=dict(
                    energy_weight=0.05,
                    force_weight=0.05,
                    stress_weight=0.0125,
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
