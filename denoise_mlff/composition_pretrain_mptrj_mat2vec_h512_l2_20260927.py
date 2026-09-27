"""Larger frozen-mat2vec composition head on global, disjoint MPtrj targets.

Consumes composition_preprocess_mptrj_global_holdout_20260927, assigned by
existing source membership with test > validation > training priority.
The stoichiometric average feeds a 200 -> 512 -> 512 -> 2 SiLU MLP.
AdamW, batch size 256, and the 100-epoch cosine schedule match the earlier
h128 setup; target aggregation and composition holdouts have changed.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_DIR = os.path.expanduser(
    "~/data/MPtrj/composition_reference/composition_preprocess_mptrj_global_holdout_20260927"
)
MODEL_ROOT = os.path.expanduser("~/models")
EPOCHS = 100


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
        return dict(
            optimizer=dict(
                name="remote_import.mlff.pipeline.build_optimizer",
                params=dict(
                    name="torch.optim.AdamW",
                    params=dict(lr=1.0e-3, weight_decay=1.0e-4),
                ),
            ),
            lr_scheduler=dict(
                name="remote_import.mlff.pipeline.build_scheduler",
                params=dict(
                    name="torch.optim.lr_scheduler.CosineAnnealingLR",
                    params=dict(T_max=EPOCHS, eta_min=1.0e-5),
                    interval="epoch",
                ),
            ),
            trainer=dict(
                output_dir=os.path.join(
                    MODEL_ROOT, "denoise_mlff", "composition_pretrain",
                    fit_config_name, _resolve_code_version_str(),
                ),
                epochs=EPOCHS,
                seed=42,
                eval_seed=1729,
                grad_clip=1.0,
                grad_accumulation_steps=1,
                log_every_steps=20,
                save_every_epochs=1,
                keep_last_checkpoints=3,
                amp=False,
                float32_matmul_precision="highest",
                compile_model=False,
            ),
            composition_pretrain=dict(
                data_dir=DATA_DIR,
                hidden_dim=512,
                hidden_layers=2,
                batch_size=256,
                bias_loss_scale_eV_per_atom=1.0,
                scale_loss_weight=1.0,
                max_train_batches=None,
            ),
        )
