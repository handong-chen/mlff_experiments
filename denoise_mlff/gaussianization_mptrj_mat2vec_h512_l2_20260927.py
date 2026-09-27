"""Fit Gaussianization after selecting a completed composition-pretraining run.

This provider contains only residual preprocessing and Gaussianization fitting.
Set COMPOSITION_RUN_DIR and COMPOSITION_CHECKPOINT_EPOCH after reviewing the run.
The source run is explicit and independent of this provider's code revisions.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_DIR = os.path.expanduser(
    "~/data/MPtrj/composition_reference/composition_preprocess_mptrj_global_holdout_20260927"
)
MODEL_ROOT = os.path.expanduser("~/models")
# Completed source run selected after reviewing training results.
# This directory is independent of the code pins used for Gaussianization.
COMPOSITION_RUN_DIR = os.path.expanduser(
    "~/models/denoise_mlff/composition_pretrain/"
    "composition_gaussianization_mptrj_mat2vec_h512_l2_20260927/"
    "mlff=6583d1d;workshop=9afaf97"
)
# None selects best.pt; an integer selects epoch_<six digits>.pt.
COMPOSITION_CHECKPOINT_EPOCH: int | None = 100


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
        epoch = COMPOSITION_CHECKPOINT_EPOCH
        if epoch is not None and (
            isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
        ):
            raise ValueError("COMPOSITION_CHECKPOINT_EPOCH must be None or a nonnegative integer")
        checkpoint_name = "best.pt" if epoch is None else f"epoch_{epoch:06d}.pt"
        run_dir = Path(COMPOSITION_RUN_DIR).expanduser()
        # Preserve the standard source <training config>/<run> identity in outputs.
        source_tag = os.path.join(run_dir.parent.name, run_dir.name, Path(checkpoint_name).stem)
        fit_config_name = Path(__file__).stem
        residual_dir = os.path.join(
            os.path.expanduser("~/data/MPtrj/energy_gaussianization"),
            fit_config_name, source_tag,
        )
        return dict(
            gaussianization_preprocess=dict(
                composition_data_dir=DATA_DIR,
                composition_checkpoint=str(run_dir / checkpoint_name),
                graph_root=os.path.expanduser("~/data/MPtrj/graph_mmap"),
                output_dir=residual_dir,
                progress_every=100000,
            ),
            gaussianization_fit=dict(
                data_dir=residual_dir,
                output_dir=os.path.join(
                    MODEL_ROOT, "denoise_mlff", "gaussianization", fit_config_name,
                    _resolve_code_version_str(), source_tag,
                ),
                linear_tail_fraction=0.1,
                fit_sample_count=100000,
                seed=1729,
                max_iterations=200,
            ),
        )
