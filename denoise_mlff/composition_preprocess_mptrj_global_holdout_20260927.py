"""Aggregate full MPtrj composition targets, then apply test > val > train.

Keep the existing Stage-1/Stage-2 source split. Any occurrence in test assigns
that composition to test; otherwise validation takes priority over training.
Targets and support counts use all retained trajectories for each composition.
The output is separate from the earlier split-local composition artifact.
"""
from __future__ import annotations

import os
from pathlib import Path


class ConfigProvider:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        return dict(
            composition_preprocess=dict(
                graph_root=os.path.expanduser("~/data/MPtrj/graph_mmap"),
                source_split_scheme="train98_val1_test1_seed0",
                output_dir=str(Path(os.path.expanduser("~/data/MPtrj/composition_reference")) / Path(__file__).stem),
                source_namespace="mptrj",
                trajectory_id_transform="drop_last_dash_component",
                n_elements=119,
                max_atoms=100,
                scale_floor_eV_per_atom=0.05,
                linear_reference_path=None,
                progress_every=100_000,
                target_policy="global_holdout_priority",
            ),
        )
