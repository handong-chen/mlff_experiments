"""Composition baseline/scale targets from the existing MPtrj graph store.

Uses the source data, trajectory parser, and 100-atom filter of the width128
frozen-mat2vec Stage-1 providers. Select the full train98_val1_test1_seed0
trajectories used by Stage 2: the Stage-1 minimum-energy subset would discard
the relaxation excursions needed for scale supervision.

The 0.05 eV/atom scale floor is an explicit initial choice, not a tuned value.
No fitted elemental reference is supplied; the composition head will learn
the full physical baseline. This provider prepares data only.
"""

from __future__ import annotations

import os
from pathlib import Path


GRAPH_ROOT = os.path.expanduser("~/data/MPtrj/graph_mmap")
OUTPUT_ROOT = os.path.expanduser("~/data/MPtrj/composition_reference")
SOURCE_SPLIT_SCHEME = "train98_val1_test1_seed0"
TRAJECTORY_ID_TRANSFORM = "drop_last_dash_component"
MAX_ATOMS = 100
SCALE_FLOOR_EV_PER_ATOM = 0.05


class ConfigProvider:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        return dict(
            composition_preprocess=dict(
                graph_root=GRAPH_ROOT,
                source_split_scheme=SOURCE_SPLIT_SCHEME,
                output_dir=str(Path(OUTPUT_ROOT) / Path(__file__).stem),
                source_namespace="mptrj",
                trajectory_id_transform=TRAJECTORY_ID_TRANSFORM,
                n_elements=119,
                max_atoms=MAX_ATOMS,
                scale_floor_eV_per_atom=SCALE_FLOOR_EV_PER_ATOM,
                linear_reference_path=None,
                progress_every=100_000,
            ),
        )
