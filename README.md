# MLFF Experiments

Experiment `ConfigProvider` modules for ML force-field training.

The `denoise_mlff` package was migrated from
`crystal_experiments/denoise_mlff`. Load providers through `remote_import`
using the `mlff_experiments` repository namespace, for example:

```bash
REMOTEIMPORT_REPO_JSON=/nas/scratch/shared/remote_import/repos.github.json \
REMOTEIMPORT='mlff:<commit>;workshop:<commit>' \
python -m remote_import.mlff.pipeline.train denoise-pretrain \
  --fit-config-package remote_import.mlff_experiments.denoise_mlff.stage1 \
  --fit-config wgt_d192_l24_attnres_c5_p6_b32k_20260812
```

Use `local:///absolute/path` repository versions only for local smoke tests.
Serious runs pin pushed runtime-code revisions such as `mlff` and `workshop`,
intentionally omit `mlff_experiments` from `REMOTEIMPORT`, and record the
observed experiments revision separately.

The configs are historical experiment definitions. Use their intended pinned
`mlff` version rather than assuming every older config matches current MLFF
`main`.

## Current EqMFT dynamic Fourier v2 provider

`denoise_mlff/stage1/eqmft_fourier_v2_20260901.py` is the reviewed fresh
Stage-1 provider for volume-measured dynamic Fourier routing. It uses:

- the checkpoint-incompatible `eqmft_denoiser_meanfield_dynamic_fourier_v2`
  architecture;
- `15.0 Angstrom^3 / cell_volume` reciprocal normalization;
- non-affine equivariant RMS normalization before the Fourier value map;
- six independent gates initialized to `0.1` and bounded by `1.0`;
- a 25,600-atom logical optimizer batch; and
- memory-tier physical caps, starting at 1,800 atoms and 72,000 edges below
  12 GiB.

Reviewed source revisions:

```bash
mlff=27520eb
mlff_experiments_observed=23f6fbf
workshop=9afaf97
```

The canonical workflow intentionally omits `mlff_experiments` from
`REMOTEIMPORT` and records its observed revision separately.

Canonical launch shape:

```bash
cd /nas/scratch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REMOTEIMPORT='mlff:27520eb;workshop:9afaf97'

/home/hdchen/.venv/hf/bin/python \
  -m remote_import.mlff.pipeline.train denoise-pretrain \
  --fit_config_package remote_import.mlff_experiments.denoise_mlff.stage1 \
  --fit_config eqmft_fourier_v2_20260901
```

Do not add a run tag to the canonical run. The provider passed local
construction and CUDA optimizer smokes. A production-shaped Stage-1 run is now
active at
`~/models/denoise_mlff/stage1/eqmft_fourier_v2_20260901/mlff=27520eb;workshop=9afaf97`.
Its epoch-0 metadata records the canonical pin set
`mlff:27520eb;workshop:9afaf97`; the observed `mlff_experiments` revision is
`23f6fbf`. The synchronized 2026-09-02 01:53 UTC snapshot contains completed
checkpoints through epoch 8 and an in-progress epoch-9 checkpoint. The full
scientific and validation record is
`documents/EQMFT_DYNAMIC_FOURIER.md` in the `mlff` repository; its
implementation evidence is anchored to `mlff:27520eb`.
