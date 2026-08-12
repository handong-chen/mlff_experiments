# MLFF Experiments

Experiment `ConfigProvider` modules for ML force-field training.

The `denoise_mlff` package was migrated from
`crystal_experiments/denoise_mlff`. Load providers through `remote_import`
using the `mlff_experiments` repository namespace, for example:

```bash
REMOTEIMPORT_REPO_JSON=/nas/scratch/shared/remote_import/repos.github.json \
REMOTEIMPORT='mlff_experiments:<commit>;mlff:<commit>;workshop:<commit>' \
python -m remote_import.mlff.pipeline.train denoise-pretrain \
  --fit-config-package remote_import.mlff_experiments.denoise_mlff.stage1 \
  --fit-config wgt_d192_l24_attnres_c5_p6_b32k_20260812
```

Use `local:///absolute/path` repository versions only for local smoke tests;
serious runs should use pushed commit pins.

The configs are historical experiment definitions. Use their intended pinned
`mlff` version rather than assuming every older config matches current MLFF
`main`.
