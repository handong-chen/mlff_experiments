# EqMFT width-128 with frozen mat2vec inputs

The prepared provider is
`stage1/eqmft_fourier_v3_sparse_site128_mat2vec_frozen_20260926.py`.
It starts fresh Stage-1 denoising and follows
`eqmft_fourier_v3_sparse_site128_20260908` with 24 untied layers and
`site_channels=field_channels=(128, 128, 128)`. Data splits, seed, optimizer,
schedule, and logical batch size remain the same as that width-128 provider.

The input changes are:

```python
atom_embedding_dim=200,
element_embedding_type="mat2vec",
element_embedding_trainable=False,
```

The raw input consists of the published 200D mat2vec vectors for 118 elements,
with a zero masked-Z=0 row. These vectors are stored as a persistent buffer and
cannot be updated by the optimizer. There is no learned input projection or
per-element residual. The rest of the network trains from scratch. The model
API retains the trainability switch; only the frozen Stage-1 config is prepared.

The architecture has distinct identities:

- Stage 1: `eqmft_denoiser_meanfield_dynamic_fourier_v3_mat2vec`.
- Trunk: `eqmft_natural_parity_periodic_meanfield_attnres_dynamic_fourier_v3_mat2vec`.
- Checkpoint marker: `(0x45514D46, 0x5454524B, 8)`; original v3 remains marker 7.

The run uses 30 epochs and a 25,600-atom logical batch. Physical atom and edge
caps use the existing GPU-memory tiers. These are inherited starting estimates
and have not been capacity-tested with 200D inputs. The graph-mmap training
split is `min_energy_frame_rms_q90_uniform_train98_val1_test1_seed0`.
Stage 1 has no energy scale/bias head; that is a Stage-2 model option.

The provider package is `remote_import.mlff_experiments.denoise_mlff.stage1`.
Use the established `remote_import.mlff.pipeline.train denoise-pretrain`
entry point. Serious launches require pushed runtime pins and an observed
experiments revision, following the repository README. No training job is
submitted by preparing this config.

Validation covers table hashes and atomic-number mappings, unchanged frozen
vectors after optimizer steps, architecture separation, conservative Stage-2
integration, and legacy checkpoint compatibility. The frozen provider is
checked against the original width-128 config across GPU-memory tiers.
