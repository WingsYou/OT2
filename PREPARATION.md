# Data preparation

The public entry point is `prepare_public_inputs.py`. It connects the preprocessing scripts used in the study to the input registry and shared cache expected by `run.py`. No downloaded AI-READI patient files, labels, or cached features are included in this repository.

## Inputs and paths

Obtain the v3.0.0 dataset independently using the [official access/download process](https://docs.aireadi.org/docs/3/preliminary/azure_storage/azure-download). Preserve its original directory structure. The preparation requires:

- `participants.tsv`, including participant IDs, age, acquisition metadata, and `recommended_split`.
- `clinical_data/person.csv`, `measurement.csv`, `observation.csv`, `condition_occurrence.csv`, and `procedure_occurrence.csv`.
- The ECG, environment, four retinal, activity, and CGM manifests plus every referenced raw recording/image asset.

The no-Observation CBM still needs `observation.csv` to construct task labels and eligible concepts stored in that table. Excluding an OMOP target domain does not remove its source CSV from input preparation.

Edit `pipeline.yaml`:

| Setting | Default | Meaning |
|---|---|---|
| `dataset_root` | `data/aireadi` | Directory containing `participants.tsv` and the raw source folders |
| `participants` | `{dataset_root}/participants.tsv` | Official participant metadata/split |
| `clinical_dir` | `{dataset_root}/clinical_data` | Raw clinical CSVs |
| `manifests` | `{dataset_root}/SOURCE/manifest.tsv` | Eight input inventories; can be set individually |
| `prepared_root` | `outputs/preprocessed` | Native preprocessing outputs |
| `source_root` | `outputs/prepared` | Ready-to-train registry and shared cache |
| `workers` | `8` | CPU workers for preprocessing |
| `retinal_weights_path` | `null` | Optional offline ImageNet ResNet-18 weights |

Relative preparation paths resolve against the repository root; absolute paths and `~` are supported. For local overrides, copy this config to `pipeline.local.yaml`, which is ignored by Git. Keep the raw and generated directories separate. Do not reuse an existing output directory after changing the dataset or preprocessing settings.

```bash
python prepare_public_inputs.py --config pipeline.yaml --dry-run
python -u prepare_public_inputs.py --config pipeline.yaml --devices 0 1 2 3
```

A dry run prints commands and paths without opening raw patient files. The actual preparation runs raw processing on CPU and distributes frozen retinal feature extraction across the selected GPUs. It does not fit a multimodal or CBM classifier.

## Processing sequence

| Component | Implementation | Output |
|---|---|---|
| Endpoint labels | `build_downstream_labels.py`, `build_history_task_labels_v1.py` | Valid per-task labels, original split assignments |
| Clinical features | `build_clinical_tabular_features_v2.py` | Task-specific wide tables with outcome exclusions |
| Retinal assets | `build_retinal_asset_index_v2.py`, `preprocess_retinal_assets_v2.py`, `build_retinal_model_view_sidecars_v2.py` | Selected deterministic model views and masks |
| ECG | `build_ecg_waveform_features_v2.py` | Parallel 12-lead waveforms, masks, machine-reported numerical features |
| Environment | `build_environment_5min_sequences.py` | Five-minute temporal sequences and masks |
| Activity and CGM | `build_wearable_5min_sequences.py` | Aligned five-minute wearable streams and masks |
| Scaling | `fit_train_scalers_v2.py` | Scalers fitted only on official training participants |
| Input indexes | `build_multimodal_index_v2.py` and the public preparation bridge | Per-task registry and label-free cache index |
| Shared cache | `data.build_cache` | Normalized native tensors and frozen ResNet-18 features |
| Current CVD views | `prepare_cardiovascular_inputs.py` | Replacement CVD labels, clinical tables/scalers, and registry entries |

These files are in `data_preprocessing_cluster/` unless their path is shown otherwise. The public entry point reuses their existing clinical definitions, masks, source availability rules, retinal selection limits, and temporal processing. The original MoCA/legacy-CVD preprocessing views are intermediate artifacts used to retain the existing clinical/scaler procedure; the final registry exposes only the **current six endpoints**.

History tasks retain the study's MoCA-derived clinical feature view, with their own target fields excluded before model input. Replacement cardiovascular tasks have their own clinical views. The common native cache includes every row in `participants.tsv` and contains no task outcome labels. Task-specific eligibility is applied when training, so participants without valid labels for one task are not lost from another task's input cache.

Concept supervision is extracted later by `cbm/labels.py` from the local clinical CSVs and `configs/task_specific_concepts.json`. The study uses task-specific Concept IDs from Measurement, Condition, Device, and Procedure; Observation-domain targets are excluded.

## Outputs and resuming

```text
outputs/preprocessed/
  labels/ clinical/ scalers/
  ecg/ environment_v4/ wearable_v5/
  retinal_index/ retinal/ retinal_model_views/
  model_full_assets_v1/inputs/
  completed_commands.json
outputs/prepared/
  inputs.json
  full_clinical_presence.json
  preparation.yaml
  PREPARATION_COMPLETE.json
  cache/index.csv
  cache/index.metadata.json
  cache/dimensions.json
  cache/people/                  # Private participant tensors/features
  history_labels/ inputs/ cardiovascular_inputs/
```

Rerun the same preparation command to reuse completed commands and cached participants. If native preprocessing is finished, resume at the cache/registry stage:

```bash
python -u prepare_public_inputs.py --config pipeline.yaml --stage cache --devices 0 1 2 3
```

To run native preprocessing only, use `--stage raw`. The resumable marker records completed commands; after changing settings or source files, use fresh output directories instead of mixing artifacts.

Then set `source_root` in `study.yaml` to the prepared directory and run:

```bash
python -u run.py --config study.yaml --devices 0 1 2 3
```

Readers reconstruct the study inputs using their own dataset copy and environment. Exact historical checkpoint weights are not distributed, and a new training run is not promised to be bit-for-bit identical across hardware/library versions.

## Existing Cluster outputs

You do not need to repeat preprocessing for an existing prepared study. Run its saved `runtime.yaml`, or create an ignored `study.local.yaml` pointing to the existing `source_root` and `clinical_dir` and a separate `output_root`. Keep saved experiment settings when resuming.

`prepare_public_inputs.py` is the only preparation entry point. Training and all paper analyses then run through `run.py`; they do not rerun preprocessing.
