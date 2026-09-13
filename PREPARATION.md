# Complete pipeline and input preparation

This folder now contains the code for raw preprocessing, task labels, train-only scalers, six-task model fitting, original12 CBM, modality ablation, concept-channel restoration, and validation/test tables. Raw participant data, manifests, processed tensors, checkpoints and pretrained ResNet18 weights remain external inputs.

## Choose the required starting point

| Situation | Command | What it does |
|---|---|---|
| Existing processed inputs and labels; fit fresh models | `python -u pipeline.py train --devices 0 1 2 3 --output-root /data/pzhao3/ot2_output/six_task_complete_v1_20260913` | Reuse `study.yaml` input locations; fit full/basic and original12 CBM; val/test tables |
| Completed existing study; update summaries only | `python -u pipeline.py report` | Read existing predictions; regenerate tables and append concept definitions/F1 details |
| Existing original-three-task processed inputs; history/concept labels need rebuilding | `python -u pipeline.py labels` | Build history labels/masters and original12 supervision; write a new study config |
| Reproduce all inputs from raw data | `python -u pipeline.py all --devices 0 1 2 3` | Raw preprocessing → labels/scalers/masters → six-task fitting → val/test tables |
| Inspect commands without reading data or writing outputs | `python pipeline.py all --dry-run` | Print raw preparation commands and subsequent stages |

For `raw`/`all`, first fill in the eight actual manifest TSV paths in `pipeline.yaml`. The raw dataset/clinical and participant paths use the supplied Cluster dataset location; verify `participants.tsv` exists there. For `labels`, only participants, clinical data and the already processed input locations are required; manifests are not used.

After `labels`, train using its generated config:

```bash
python -u pipeline.py train \
  --study-config /data/pzhao3/ot2_output/six_task_prepared_v1_20260913/study.prepared.yaml \
  --devices 0 1 2 3
```

Preparation is explicit and creates new outputs. `raw`/`all` require a fresh `prepared_root`; `labels` requires a fresh `prepared_root/supplemental_inputs`. Individual raw scripts remain available if a preparation stage needs to be rerun. To run the raw and binding stages separately, use:

```bash
python -u pipeline.py raw
python -u pipeline.py labels \
  --study-config /data/pzhao3/ot2_output/six_task_prepared_v1_20260913/study.inputs.yaml
```

Do not use a changed code bundle to resume an older unfinished model run. Use its original bundle to resume, or choose a new model `output_root` for a new experiment. Completed results can still be summarized using `pipeline.py report --output-root <completed-results-root>`; use that run's `runtime.yaml` as `--study-config` if its settings differ from the supplied defaults.

## Included preparation code

All files below are under `data_preprocessing_cluster/`.

| File(s) | Output / role | Source |
|---|---|---|
| `configs/downstream_tasks.json` | Exact original MoCA/CVD broad/CVD strict label and clinical exclusions | Supplied Cluster config; copied byte for byte; no newly added temporal-estimand rules |
| `fit_train_scalers_v2.py` | Clinical/ECG/environment/wearable normalization fitted on the official training population | Supplied Cluster script; copied byte for byte |
| `build_downstream_labels.py`, `build_clinical_tabular_features_v2.py` | Original-three-task outcomes and full clinical tables | Previously supplied Cluster preprocessing |
| `build_ecg_waveform_features_v2.py` | ECG waveforms, masks and machine-report values | Cluster preprocessing |
| `build_environment_5min_sequences.py`, `build_wearable_5min_sequences.py` | Longitudinal tensors and masks | Cluster preprocessing |
| `build_retinal_asset_index_v2.py`, `preprocess_retinal_assets_v2.py` | All canonical retinal assets and processed image views | Cluster preprocessing |
| `build_retinal_model_view_sidecars_v2.py` | Compact training views, per-person caps: photo 6 / OCT 2 / OCTA 16 / FLIO 4 | Existing OT2 full-assets code |
| `build_multimodal_index_v2.py` | Join artifacts, labels and official splits | Cluster preprocessing |
| `build_history_task_labels_v1.py`, `history_tasks_v1.py`, `configs/history_tasks*.json` | Audited source-question/answer-ID definitions for history outcomes | Fast v2 fix1 release; only helper imports adapted |
| `build_cbm_concepts_v1.py`, `configs/cbm_concepts_v1.json` | Original12 measured concept labels, units, category mappings and missing-value rules | Fast v2 fix1 release; exact copies |
| `preparation_helpers.py` | Six existing JSON/hash/atomic-write helpers needed by the history builder | Extracted unchanged from the release's contract module |

The original history builder writes labels/counts for all seven candidate history outcomes. Only osteoporosis, kidney problems and lung problems enter this six-task study. No extra models are scheduled.

## How preparation connects to training

`pipeline.py` runs the original preprocessing scripts. It uses all canonical retinal assets before applying the existing training-view caps, builds 5-minute environment/wearable sequences, and enables condition/procedure features in the full clinical tables. Existing downstream exclusions are still applied. Scalers use official training participants only. The original three tasks keep their separate label definitions and full clinical views.

`prepare_study_inputs.py` joins new history labels onto the original three masters' union of available participants, preserving official splits and nonclinical artifact paths. It copies fitted scaler numbers. History full-clinical data retain the study's existing MoCA-view definition; `data.clinical_inputs` removes each history task's own questionnaire/condition target fields before fitting, and physically selects Basic23 for the Basic model. It does not redefine an unrestricted full history-clinical feature set.

Original12 concept extraction scans the raw clinical tables once for the union of the six tasks' **train/val participants**. The same task-independent labels/schema are copied into each task's expected label directory. Test concept values are not extracted for this experiment. The downstream test labels are used only for evaluation; concept restoration uses predicted full-input channels, as in the existing study.

The generated `study.prepared.yaml` connects all new preparation outputs to the existing four-GPU training runner. The encoders, fusion, Basic23 selection, original12 CBM training and four result-table definitions are unchanged. `pipeline.py train/report` also appends the concept reference and F1 operating-point explanation to both summaries.

## Dependencies and scope

Reuse the working Cluster `ot2` environment. `requirements.txt` lists model dependencies; `data_preprocessing_cluster/requirements-preprocessing-v2.txt` lists raw preprocessing dependencies. Compressed DICOM inputs may additionally need the corresponding pydicom decoder already used on the Cluster. No raw files, manifests or weights are bundled, and no downloads are performed by preparation.

`SOURCE_FILES.json` records the supplied/copied source files and hashes. Regenerating raw outputs with this consolidated recipe has not been validated on the real Cluster dataset; it should not be described as a byte-identical recreation of all historical preprocessing runs. Previously processed inputs remain the default for the existing experiments.

Local integration checks used synthetic data only: preprocessing CLI arguments, history-label binding into the six-task data reader, Basic23 selection and own-target exclusion, original12 extraction, and exclusion of test participants from concept supervision. No real-data preprocessing or model fitting was run locally. No test suite is added to this release.
