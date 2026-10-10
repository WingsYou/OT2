# Task-Specific Information Signatures

Code for **Bridging the Translational Gap for Multimodal Foundation Models via Task-Specific Information Signatures**.

This repository contains the data preprocessing, multimodal models, modality ablation, Mixed Concept Bottleneck Model (CBM), and concept-restoration analyses used in our AI-READI study. The default experiment uses the current six outcomes and excludes Observation-domain concepts from the CBM.

**AI-READI participant data are not distributed here.** `data/aireadi/` is an empty placeholder. Readers must obtain their own authorized copy. Raw data, processed patient tables, labels, checkpoints, cached representations, and participant-level predictions are excluded from Git.

## 1. Obtain and place the dataset

Request **AI-READI v3.0.0** through the [official FAIRhub dataset page](https://doi.org/10.60775/fairhub.3) and follow its access requirements and [download instructions](https://docs.aireadi.org/docs/3/preliminary/azure_storage/azure-download). Our study uses the 2,280-participant release and its `recommended_split` assignments. A different release or the mini dataset will not reproduce the reported cohort.

Place the contents of the downloaded **dataset** directory under `data/aireadi/`, preserving filenames and the original structure:

```text
data/aireadi/
├── participants.tsv
├── clinical_data/
│   ├── person.csv
│   ├── measurement.csv
│   ├── observation.csv
│   ├── condition_occurrence.csv
│   └── procedure_occurrence.csv
├── cardiac_ecg/                 # manifest.tsv and all referenced recordings
├── environment/                 # manifest.tsv and sensor recordings
├── retinal_flio/                # manifest.tsv and image assets
├── retinal_oct/
├── retinal_octa/
├── retinal_photography/
├── wearable_activity_monitor/   # manifest.tsv and wearable recordings
└── wearable_blood_glucose/      # manifest.tsv and CGM recordings
```

Download all nine sources, their manifests, and `participants.tsv`. Clinical CSVs alone are insufficient. The paths inside each manifest must resolve relative to the dataset root. See the [official dataset structure](https://docs.aireadi.org/docs/3/dataset/overall-description).

The dataset may stay on another disk or Cluster mount: set `dataset_root` in [`pipeline.yaml`](pipeline.yaml) to that dataset directory and `clinical_dir` in [`study.yaml`](study.yaml) to its `clinical_data` subdirectory. Update the manifest paths if you store them elsewhere. Nothing needs to be copied into `data/` in that case. Do not commit the dataset or download credentials.

## 2. Install the environment

Use Linux and Python 3.12 with a CUDA-compatible PyTorch/torchvision pair for the full experiment. The study was run on H100 GPUs. Four GPUs allow independent tasks to run concurrently; one GPU is also supported.

```bash
git clone https://github.com/WingsYou/OT2.git
cd OT2
conda create -n infosign python=3.12
conda activate infosign
# Install the matching CUDA torch/torchvision packages for your system first.
python -m pip install -r requirements-all.txt
```

Select the CUDA build using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). `requirements-all.txt` includes model and preprocessing dependencies; an existing working environment can be reused. The frozen ResNet-18 image backbone uses ImageNet weights, downloaded on first use. For offline preparation, set `retinal_weights_path` in `pipeline.yaml` to a compatible local ResNet-18 state dictionary.

## 3. Prepare model inputs

Run commands from the repository root. Review the paths/resources in `pipeline.yaml`, then run:

```bash
# Preview the commands without reading patient files or executing preparation.
python prepare_public_inputs.py --config pipeline.yaml --dry-run

# Raw preprocessing, labels, train-only scalers, and reusable native/image cache.
python -u prepare_public_inputs.py --config pipeline.yaml --devices 0 1 2 3
```

This uses the original builders in `data_preprocessing_cluster/` and writes:

```text
outputs/preprocessed/   # Clinical tables, labels, image/signal tensors, scalers
outputs/prepared/       # inputs.json and shared normalized native/ResNet cache
```

The participant split assignments are preserved. Retinal features are cached for the **frozen ImageNet backbone**; modality encoders and fusion are fitted in the next stage. No AI-READI dataset or study checkpoint is downloaded by the code. See [PREPARATION.md](PREPARATION.md) for processing steps, path configuration, and resuming preparation.

## 4. Run the six-task study

[`study.yaml`](study.yaml) expects `outputs/prepared` and writes a separate experiment to `outputs/study_without_observation`. If you change preparation paths, update `source_root` and `clinical_dir` here too.

```bash
python -u run.py --config study.yaml --devices 0 1 2 3
```

For one GPU, use `--devices 0`. Device numbers refer to GPUs visible to the job; allocate them through your Cluster scheduler first. Native tensors and frozen image features are shared across tasks. Rerunning the same command reuses completed training/analysis units.

| Task | Recorded outcome |
|---|---|
| `moca_binary` | MoCA score < 26 |
| `cardiovascular_conditions_binary` | Hypertension, dyslipidemia, circulation problems, retinal vascular occlusion, or other heart issues |
| `major_cardiovascular_events_binary` | Myocardial infarction or stroke history |
| `osteoporosis_history_binary` | Osteoporosis history |
| `kidney_history_binary` | Kidney-problem history |
| `lung_history_binary` | Lung-problem history |

The experiment includes:

1. **Downstream prediction:** Full Clinical; the same Full model with nonbasic clinical inputs withheld at inference; and a separately trained Basic Clinical model. Basic retains 23 encoded basic patient features and all eight nonclinical modalities.
2. **Modality importance:** single-modality and retinal/wearable family ablations in Full and Basic models. All 11 removals are reported.
3. **Mixed CBM:** one shared concept predictor/classifier trained on full and masked representations from the frozen Full model; evaluated with full and basic-only clinical inputs.
4. **Predicted-concept restoration:** replace one ID's masked-input predictions with the participant's full-input concept predictions. Also restore all IDs simultaneously.
5. **Measured-concept supplementation:** supply available, encodable recorded values at the bottleneck. Report concept prediction quality and both all-participant and observed-label-subset effects.

All channels sharing a Concept ID are manipulated together. The default bank retains eligible Measurement, Condition, Device, and Procedure concepts with task-specific outcome exclusions. [`configs/task_specific_concepts.json`](configs/task_specific_concepts.json) contains definitions and source-field mappings, not patient values. Supervision is extracted from local clinical data. `observation.csv` is still required: the excluded **concept domain** is distinct from the source table containing questionnaire answers used for labels and eligible concepts.

Training uses seed 42 and minority oversampling to balance the **training set only**. Validation/test retain natural class proportions. Checkpoints and F1 thresholds are validation-selected and remain fixed during perturbations. Reports include AUROC, positive-class F1, balanced accuracy, diagnostic Brier scores, and 10,000 paired, outcome-stratified participant bootstrap resamples. Individual IDs with ΔAUROC above their task/intervention/split mean + 2 population SD are highlighted. Test ranking is exploratory, not a significance criterion; intervals do not adjust for this selection or between-training-seed variation.

## 5. Read the outputs

Under `outputs/study_without_observation/tables/`:

| File | Contents |
|---|---|
| `SUMMARY_test.md`, `SUMMARY_val.md` | Classification, Full/Basic ablations, and Mixed-CBM predicted restoration |
| `MODALITY_ABLATION_full_test.csv`, `MODALITY_ABLATION_basic_test.csv` | Every modality removal |
| `MIXED_CBM_PERFORMANCE_test.csv` | Full, masked, and all-predicted-concepts-restored performance |
| `CONCEPT_RESTORATION_ALL_test.csv` | Every **individually** restored Concept ID |
| `CONCEPT_RESTORATION_SELECTED_AUROC_test.csv` | IDs above the descriptive ΔAUROC cutoff |
| `ALL_CONCEPTS_RESTORED_test.csv` | All predicted concept channels restored **simultaneously** |
| `concept_diagnostics/SUMMARY_CONCEPT_DIAGNOSTICS_test.md` | Concept quality, Brier, and measured supplementation |
| `concept_diagnostics/MEASURED_CONCEPT_SUPPLEMENTATION_test.csv` | Every measured-ID intervention, coverage, and effects |

Equivalent validation CSVs are produced. Diagnostic CSV names containing `BRIER` also include AUROC, F1, and balanced accuracy. Checkpoints, patient predictions, and logs stay under the local output directory.

Rebuild tables without fitting:

```bash
python run.py --config outputs/study_without_observation/runtime.yaml --stage report
python run.py --config outputs/study_without_observation/runtime.yaml --stage concept-diagnostics-report
```

After the main study finishes, optional separate recovery and SI-control reports reuse its saved outputs:

```bash
python -u run_concept_recovery.py --devices 0 1 2 3
python -u run_modality_control.py --devices 0 1 2 3
```

Their configs point to the same six-task run by default and write to `outputs/concept_recovery` and `outputs/modality_control`.

## Code map and optional experiments

| Location | Role |
|---|---|
| `data_preprocessing_cluster/` | Label/clinical construction, ECG/sensor/retinal processing, train-only scalers |
| `prepare_public_inputs.py` | Preparation from an independently downloaded dataset |
| `data.py`, `train.py` | Input loading/caching and Full/Basic training |
| `vendor/task2_v2_model.py` | Modality encoders, 256-dimensional fusion, binary classifier |
| `cbm/model.py`, `cbm/experiment.py` | Concept predictor and Mixed-CBM training |
| `cbm/labels.py` | Local concept supervision and exclusions |
| `analysis.py`, `cbm/diagnostics.py`, `stats.py` | Ablations, restoration, paired statistics |
| `report.py`, `cbm/*report.py` | Markdown/CSV tables |

`study_with_observation.yaml` adds concepts from the four approved Observation surveys, reuses the default Full/Basic backbones, and writes a separate output folder; run it after the default study. `study_cardiovascular*.yaml` provides two-task runs. Optional Basic-CBM classification is enabled with `cbm_modes: [basic, mixed]`; the default paper workflow uses `[mixed]`.

Existing Cluster experiments should continue with their saved `runtime.yaml` and original paths. Portable defaults do not point to or overwrite those runs. `pipeline.py`, `prepare_study_inputs.py`, and `preprocessing_study.yaml` retain historical preparation helpers; new users should use `prepare_public_inputs.py`.

Predicted restoration probes internal concept dependence. Measured supplementation evaluates additional recorded information. Neither demonstrates autonomous recovery of missing data. Restoring every full-input predicted channel reproduces the strict CBM's full-input predictions by construction.
