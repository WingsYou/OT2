# Task-Specific Information Signatures

Code for **Bridging the Translational Gap for Multimodal Foundation Models via Task-Specific Information Signatures**.

The repository contains data preparation and the three experiment groups used in the paper:

| Experiment | Evaluation |
|---|---|
| **Full Clinical model** | Downstream classification, basic-only clinical inputs at inference, and 11 modality ablations |
| **Basic Clinical model** | Downstream classification and the same 11 modality ablations |
| **Mixed CBM** | Full/masked-input classification, predicted-concept restoration, and measured-concept supplementation |

**Participant data are not included.** Obtain AI-READI separately; `data/aireadi/` is an empty placeholder. Data, labels, cached features, predictions, and model weights remain outside version control.

## Setup and data

Use Python 3.12 and install a compatible [CUDA PyTorch/torchvision build](https://pytorch.org/get-started/locally/), followed by:

```bash
git clone https://github.com/WingsYou/OT2.git
cd OT2
python -m pip install -r requirements.txt
```

Obtain **AI-READI v3.0.0** through the [official dataset page](https://doi.org/10.60775/fairhub.3) and follow the [download instructions](https://docs.aireadi.org/docs/3/preliminary/azure_storage/azure-download). The study uses the 2,280-participant release and its `recommended_split` assignments.

Place the downloaded dataset contents here, preserving all manifests and referenced files:

```text
data/aireadi/
├── participants.tsv
├── clinical_data/              # person, measurement, observation,
│                              # condition_occurrence, procedure_occurrence CSVs
├── cardiac_ecg/
├── environment/
├── retinal_flio/
├── retinal_oct/
├── retinal_octa/
├── retinal_photography/
├── wearable_activity_monitor/
└── wearable_blood_glucose/
```

The dataset can also remain elsewhere. Edit [`config.yaml`](config.yaml): `preprocessing.dataset_root` and `study.clinical_dir` set data locations; `preprocessing.source_root` and `study.source_root` must refer to the same prepared-input directory. Manifest paths are relative to the dataset root. For personal settings, copy the file to `config.local.yaml`, which is ignored by Git, and pass `--config config.local.yaml` to either entry point.

## Prepare and run

Run these commands from the repository root:

```bash
# Build labels, native inputs, train-only scalers, and the shared image cache.
python -u prepare.py --devices 0 1 2 3

# Train and evaluate all three experiment groups on validation and test sets.
python -u run.py --devices 0 1 2 3
```

Use `--devices 0` for one GPU. Independent jobs share cached inputs and run concurrently on the selected visible GPUs. The image cache uses a frozen ImageNet-pretrained ResNet-18; modality encoders and fusion are trained afterward. ImageNet weights are downloaded on first use; for offline preparation, set `preprocessing.retinal_weights_path` to a local compatible state dictionary.

Preparation runs the original clinical, ECG, sensor, and retinal builders in `preprocessing/`. It preserves participant splits, constructs task-specific clinical views, and fits scalers on training participants only. Historical MoCA/CVD preprocessing views are intermediate files needed by the existing procedure; the final registry exposes the six outcomes below. The shared native cache includes every participant, with task-specific label eligibility applied during training.

```text
outputs/preprocessed/                 # Native preprocessing products and scalers
outputs/prepared/                     # Input registry and shared native/image cache
outputs/study_without_observation/    # Models, analyses, logs, and tables
```

If inputs are already prepared, start directly with `run.py`. Rerunning the same command reuses completed units. `prepare.py --dry-run` previews preparation commands; `prepare.py --stage raw` performs native preprocessing only, and `--stage cache` resumes cache/registry preparation. Use fresh output directories after changing scientific settings.

To refresh analyses or tables from a completed run:

```bash
python -u run.py --config outputs/study_without_observation/runtime.yaml --stage analyze
python run.py --config outputs/study_without_observation/runtime.yaml --stage report
```

## Tasks and experiment definitions

| Task | Recorded outcome |
|---|---|
| `moca_binary` | MoCA score < 26 |
| `cardiovascular_conditions_binary` | Hypertension, dyslipidemia, circulation problems, retinal vascular occlusion, or other heart issues |
| `major_cardiovascular_events_binary` | Myocardial infarction or stroke history |
| `osteoporosis_history_binary` | Osteoporosis history |
| `kidney_history_binary` | Kidney-problem history |
| `lung_history_binary` | Lung-problem history |

Full Clinical uses eligible structured clinical features plus eight nonclinical modalities. Basic Clinical retains 23 encoded basic patient features and the same eight modalities. Both models undergo **11 removals**: clinical information, ECG, environment, FLIO, OCT, OCTA, retinal photography, wearable activity, CGM, all retinal modalities together, and both wearable modalities together. Removing clinical information removes the entire clinical branch of the evaluated model.

Mixed CBM freezes the Full model and trains one shared concept predictor/classifier on paired full and masked representations. **Masking removes only nonbasic clinical features and their availability indicators**, retaining basic patient information and the eight other modalities. The two interventions start from this masked baseline:

- **Predicted restoration:** replace one Concept ID's channels with the same participant's full-input predictions. This probes internal concept-channel dependence. All predicted IDs are also restored simultaneously as a reference; this reproduces full-input CBM predictions by construction.
- **Measured supplementation:** supply recorded, encodable values for the learnable items of one ID. Missing items and unseen categories retain masked predictions. This estimates the potential value of an additional clinical record, on both the full cohort and its observed-label subset.

All channels sharing a Concept ID are changed together. Definitions and source mappings are in [`infosign/cbm/concepts.json`](infosign/cbm/concepts.json); supervision is extracted from local data. The default bank excludes the **Observation target domain**, retaining eligible Measurement, Condition, Device, and Procedure concepts with task-specific outcome exclusions. The source file `observation.csv` is still required for outcome labels and eligible concepts stored there.

Training uses seed 42 and training-only minority oversampling. Validation and test retain natural class proportions. Checkpoints and F1 thresholds are validation-selected; thresholds stay fixed during interventions. Reports include AUROC, positive-class F1, balanced accuracy, supplementary Brier scores, and 10,000 paired, outcome-stratified participant bootstrap resamples. Concept IDs above their task/intervention/split mean ΔAUROC + 2 population SD are highlighted. Test ranking is exploratory; this cutoff is not a significance test, and intervals do not adjust for selection or training-seed variability.

## Results and code

All reports are written under the run's `tables/` directory, with validation and test versions:

| Output | Contents |
|---|---|
| `SUMMARY_test.md` | Classification, both modality-ablation tables, and predicted restoration |
| `SUMMARY_CONCEPT_RECOVERY_test.md` | Both concept interventions and selected IDs |
| `MODALITY_ABLATION_{full,basic}_test.csv` | All 11 removals per model |
| `MIXED_CBM_PERFORMANCE_test.csv` | Full, masked, and all-predicted-IDs-restored performance |
| `PREDICTED_CONCEPT_RECOVERY_{ALL,SELECTED}_test.csv` | Individual predicted-ID effects and selection statistics |
| `MEASURED_CONCEPT_RECOVERY_{ALL,SELECTED}_test.csv` | Individual measured-ID effects and selection statistics |
| `ALL_CONCEPTS_RESTORED_test.csv` | Simultaneous restoration of all predicted IDs |
| `concept_diagnostics/` | Concept prediction quality, probability diagnostics, and observed-label-subset results |

```text
prepare.py         # Data preparation entry point
run.py             # Experiment entry point and parallel scheduling
config.yaml        # All user-editable paths and experiment settings
infosign/
  encoders.py      # Modality encoders, 256-dimensional fusion, classification head
  native_data.py   # Native artifact loading and masks
  data.py          # Shared cache and training data loaders
  train.py         # Full/Basic model training
  analysis.py      # Downstream comparisons and modality ablations
  stats.py         # Sampling, metrics, paired bootstrap statistics
  report.py        # Downstream and ablation reports
  cbm/
    model.py       # Concept predictor, Mixed training, ID restoration
    labels.py      # Local concept supervision
    concepts.json  # Concept definitions and task-specific exclusions
    diagnostics.py # Predicted/measured intervention evaluation
    report.py      # Both concept reports and CSV exports
preprocessing/     # Clinical, ECG, sensor, and retinal preprocessing
```
