# Task-specific CBMs with four Observation surveys (v2)

This experiment retains Observation source fields from these four forms:

- `px_food_insecurity_survey`
- `px_neighborhood_environment_survey`
- `px_racial_ethnic_discrimination_survey`
- `px_sdoh_combined_survey`

Each task retains its own outcome exclusions. Measurement, Device, Condition and Procedure concepts keep their previous task-specific eligibility. The form restriction applies only to **Observation**: eligible Condition concepts from `health_survey`, for example, remain available.

Version 2 removes these four IDs and all 20 of their source fields from every task's concept targets:

| Removed Concept ID | Name | Source fields removed |
|---|---|---:|
| 4160030 | Performed (survey dates) | 3 |
| 2005200102 | PhenX MetaData | 10 |
| 2005200558 | Survey Started Timestamp | 4 |
| 2005200559 | Survey Completed Timestamp | 3 |

This is an exact four-ID exclusion. Other IDs retain their original members, including both blood-pressure readings and the monofilament fields, including `msscmpdat`, under ID 4047085. This changes the supervised concept bank; the existing multimodal checkpoints and cached backbone features remain the inputs to the new CBMs.

The bank is derived from `configs/task_specific_concepts.json` by `load_bank()` in `individual_concept_labels.py`. It selects exact ID/source pairs and removes the specified IDs before extracting local records. No form pooling or new clinical composites are introduced.

| Task | Concept IDs | Observation IDs | Measurement | Condition | Procedure | Device | ID/source pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoCA | 222 | 113 | 76 | 29 | 3 | 1 | 245 |
| CVD broad | 217 | 113 | 76 | 24 | 3 | 1 | 240 |
| CVD strict | 220 | 113 | 76 | 27 | 3 | 1 | 243 |
| Osteoporosis history | 224 | 113 | 76 | 31 | 3 | 1 | 247 |
| Kidney-problem history | 224 | 113 | 76 | 31 | 3 | 1 | 247 |
| Lung-problem history | 224 | 113 | 76 | 31 | 3 | 1 | 247 |

Each set has four fewer IDs and 20 fewer ID/source pairs than version 1. Each task retains its own configured label/proxy exclusions; MoCA retains its existing precautionary cognitive-diagnosis exclusions. Units and qualifiers may split an ID/source pair into multiple supervised targets during extraction.

## Run on the Cluster

Inside the allocation of four H100 GPUs:

```bash
cd ~/OT2
git pull --ff-only
conda activate ot2
python -u run_individual_concept_cbm.py \
  --source-root /data/pzhao3/ot2_output/six_task_study_v1_20260913 \
  --concept-bank four-surveys-v2 \
  --mode both \
  --devices 0 1 2 3
```

The default output is a **new** directory:

```text
/data/pzhao3/ot2_output/six_task_study_v1_20260913/task_specific_four_survey_cbm_v2/
```

Existing preprocessing, train/validation/test splits, Basic/Full multimodal checkpoints and feature caches are reused. The new concept labels are extracted from the existing local clinical CSVs once for all six tasks. All twelve CBMs (six tasks × Basic/Mixed) are fitted in the new directory. Concept predictors and classification heads are refitted after the four-ID removal; previous CBM weights are not reused. Subsequent runs with the same command resume completed stages.

The default clinical folder remains:

```text
/data/EMED/AIREADI/aireadi300/d55855bd-e2fb-4e53-a07f-e5fd3c41707a/dataset/clinical_data
```

Use `--clinical-dir` or `--full-cbm-root` only if those existing paths differ. A custom `--output-root` must be separate from results for another concept bank.

## Training and evaluation

| Experiment | Frozen backbone | CBM training inputs | Evaluation and restoration |
|---|---|---|---|
| Basic | Existing Basic Clinical model | Basic clinical + eight other modalities | Val/test classification; single/family modality removal followed by Concept-ID restoration |
| Mixed | Existing Full Clinical model | Equal-weight full and nonbasic-clinical-masked representations | Val/test full/masked classification; restoration after removing nonbasic clinical information |

The two experiments do **not** share fused features. Basic reads `<source-root>/<task>/basic/features_{split}.pt`. Mixed reads the Full model's `<source-root>/<task>/full/features_{split}.pt` and `full_clinical_cbm/<task>/basic_inference_{split}.pt`. If a Mixed cache is missing, it is generated with the Full model's `best.pt`. A masked Full-model representation is not a representation from the separately trained Basic model, even though both retain the same input fields. These caches contain frozen backbone features; the new CBM weights are fitted separately for each task and mode.

The training method is unchanged: concept warmup, classifier warmup and joint fine-tuning (`task BCE + mean concept loss`, lambda = 1). The existing validation alignment constraint may select the sequential fallback. The original encoders and fusion remain frozen. GPU batches, four-worker scheduling and cached-feature restoration follow the existing fast implementation.

Each Concept ID is restored independently on validation. Its source fields, unit/qualifier targets and categorical channels move together. The validation-selected top three IDs are evaluated individually and jointly on test, along with all-ID restoration. Restoration uses the same participant's unperturbed **predicted** concepts as a diagnostic reference. Test data do not choose transformations, checkpoints, thresholds or the top three IDs.

## Outputs

```text
task_specific_four_survey_cbm_v2/
  runtime.yaml
  labels/<task>/labels.csv, schema.json, coverage.csv
  basic/<task>/model.pt, training.csv, by_concept_id/
  mixed/<task>/model.pt, training.csv, by_concept_id/
  tables/SUMMARY_CONCEPT_ID_basic_val.md
  tables/SUMMARY_CONCEPT_ID_basic_test.md
  tables/SUMMARY_CONCEPT_ID_mixed_val.md
  tables/SUMMARY_CONCEPT_ID_mixed_test.md
```

The four summaries include AUROC, balanced accuracy, F1, actual concept coverage, selected training status, single-ID restoration and top-three joint restoration. Detailed CSVs contain all validation IDs and all Basic modality-removal conditions. To regenerate reports, rerun the command with `--report-only`.

### Print the top 10 concepts without rerunning experiments

```bash
python print_concept_top.py \
  --result-root /data/pzhao3/ot2_output/six_task_study_v1_20260913/task_specific_four_survey_cbm_v2 \
  --top-k 10
```

This standard-library-only script reads saved validation results, prints both Basic and Mixed rankings, and saves `tables/TOP10_CONCEPT_ID_basic_val.md` and `tables/TOP10_CONCEPT_ID_mixed_val.md`. Tables include the official concept name, ID, all source fields, participant count, and AUROC/balanced-accuracy/F1 recoveries with intervals. It follows the existing validation AUROC ranking and keeps negative recoveries. Basic uses each task's focal modality removal, matching the original summaries; add `--all-removals` for every saved removal. Change `--top-k` to show more IDs or use `--mode mixed` for one experiment.

Validation already contains all IDs, so this requires no training, inference, GPU, or new bootstrap calculation. Test contains only the original validation-selected top three individual IDs; ranks 4–10 would need additional test evaluation using the fixed trained CBM. The script leaves the existing test selection and summaries unchanged.

Version 1 remains available with `--concept-bank four-surveys`, using `task_specific_four_survey_cbm_v1/` and retaining the four administrative IDs. The previous all-entry experiment remains available with `--concept-bank all` (the existing default), using its original output directory. Full extraction and analysis details are in [INDIVIDUAL_CONCEPT_CBM.md](INDIVIDUAL_CONCEPT_CBM.md).
