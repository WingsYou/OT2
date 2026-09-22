# Task-specific individual-concept CBMs

This experiment uses **TASK_SPECIFIC_CONCEPT_SUBSETS.xlsx**, with all its retained entries and the approved outcome exclusions for each of the six tasks. It does not apply the suggested administrative/date exclusions. The original12 experiments remain available through their existing commands.

## Cluster command

In the repository, inside your allocation of four H100 GPUs:

```bash
git pull --ff-only
conda activate ot2
python -u run_individual_concept_cbm.py \
  --source-root /data/pzhao3/ot2_output/six_task_complete_v1_20260913 \
  --devices 0 1 2 3
```

Set `--source-root` to the **actual completed six-task result folder**, containing `runtime.yaml`, `inputs.json`, and `<task>/basic/features_train.pt`, `features_val.pt`, `features_test.pt`, `removals_val.pt`, `removals_test.pt`. Full Clinical checkpoints/caches are read from the same run. If your completed run uses the other existing name, use `/data/pzhao3/ot2_output/six_task_study_v1_20260913` instead.

The default runs **both** experiments for all six tasks. One task/experiment runs per GPU; jobs fill the four GPUs as they become available. Use `--mode basic` or `--mode mixed` to run just one. The parent sets each worker's visible GPU, including inside a scheduler-provided GPU allocation.

The default clinical folder is:

```text
/data/EMED/AIREADI/aireadi300/d55855bd-e2fb-4e53-a07f-e5fd3c41707a/dataset/clinical_data
```

Change it with `--clinical-dir /path/to/clinical_data`. Required files are `person.csv`, `observation.csv`, `measurement.csv`, `condition_occurrence.csv`, and `procedure_occurrence.csv`. Only these supplied files provide participant data. No external data are downloaded.

The original multimodal preprocessing, splits and backbone training are reused. The only new data preparation extracts the **new concept labels**, because the existing original12 label files cannot supply the expanded bank. Each clinical table is read once for all six tasks. If previous full/masked caches live elsewhere, add `--full-cbm-root /path/to/full_clinical_cbm`.

## What is fitted

| Experiment | Frozen representation | CBM training inputs | Deployment evaluation | Restoration experiment |
|---|---|---|---|---|
| Basic | Existing Basic Clinical model | Basic Clinical + eight modalities | Same inputs | Existing single/family modality removals; participant eligible if a targeted modality was available |
| Mixed | Existing Full Clinical model | Equal-weight full and nonbasic-clinical-masked views | Basic clinical retained; other clinical values **and masks** removed; eight modalities retained | Remove nonbasic clinical information; whole labeled split |

Both use the previous linear concept predictor and 32-hidden-unit MLP classifier, with concept warmup, classifier warmup and joint training (`task BCE + mean concept loss`, lambda=1). Continuous concepts use standardized MSE; categorical concepts use cross-entropy and pass probabilities to the head. AdamW settings and epoch budgets follow the previous experiment (400/300/300 epochs by default, patience 40). The previous per-concept 10% validation-loss constraint and sequential fallback remain. For mixed training, checkpoint selection and the Youden threshold use masked validation. For Basic training they use unperturbed Basic validation. Test never chooses the checkpoint, transformation or threshold.

The bottleneck expands to the task's approved items. There is no latent shortcut or direct clinical bypass. Category blocks and continuous losses are batched on GPU. Restoration reuses the additive first classifier layer, and paired bootstrap calculations are batched across interventions. This avoids rerunning the nine encoders for every concept.

## Individual concepts and their values

The runnable dictionary is [`configs/task_specific_concepts.json`](configs/task_specific_concepts.json), exported from `TASK_SPECIFIC_CONCEPT_SUBSETS.xlsx` and the previously saved official AI-READI dictionary. Only definitions and task exclusions are included in the repository; the source workbook and its local-record examples are not needed on the Cluster. No new clinical composites are defined.

| Task | Retained IDs in workbook | Separate ID/source pairs |
|---|---:|---:|
| MoCA | 355 | 492 |
| CVD broad | 350 | 483 |
| CVD strict | 353 | 489 |
| Osteoporosis | 357 | 497 |
| Kidney problems | 357 | 497 |
| Lung problems | 357 | 497 |

An individual item is **Concept ID + source field**, separated further if units or qualifiers differ. Thus two blood-pressure readings and the right/left eye values remain separate. Shared question ID `2005200014` is separated by disease source/qualifier. FORM_NAME is descriptive metadata; it never pools items. One categorical item can use multiple probability channels, which are always restored together. Aliases/overlapping question and response IDs from the approved inventory are retained and identifiable, rather than deduplicated into a clinical construct.

- Every raw row is first filtered by that task's blocked source fields/patterns and blocked IDs in **any concept-ID column**, including disease qualifiers and response IDs. Only approved ID/source pairs are then extracted. Other tasks' nonexcluded history questions remain.
- Numeric/categorical interpretation uses the source field's official type and actual training values. Dates are parsed from the local `M/D/YY H:MM` format (or ISO exports) and encoded in days; a date-like concept name alone does not make a value a date. Scalers and category vocabulary use training participants only. Missing units remain unspecified rather than being assigned a clinical unit.
- Raw categories, including recorded refusal/unknown codes, remain. Test categories absent from training are unsupported, counted explicitly and omitted from supported-label concept accuracy; they do not get mapped to a known category. Task-level performance still uses the entire labeled cohort.
- Missing records are not negative labels. Actual responses take precedence over a matching occurrence-only record. Conflicting repeated values are reported and masked rather than averaged into an invented response. Continuous values are not clinically thresholded or clipped.
- Occurrence-only Procedure entries are explicitly `recorded_occurrence`; a procedure's presence or OMOP quantity does **not** supply the missing per-site sensation result. Some approved items therefore have only a constant recorded target. The source/ID and label coverage are still reported.
- Constant items are retained as fixed outputs; items with no training labels are retained as inactive placeholders. Neither is allowed to become an unconstrained learned channel. They are not claimed to have learned clinical content. Validation-empty items do not contribute a validation alignment constraint.

This is the requested all-entry exploratory bank: identifiers, dates, completion indicators and constant metadata are included. It should not be described as a bank containing only disease-state concepts or only measurements absent from Basic Clinical inputs.

## Reports and resuming

Results are separate from old experiments:

```text
<source-root>/task_specific_individual_cbm_v1/
  labels/<task>/labels.csv, schema.json, coverage.csv
  basic/<task>/model.pt, training.csv, selection.json, ...
  mixed/<task>/model.pt, training.csv, selection.json, ...
  tables/SUMMARY_INDIVIDUAL_basic_val.md
  tables/SUMMARY_INDIVIDUAL_basic_test.md
  tables/SUMMARY_INDIVIDUAL_mixed_val.md
  tables/SUMMARY_INDIVIDUAL_mixed_test.md
```

Each summary reports **AUROC, balanced accuracy and positive-class F1**, item counts/status, and validation-selected top-three single-item restoration with paired 95% intervals. Validation evaluates every item separately; test evaluates those three, their joint combination, and all-item restoration. Selection always uses validation AUROC recovery. The Basic summary uses the previously specified focal removal for each task; detailed CSVs contain every single/family removal, the full validation screen and all per-item concept-prediction metrics. Per-task `concept_definitions.json` provides IDs, source questions, units, qualifiers and train-fitted encoding. Coverage distinguishes raw records from usable, conflicting and supported labels.

Restoration substitutes **unperturbed predicted concepts from the same participant and CBM**, just as before. It is a diagnostic analysis, not measured-concept correction or an autonomous reconstruction method. All-item restoration reproduces the unperturbed classifier by construction. Report negative recovery as well as positive recovery. The unadjusted bootstrap intervals do not resolve multiple testing across hundreds of candidate items.

Rerun the same command to reuse completed labels, model checkpoints and finished task experiments. An interruption during fitting restarts that task's unsaved CBM fit; a saved `model.pt` is reused for evaluation. No new code SHA checks are added. Use a new `--output-root` for changed definitions or settings.

To regenerate summaries without training:

```bash
python -u run_individual_concept_cbm.py \
  --source-root /data/pzhao3/ot2_output/six_task_complete_v1_20260913 \
  --report-only
```

Optional overrides include `--tasks moca_binary kidney_history_binary`, `--bootstrap-samples 2000`, and `--output-root /path/to/new/results`. Existing original12 commands and results are unchanged.
