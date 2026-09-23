# OT2 — six-task F1 and balanced accuracy study

One entry point runs the six requested experiment types using **existing processed inputs**. It fits new Full and Basic multimodal models, fits two concept bottlenecks, and writes validation/test tables. The raw preprocessing definitions are unchanged.

## Run on the cluster

Use the existing `ot2` conda environment and the four GPUs allocated to the job:

```bash
cd ~/OT2
git pull origin main
conda activate ot2
python -u run.py --devices 0 1 2 3
```

Defaults in [study.yaml](study.yaml):

- Read-only input/cache root: `/data/pzhao3/ot2_output/six_task_study_v1_20260913`.
- New results: `/data/pzhao3/ot2_output/ot2_f1_study_v1_20260922`.
- Raw clinical CSV directory: the existing AI-READI `dataset/clinical_data` directory.
- **Seed 42, positive-class F1 + balanced accuracy, 10,000 bootstrap resamples.**

The source root must contain `inputs.json`, `full_clinical_presence.json`, `cache/dimensions.json`, and `cache/people/*.pt` from the earlier six-task run. `inputs.json` points to the existing task masters, clinical tables and fitted scalers. These files and cache tensors are reused without rebuilding them. Old fitted models, fusion embeddings and CBMs are **not** reused.

To change the two result locations:

```bash
python -u run.py --devices 0 1 2 3 \
  --source-root /data/pzhao3/ot2_output/six_task_study_v1_20260913 \
  --output-root /data/pzhao3/ot2_output/ot2_f1_study_v1_20260922
```

Rerun the same command after interruption. Completed stages are skipped; interrupted multimodal fitting resumes from its last saved epoch. An interrupted small CBM fit restarts that CBM only. Use a new output directory when changing the experimental settings. A small `runtime.yaml` records settings; there are no code bundles, SHA manifests, or chained audit requirements.

Optional stages: `--stage train`, `--stage analyze`, `--stage report`. `all` is the default. `analyze` uses the new fitted models/features; `report` rebuilds tables from saved analysis CSVs. GPU numbers are relative to `CUDA_VISIBLE_DEVICES` when set by the scheduler. Each worker receives one GPU; up to four independent task/model jobs run concurrently. No model-parallel setup is needed.

## Update completed results: all IDs, test ranking and all-channel restoration

```bash
git pull origin main
python -u run.py --stage analyze --devices 0 1 2 3
```

For a non-default result directory, add `--config /path/to/results/runtime.yaml`.
Existing checkpoints and cached embeddings are reused; no model is retrained and
no preprocessing is rerun. The first upgrade reruns the small CBM heads to evaluate
all Concept IDs on test, because older runs saved only the validation-selected ten.
Subsequent unchanged analysis runs reuse the completed results.

For both Basic and Mixed CBM, every ID is exported to CSV, including constant or
unobserved IDs (`learnable=false`, no rank). Learnable IDs are ranked separately
within val and test by F1 recovery. The MD displays that split's Top 10. **Test
rankings are exploratory descriptions of test results, not held-out confirmation
of validation-selected concepts.** Thresholds and the Basic model's top-three
modality removals remain validation-selected.

Both F1 and balanced accuracy (BA), CIs and paired p/q values are retained.
BA uses the same F1-selected threshold and is not a new ranking objective.
All-channel restoration is also evaluated for each Basic removal and for Mixed
CBM with nonbasic clinical inputs removed (Basic23 plus other modalities remain).
It replaces the entire concept vector with the same participant's unperturbed
predicted vector. Since this strict CBM has no bypass, this must equal that CBM's
full-input performance; it is a diagnostic reference, not autonomous reconstruction.

Files under `tables/`:

- `SUMMARY_val.md`, `SUMMARY_test.md`: Top 10 plus all-channel restoration tables.
- `CONCEPT_RESTORATION_ALL_val.csv`, `CONCEPT_RESTORATION_ALL_test.csv`: every ID,
  professional name, source fields, learnability, split-specific rank, masked/full/
  restored performance, recovery, 95% CIs and p/q for F1 and BA.
- `ALL_CONCEPTS_RESTORED_val.csv`, `ALL_CONCEPTS_RESTORED_test.csv`: simultaneous
  restoration of every channel, including absolute performance and recovery.

These supplementary CSVs contain aggregate results, not participant records.
Use `analyze`, not `report`, for the first upgrade; `report` only formats existing statistics.

## Experiments retained

| Experiment | What is fitted / compared | Selection / reporting |
|---|---|---|
| Downstream classification | Full Clinical; the same Full model with only Basic Clinical available; separately retrained Basic Clinical | Each fitted classifier's best checkpoint and F1 threshold |
| Modality ablation | Remove each of nine sources, all four retinal sources, or both wearable sources from the Basic model | Top **3 of 11** removals by F1 drop |
| Basic CBM | Frozen new Basic model → predicted concepts → new classifier | Concept warm-up, head fitting, joint readout tuning |
| Mixed CBM | Frozen new Full model → one shared CBM trained on paired full and basic-only clinical inputs | Model/threshold selection on basic-only validation inputs; evaluate both views |
| Basic concept restoration | For each selected removal, restore each Concept ID's channels from that participant's unperturbed Basic-CBM predictions | All IDs evaluated; split-specific Top **10** by F1 recovery |
| Mixed concept restoration | Only basic clinical inputs available; restore each ID from that participant's full-input Mixed-CBM predictions | All IDs evaluated; split-specific Top **10** by F1 recovery |

All conditions retain the other eight modalities unless the experiment explicitly removes them. “Basic Clinical” means the previously agreed **23 encoded basic features**. Full models keep the existing task-specific clinical views, including the historical MoCA-derived full view for the three history tasks; each history endpoint's own target fields are excluded as before.

Single-source and family removals can overlap in the top three. Their effects are not additive. Rankings include the largest numerical effects even when fewer than three/ten effects are positive. Test uses validation-selected modality removals but independently ranks concepts on test. Every concept remains available in the supplementary CSV. The retired all-eight-nonclinical removal, original12/expanded13 comparisons, concept-family grouping, field-level ranking, measured-value substitution, and joint-top-k restoration are not run.

## Training and concept definitions

Training oversamples the minority outcome class with replacement to equal the majority count **within the training split only**, retaining every original training participant each epoch. BCE has no additional positive-class weight. Validation/test are never oversampled; their original prevalence and unique participants are preserved. Balancing is a training choice, not a prerequisite for calculating F1.

The multimodal encoders/fusion are unchanged. Frozen ImageNet ResNet-18 outputs are read from the existing cache; temporal encoders, image pooling, fusion and classifier are freshly fitted. Each new model exports unique-participant features for train/val/test. Ablation reruns only fusion/classification on cached source tokens. The stored classifier checkpoint omits the unused CNN; image features continue to come from the source cache.

The approved **task-specific, four-survey concept inventory** is used, not the original12 bank. Observation is restricted to food insecurity, neighborhood environment, racial/ethnic discrimination, and the combined SDOH survey. Other retained domains are unchanged. The four excluded IDs `4160030`, `2005200102`, `2005200558`, `2005200559` and all their source fields remain excluded. Task-specific outcome exclusions also remain in force. The resulting inventory contains 222 IDs for MoCA, 217 for CVD broad, 220 for CVD strict, and 224 for each history task; the number with learnable targets depends on training coverage/variation.

`cbm/labels.py` reads only the existing local clinical CSVs to obtain supervision. It does not rebuild the multimodal inputs or downstream outcomes, and never downloads patient data. Actual values and their observation masks are retained; an absent occurrence is not fabricated into a negative disease label. Units/qualifiers and distinct repeated measurements retain separate supervised channels. Constant or unobserved targets cannot be ranked as learnable concepts. At restoration, **all channels sharing a Concept ID are changed together**; e.g., both systolic blood-pressure measurements are one intervention.

Both CBMs retain the existing small concept predictor and 32-unit classification head, with no direct embedding bypass. Training comprises concept-only warm-up, head fitting, then joint concept/head tuning with `BCE + λ × masked concept loss` (λ = 1). Numeric targets use MSE; categorical targets use cross-entropy. Mixed training equally weights the full and masked view of each sampled participant. Backbone weights stay frozen. The joint checkpoint must retain each evaluable concept's validation loss within 10% of its sequential reference; otherwise the sequential model is retained and reported in `DONE.json`. No extra joint-loss sweep is performed.

Concept encodings and feature normalization are fitted on unique training participants before oversampling. Concept warm-up selects validation concept loss; head/joint phases select validation F1, with BCE as a tie-breaker. Continuous concept loss is still necessary for supervision even though downstream reporting uses classification metrics.

## F1, balanced accuracy, and statistics

- Threshold = highest threshold attaining maximum positive-class validation F1. Checkpoint ties use lower validation BCE.
- Full-model basic-only inference uses the Full model's original threshold. Basic ablations/restorations keep their unperturbed classifier/CBM thresholds. Both Mixed views/restorations use the threshold selected on masked validation inputs.
- F1 **drop** = baseline − removed; F1 **recovery** = restored − removed. Report point estimates and percentile 95% confidence intervals from 10,000 **paired participant**, outcome-stratified bootstrap draws.
- Two-sided p values use 10,000 within-participant prediction-swap permutations. This tests paired prediction exchangeability conditional on the fitted models/thresholds, rather than treating bootstrap draws as independent training runs. Identical decisions give p = 1. Minimum Monte Carlo p is 1/10,001.
- Benjamini–Hochberg q values are calculated separately by metric for downstream comparisons, CBM comparisons, modality ablations, Basic restoration and Mixed restoration, across the six tasks within each split. Both splits include every tested individual ID in the correction, not just Top 10. All-channel restoration uses separate Basic/Mixed test families. Final q values require all six tasks to finish.
- All main comparisons use the same full cohort for each task/split. Available-source counts are included in the ablation CSV; no additional availability subgroup experiment is introduced.

The supplied reference paper motivates reporting performance, paired changes, CIs and corrected significance. Its five-generation Wilcoxon/Mann–Whitney design is not copied into a single-seed patient-level F1 experiment. Bootstrap CIs here condition on the trained model, chosen threshold and observed class counts; they do not include training-seed or threshold-selection uncertainty. Validation results and test-ranked concept findings remain exploratory; the intervals do not correct for selecting the largest test effects. The previously inspected test split is not a newly untouched confirmatory cohort.

High positive prevalence can make an all-positive classifier's F1 high. Each report includes class counts and that F1 reference, so an apparently high F1 is not automatically interpreted as good discrimination. Oversampling does not guarantee better F1 or larger ablation effects.

Concept restoration is a **diagnostic intervention using a full-input reference**. It does not demonstrate autonomous reconstruction of missing clinical information at deployment, and is not substitution with measured ground-truth concepts.

Statistical references: [paired permutation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [training-only resampling](https://imbalanced-learn.org/dev/common_pitfalls.html).

## Outputs and code map

```
output_root/
  runtime.yaml, STATUS.json, logs/
  labels/<task>/                  # local concept labels, coverage and schema
  <task>/full/, basic/            # fresh checkpoints, logs and feature caches
  <task>/basic_cbm/, mixed_cbm/   # CBM weights, concept definitions, fitting status
  <task>/analysis/                # fixed SELECTION.json; paired predictions, CSVs
  tables/SUMMARY_val.md
  tables/SUMMARY_test.md
  tables/*.csv                   # performance, differences, p and q
```

| Code | Responsibility |
|---|---|
| `run.py`, `worker.py`, `study.yaml` | One entry point, four-GPU scheduling and fixed settings |
| `data.py`, `common.py` | Existing input/cache readers, Basic23 selection and source definitions |
| `train.py`, `vendor/task2_v2_model.py` | Multimodal fitting, source-token ablation and feature export |
| `cbm/labels.py`, `cbm/model.py`, `cbm/experiment.py` | Approved labels, typed bottleneck, two CBM training modes |
| `analysis.py`, `stats.py`, `report.py` | Modality selection, split-specific concept rankings, F1/BA statistics and English reports |
| `data_preprocessing_cluster/` | Preserved preprocessing source; not invoked by the new runner |

Only code/configuration belong in Git. Raw data, patient-level predictions, labels and checkpoints remain on the cluster. Old experiment implementations remain available in Git history (before this F1 refactor). See [PREPARATION.md](PREPARATION.md) for the retained optional input-preparation recipe.
