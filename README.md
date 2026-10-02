# OT2 — multimodal prediction, modality importance and Mixed CBM

The results follow three questions: how well do the Full/Basic models predict;
how much does removing each modality affect them; and which Mixed-CBM concept
channels recover performance when nonbasic structured clinical inputs are missing?
Raw preprocessing and the original train/validation/test split are unchanged.

## Six-task concept recovery from completed models (no retraining)

This entry point combines MoCA, **the two replacement cardiovascular tasks**,
Osteoporosis, Kidney and Lung. For each task, it evaluates both:

- **Predicted concept recovery:** replace one ID's masked-input concept channels
  with the same participant's full-input predicted concepts.
- **Measured concept supplementation:** replace the observed, encodable channels
  of one ID with recorded measurements/responses, using the checkpoint's saved
  normalization/category order. Other channels remain predicted. This edits the
  concept bottleneck; it does not put the original clinical block back into the
  input encoder.

Start with the no-Observation bank:

```bash
cd ~/OT2
git pull origin main
conda activate ot2
python -u run_concept_recovery.py --bank without_observation --devices 0 1 2 3
```

To analyze both banks, after **both** cardiovascular runs below have finished:

```bash
python -u run_concept_recovery.py --bank both --devices 0 1 2 3
```

`concept_recovery.yaml` maps the original four tasks to the completed September
CBM runs and the two new cardiovascular tasks to the October runs. Edit those
source paths if your cluster directories differ. Each source run's `runtime.yaml`
identifies its own frozen backbone. Full and masked CBM predictions use that
CBM's saved masked-validation threshold; test is not used to optimize thresholds.

No model fitting, raw preprocessing or feature extraction occurs. Complete
AUROC-enabled diagnostics with matching seed/bootstrap settings are copied into
the new report directory and reused. Older Brier-only diagnostics are recomputed
from cached embeddings, saved concept targets and the existing Mixed-CBM head.
The four GPU workers process the remaining tasks. Source runs are read-only, and
interrupted runs resume from completed analyses.

Output:

```text
/data/pzhao3/ot2_output/ot2_six_task_concept_recovery_20261002/
  without_observation/tables/
  with_observation/tables/
```

Each bank has a concise `SUMMARY_CONCEPT_RECOVERY_test.md` and validation
counterpart, plus `PREDICTED_CONCEPT_RECOVERY_ALL_test.csv`,
`MEASURED_CONCEPT_RECOVERY_ALL_test.csv` and their `SELECTED` counterparts.
`ALL` means **all individually evaluated IDs**, not simultaneous restoration.
These tables use the same all-participant cohort for each ID; observed-label
subset analyses and concept-prediction quality remain under
`tables/concept_diagnostics/`. Eligibility counts make measured-label coverage
explicit. All comparisons retain AUROC, F1, BA and Brier with paired CIs.

For each task, split and restoration type separately, the report ranks single
IDs by ΔAUROC and displays every ID strictly above μ + 2σ (population SD,
`ddof=0`). Neither tasks nor measured/predicted gains are pooled to set the
cutoff. Test selects directly on test; validation has its own descriptive list.
No fixed Top10 is imposed. The rule is exploratory and is not a significance
test. Existing F1/BA/Brier q values are recalculated across the six completed
tasks within each metric/intervention family; they can differ from earlier
two-task reports. AUROC receives paired bootstrap CIs without p/q values.

Regenerate just the consolidated tables later with
`python run_concept_recovery.py --bank without_observation --stage report`.
This reads saved CSVs and uses no GPU. `SOURCE_RUNS.json` records each task's
original experiment directory.

## Replacement cardiovascular tasks and AUROC concept selection

Run only the two replacement tasks, with the same training pipeline:

```bash
cd ~/OT2
git pull origin main
conda activate ot2
python -u run.py --config study_cardiovascular.yaml --devices 0 1 2 3
python -u run.py --config study_cardiovascular_with_observation.yaml --devices 0 1 2 3
```

Run these commands in order. The first trains new Full/Basic backbones and
Basic/Mixed CBMs without Observation. The second reuses **these new task-specific
backbones**, and trains new CBMs with the four approved Observation surveys.
It does not reuse the old CVD-broad/strict fitted models. Other tasks are not
rerun. Both commands produce the main val/test reports, Full/Basic modality
ablations, individual and all-ID predicted restoration, plus concept-quality,
Brier and measured-ID supplementation diagnostics. Completed stages are reused
if a run is interrupted.

| New task ID | Outcome definition (recorded history) | Source fields | Concept IDs |
|---|---|---|---|
| `cardiovascular_conditions_binary` | Broad cardiovascular conditions | `mhoccur_hbp`, `mhoccur_clsh`, `mhoccur_circ`, `mhoccur_rvo`, `mhoccur_cvdot` | 316866, 4159131, 2005200015, 440392, 2005200627 |
| `major_cardiovascular_events_binary` | Myocardial infarction or stroke history | `mhoccur_mi`, `mhoccur_strk` | 4329847, 2005200628 |

Label adjudication is unchanged: any confirmed Yes gives a positive label;
every constituent question must explicitly be No to give a negative label.
Conflicting or discordant condition/response evidence is excluded, as are
incomplete negatives. The two outcomes can overlap. Each task excludes its own
outcome fields from Full Clinical inputs and its outcome IDs/fields from the CBM
supervision bank. Basic Clinical retains exactly the same 23 encoded features.

| Task | Train positive / negative | Validation positive / negative | Test positive / negative | Concept IDs: no / with Observation |
|---|---:|---:|---:|---:|
| Broad cardiovascular conditions | 1,117 / 447 | 269 / 83 | 277 / 72 | 107 / 220 |
| Major cardiovascular events | 127 / 1,444 | 37 / 315 | 51 / 298 | 110 / 223 |

These counts were verified from the supplied local dataset. The cluster runner
rebuilds the labels, the task-dependent clinical tables and clinical train
scalers, writing `prepared_inputs/LABEL_COUNTS.csv`. It reuses processed
nonclinical signals and frozen retinal features from `source_root`; it does not
repeat their preprocessing. `participants_tsv`, `clinical_dir` and
`source_root` are configurable. Newly eligible patients must already have
processed cache entries; absent entries are reported rather than silently
dropping those patients.

Training-only minority oversampling remains exactly 1:1 each epoch, with plain
BCE, seed 42, and the original optimization settings. Validation/test keep their
natural proportions. Checkpoints and positive-class F1 thresholds still use
validation only; each model's threshold remains fixed for deletion/restoration.
The reports now include AUROC alongside F1/BA, with 10,000 paired, stratified
bootstrap samples. AUROC is threshold-independent and receives bootstrap CIs,
not permutation p/q values; existing F1/BA/Brier inference is retained.

For these configs, `concept_ranking: auroc_mean_plus_2sd` replaces fixed Top10
display. Within each task and split, rank single Concept IDs by
`AUROC(restored) - AUROC(masked)`. Compute the mean μ and population SD σ
(`ddof=0`) across all finite evaluated individual-ID effects, and report **every
ID strictly above μ + 2σ**. Joint/all-ID restoration is excluded from this
distribution; source fields sharing an ID are still restored together. No rows
are added if none pass, and selection uses unrounded effects. Test reports select
directly on test; validation reports independently use validation. This is an
exploratory descriptive rule, **not statistical significance**. Its bootstrap
CIs do not adjust for selecting on the test results. Predicted restoration and
measured supplementation are ranked separately. Measured-label subsets have
different denominators, so the common all-participant cohort is the primary
supplementation comparison.

The two new output roots are:

- `/data/pzhao3/ot2_output/ot2_cardiovascular_v2_no_observation_20261002`
- `/data/pzhao3/ot2_output/ot2_cardiovascular_v2_with_observation_20261002`

Under each root, `tables/SUMMARY_test.md` and `SUMMARY_val.md` summarize the
main experiments. `CONCEPT_RESTORATION_ALL_test.csv` contains every individual
predicted restoration, including AUROC rank, μ, σ, cutoff and selection flag;
`CONCEPT_RESTORATION_SELECTED_AUROC_test.csv` contains only IDs passing the rule;
`ALL_CONCEPTS_RESTORED_test.csv` reports simultaneous predicted restoration.
Measured supplementation and concept quality are in
`tables/concept_diagnostics/SUMMARY_CONCEPT_DIAGNOSTICS_test.md`, with all-ID CSVs
alongside it. Diagnostic CSV names containing `BRIER` are retained for continuity
and now also contain AUROC. Equivalent validation files are saved.

Existing completed runs need `--stage concept-diagnostics` once to compute the
new AUROC columns; this recomputes only diagnostics, without retraining. Their
default Brier ranking remains unchanged unless the AUROC rule is configured.

## Add Brier, concept-quality and measured-value analyses (no retraining)

For the **two completed** Observation experiments, run:

```bash
cd ~/OT2
git pull origin main
conda activate ot2
python -u run.py --config study_no_observation.yaml --stage concept-diagnostics --devices 0 1 2 3
python -u run.py --config study_with_observation.yaml --stage concept-diagnostics --devices 0 1 2 3
```

Run the commands sequentially; each uses the four GPUs for six task workers.
This stage reads existing Full-model feature caches, Mixed-CBM checkpoints and
already-extracted concept labels. It does not fit models, extract raw data, change
the concept bank, recompute image features, or select thresholds. No new package
is required. If paths/settings differ, use each completed run's `runtime.yaml`
instead of the repository config. Completed task diagnostics are reused on restart.

To regenerate completed diagnostic Markdown and CSV files using **test Brier
ranking** for test results (validation results use their own validation ranking):

```bash
python -u run.py --config /data/pzhao3/ot2_output/ot2_cbm_no_observation_v1_20260923/runtime.yaml --stage concept-diagnostics-report
python -u run.py --config /data/pzhao3/ot2_output/ot2_cbm_with_observation_v1_20260926/runtime.yaml --stage concept-diagnostics-report
```

This report-only stage reads saved per-task diagnostic CSVs. It does not load
checkpoints, run inference, resample bootstrap draws, or use a GPU. It replaces
the diagnostic summaries/aggregate CSVs in the existing output folders.

Results are separate from the original reports, under **each** experiment's
`tables/concept_diagnostics/`:

- `SUMMARY_CONCEPT_DIAGNOSTICS_val.md` and `SUMMARY_CONCEPT_DIAGNOSTICS_test.md`:
  English summaries to share with coauthors.
- `DOWNSTREAM_BRIER_{split}.csv` and `INPUT_REMOVAL_BRIER_{split}.csv`: original
  Full model and Mixed CBM, each with full/masked inputs; Brier, F1, BA and paired
  input-removal effects. Lower Brier is better.
- `CONCEPT_QUALITY_{split}.csv`: every checkpoint concept item, retaining source,
  qualifier and unit. Continuous quality includes raw-unit MAE/RMSE and training-
  standardized MSE; categorical quality includes accuracy, macro recall, log loss
  and multiclass Brier. Training-mean/category-frequency baselines and label
  coverage distinguish weak prediction from a small full/masked difference.
- `PREDICTED_RESTORATION_BRIER_{split}.csv`: all single-ID full-input-prediction
  restorations, with Brier gains added alongside the existing F1/BA outcomes.
- `MEASURED_CONCEPT_SUPPLEMENTATION_{split}.csv`: all single-ID measured-target
  interventions, both over the whole cohort and over participants with at least
  one usable label for that ID. Includes coverage, complete-ID coverage,
  confusion counts, corrected/newly-wrong counts, paired CIs, p/q values, and
  within-split Brier ranks. Both restoration CSVs are sorted within each task
  and population by descending Brier gain; unevaluable IDs appear last.

**Measured targets use the original saved encoding.** Continuous labels are
normalized by the checkpoint's training mean/SD; categorical labels are mapped
to its category order and supplied as one-hot probabilities. Same-ID items move
together, but only observed/encodable/learnable items are changed. Missing labels,
unknown categories and constant/untrained channels stay predicted. Repeated
measurements keep their separate source/unit channels. Labels are never refitted
on validation/test; absent occurrence records are never invented as negatives.
“Measured” includes recorded questionnaire responses, not only numeric laboratory
measurements. Performance may worsen when a predicted-concept classifier receives
measured values; this experiment explicitly evaluates that possibility.

Positive Brier gain means `masked Brier − supplemented Brier`; positive F1/BA
gain means `supplemented − masked`. F1/BA and all interventions use the saved
Mixed-CBM masked-validation threshold. Brier is an **additional exploratory**
probability metric, not a replacement for the original outcomes. Displayed
candidates use the displayed split's Brier rank ≤10 with ties included: **test
reports rank directly on test**, without validation-based selection. The test
header is `Test Brier rank`; CSVs use `brier_rank_within_split` and retain every ID.
Test ranks are exploratory, and intervals do not adjust for selecting the largest
test effects. Observed-label subsets differ by ID; report their sample sizes rather
than comparing their effects as if they had identical populations. Single-class
subsets have undefined BA (reported as NA). Bootstrap/permutation calculations
batch concept IDs on GPU; observed cohorts sharing availability are evaluated
together. The original summary, model checkpoints and analysis completion flags
are not replaced. All-concept **predicted** restoration remains the full-input
CBM endpoint by construction; this extension adds only individual measured-ID
interventions, not joint measured supplementation.

Per-task results and participant-level measured probabilities/availability are
under `TASK/concept_diagnostics/`. Run status is in
`CONCEPT_DIAGNOSTICS_STATUS.json`; progress logs end in `_concept_diagnostics.log`.

## Run the Observation comparison in a separate folder

```bash
cd ~/OT2
git pull origin main
conda activate ot2
python -u run.py --config study_with_observation.yaml --devices 0 1 2 3
```

Use the default `all` stage for this new run: it extracts the concept supervision,
fits new Basic/Mixed CBMs, then produces all current val/test tables. It reuses
the same frozen Full/Basic models and features as the no-Observation experiment;
neither the multimodal models nor raw preprocessing are rerun. Training settings,
seed 42, F1/BA reporting and 10,000 bootstrap draws are unchanged.

Only Observation targets from these four forms are added back:

- `px_food_insecurity_survey`
- `px_neighborhood_environment_survey`
- `px_racial_ethnic_discrimination_survey`
- `px_sdoh_combined_survey`

Each task retains its own outcome exclusions. Measurement, Device, Condition and
Procedure concepts keep their previous task-specific eligibility. The form
restriction applies only to **Observation**: eligible Condition concepts from
`health_survey`, for example, remain available. Survey administration IDs remain
excluded: `4160030` (performed/survey dates), `2005200102` (metadata), `2005200558`
(start timestamps) and `2005200559` (completion timestamps). Substantive survey
questions about frequency or the past 12 months are retained; they are not
administrative timestamps.

| Task | Without Observation | With the four surveys |
|---|---:|---:|
| MoCA | 109 | 222 |
| CVD broad | 104 | 217 |
| CVD strict | 107 | 220 |
| Osteoporosis | 111 | 224 |
| Kidney problems | 111 | 224 |
| Lung problems | 111 | 224 |

These are inventory Concept-ID counts, not the number of learnable channels.
The clean four-survey inventory is the same one used before removing Observation;
this run uses the latest report structure, with Mixed-only concept restoration.

The two output locations are independent:

- With Observation: `/data/pzhao3/ot2_output/ot2_cbm_with_observation_v1_20260926/tables/`
- Without Observation: `/data/pzhao3/ot2_output/ot2_cbm_no_observation_v1_20260923/tables/`

Both produce Full/Basic modality ablations, retained Basic-CBM classification,
Mixed full/masked/all-concepts-restored performance, and Mixed single-ID Top10
plus complete CSV results. Neither runs Basic-CBM task-specific Top3 restoration.
Rerun the same command after interruption; completed stages are reused.

## Update the completed no-Observation experiment

```bash
cd ~/OT2
git pull origin main
conda activate ot2
python -u run.py --config study_no_observation.yaml --stage analyze --devices 0 1 2 3
```

This command reuses the **existing Full/Basic models and both CBM checkpoints**.
It adds the missing Full-model ablations from cached source tokens, using only the
trained fusion and classification head. It neither trains models nor reruns raw
preprocessing or image encoders. Six task workers run over the four allocated GPUs.

Locations in `study_no_observation.yaml`:

- Frozen Full/Basic checkpoints/features (`backbone_root`):
  `/data/pzhao3/ot2_output/ot2_f1_study_v1_20260922`.
- Existing no-Observation CBMs and updated output tables (`output_root`):
  `/data/pzhao3/ot2_output/ot2_cbm_no_observation_v1_20260923`.
- Existing processed inputs/cache (`source_root`):
  `/data/pzhao3/ot2_output/six_task_study_v1_20260913`.

For changed local paths or settings, pass the completed experiment's own
`--config /path/to/results/runtime.yaml`. The first analysis upgrade replaces
analysis tables and the summaries under `output_root`; training files and the
source run stay unchanged. New Full ablation caches are saved under each task's
`analysis/` directory. Old Basic-CBM restoration prediction files, if present,
are historical artifacts and are not read or included in the new reports.
Completed analysis version 3 is reused on subsequent runs; interrupted analysis
restarts only the unfinished task and reuses its saved ablation predictions.

Use `--stage report` only to reformat already-upgraded tables. Do not use it for
the first upgrade, because Full-model ablations must be computed first.

## Experiments and report structure

| Results | Conditions | Outputs |
|---|---|---|
| Downstream classification | Full Clinical model; the same model with only Basic Clinical at inference; separately trained Basic Clinical model | F1, balanced accuracy (BA), 95% CIs and paired comparisons |
| Modality importance | The same 11 branch/family removals in both Full and Basic models | Separate Full/Basic tables, absolute removed performance, drops, availability counts and p/q values |
| Mixed CBM | Full inputs; only Basic Clinical retained in the clinical branch; all predicted concept channels restored after that removal | Direct downstream-performance table plus paired recovery statistics |
| Individual concept restoration | One Concept ID at a time, always from the same Mixed-CBM masked baseline | All IDs in CSV; split-specific Top10 in Markdown |
| Additional Basic CBM result | Basic model versus Basic model + CBM | Classification and fit status retained; no task-specific modality-restoration analysis |

The 11 removals are clinical, ECG, environment, FLIO, OCT, OCTA, photography,
wearable activity, wearable glucose, all retinal modalities and both wearables.
**Full-model clinical ablation removes the entire clinical branch, including basic
information.** Basic-model clinical ablation removes its Basic23 branch. Other
branch removals leave clinical measurements intact; this is not coupled clinical
feature removal. Neither the sign nor size of the result is predetermined.
Validation-F1 top-three markers summarize modality results only. They do not
choose any CBM removal conditions.

Every task's Mixed-CBM experiment removes **only nonbasic clinical inputs**,
retaining the agreed 23 encoded basic features and the other eight modalities.
Full and masked views share one CBM and one validation-selected threshold.

Each concept restoration replaces all channels sharing that Concept ID with the
same patient's **full-input predicted channels**, not measured labels. Repeated
source measurements (e.g. both systolic BP fields) remain one ID-level intervention.
Individual interventions are independent, not cumulative; there is no joint Top10
experiment. Simultaneous all-ID restoration returns the same CBM's full-input
prediction because its classifier has no bypass. It is a diagnostic reference,
not evidence of autonomous reconstruction at deployment.

## CSV tables

Both `val` and `test` are written under `output_root/tables/`:

| File (`test` also has a `val` counterpart) | Contents |
|---|---|
| `SUMMARY_test.md` | All requested tables and statistical interpretation |
| `MODALITY_ABLATION_full_test.csv` | Full model's 11 removals for each task |
| `MODALITY_ABLATION_basic_test.csv` | Basic model's same 11 removals |
| `MIXED_CBM_PERFORMANCE_test.csv` | Full, masked and all-concepts-restored absolute F1/BA with CIs |
| `CONCEPT_RESTORATION_ALL_test.csv` | Every individual Mixed-CBM Concept-ID restoration, ranked on test F1 recovery |
| `ALL_CONCEPTS_RESTORED_test.csv` | All IDs restored simultaneously; absolute performance, recovery, CIs and p/q |

`classification.csv`, `comparisons.csv`, `ablation.csv`, `restoration.csv` and
`cohorts.csv` also combine splits. In `ablation.csv`, **model** distinguishes the
Full and Basic model; join on task, split, model and removed_input.

Concept ranks use competition ranking (1, 1, 3): equal F1 recoveries share a rank.
The Markdown displays up to ten learnable IDs, using ID to break display-order
ties, including at the boundary. `tied_concepts` reports the full number sharing
the effect. Constant/unobserved IDs remain in the CSV without a rank.

To interpret identical effects, the individual CSV also includes:

- `mean_abs_probability_change`, `max_abs_probability_change`: changes relative
  to masked predictions, before thresholding; these are magnitudes, not accuracy gains.
- `n_prediction_changes`: participants whose binary classification changes.
- `tp`, `fp`, `fn`, `tn`: restored confusion counts.
- `probability_group`, `decision_group`: IDs with exactly equal per-patient
  probability or decision vectors share a group number within that task/split/
  removal. These are diagnostic labels, not semantic concept clusters.

The CSVs above contain aggregate results. Detailed participant probabilities stay
in each task's `analysis/` directory on the cluster; do not upload them to Git.

## Concept inventory and training

The current `study_no_observation.yaml` excludes **all Observation-domain targets**.
It retains the approved Measurement, Condition, Procedure and Device entries,
with task-specific outcome exclusions and the four excluded administrative IDs
`4160030`, `2005200102`, `2005200558`, `2005200559`.
Inventory sizes: MoCA 109, CVD broad 104, CVD strict 107, osteoporosis/kidney/lung
111 each. Learnable counts depend on observed training labels and variation.
The domain belongs to the concept dictionary: retained Condition responses can
still be read from `observation.csv`.

`cbm/labels.py` reads only existing local clinical CSVs for supervision. It never
downloads patient data or treats absent disease occurrences as observed negatives.
Distinct source fields, units and qualifiers retain separate supervised channels.
The training split alone defines normalization and concept encodings.

Training uses seed 42 and exact minority oversampling to majority count within
the training split, with unweighted BCE. Val/test are not oversampled. Full/Basic
models use their existing task-specific clinical views; each history task excludes
its own target fields from the historical MoCA-derived full view.

Basic CBM uses Basic-backbone features; Mixed CBM uses Full-backbone features from
paired full/basic-only inputs. Backbones are frozen. Each CBM trains a concept
predictor, a 32-unit classification head, then joint readout tuning with
`BCE + lambda * masked concept loss` (lambda=1). Numeric targets use MSE,
categorical targets cross-entropy. Mixed training gives equal weight to both views.
A joint checkpoint must improve validation F1/BCE and keep each evaluable concept
loss within 10% of the sequential reference, otherwise `sequential_fallback` is
reported. Concept warm-up selects validation concept loss. No new fitting is
performed by the analysis-upgrade command above.

To start a fresh complete fitting run from the existing normalized cache:
`python -u run.py --config study.yaml --devices 0 1 2 3`.
`study.yaml` retains the earlier four-survey concept configuration for reproduction;
`study_no_observation.yaml` is the current no-Observation experiment. Running its
`all` stage fits missing CBMs from the specified frozen backbones. The training
and preprocessing entry points are otherwise unchanged. See [PREPARATION.md](PREPARATION.md)
for the optional historical input-preparation recipe.

## Metrics and inference

The highest threshold maximizing validation positive-class F1 is frozen for each
classifier; checkpoints use validation F1, then BCE to break ties. Full-model
partial-clinical inference uses the original Full threshold. Each model's modality
ablations use its own unperturbed threshold. Both Mixed views and all restorations
use the masked-validation threshold. BA never selects a second threshold.

Drop = baseline minus removed; recovery = restored minus masked. Confidence
intervals use 10,000 paired participant bootstrap resamples stratified by outcome.
Two-sided p values use 10,000 within-participant prediction swaps. BH q values are
separate by split, metric and family: downstream comparisons, CBM comparisons,
Full ablations, Basic ablations, Mixed individual restorations and Mixed all-ID
restoration. Every tested ID enters correction, not just Top10; final q values
require all tasks to finish. Identical binary decisions give p=1.

Test concepts are ranked directly on test as requested. This is exploratory;
ordinary CIs do not account for selecting the largest test effects. Bootstrap
intervals condition on this model, threshold and class counts, and do not include
training-seed variability. Previously inspected test data are not a new independent
confirmatory cohort. All-positive F1 and BA references are provided because F1
alone can be high with high outcome prevalence.

## Code map

- `run.py`, `worker.py`: four-GPU task scheduling and resumable stages.
- `train.py`: model fitting, feature export and fusion-only cached ablations.
- `analysis.py`: Full/Basic ablations and Mixed-only concept restoration.
- `stats.py`: paired F1/BA bootstrap, permutation tests and BH adjustment.
- `report.py`: English Markdown and aggregate CSV tables.
- `cbm/`: approved target extraction and two CBM training modes.
- `data.py`, `common.py`, `vendor/`: existing model/input definitions.
- `data_preprocessing_cluster/`: preserved preprocessing; not invoked by analyze.
