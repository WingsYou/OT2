# OT2 — multimodal prediction, modality importance and Mixed CBM

The results follow three questions: how well do the Full/Basic models predict;
how much does removing each modality affect them; and which Mixed-CBM concept
channels recover performance when nonbasic structured clinical inputs are missing?
Raw preprocessing and the original train/validation/test split are unchanged.

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
