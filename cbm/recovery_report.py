"""Compact summaries of single-ID recovery using existing diagnostic statistics."""
from pathlib import Path

import pandas as pd

from common import TASK_LABELS
from report import md
from cbm.diagnostics_report import ci


SOURCES = (
    ('full_input_prediction', 'PREDICTED', 'Full-input predicted concepts',
     'PREDICTED_RESTORATION_BRIER'),
    ('measured', 'MEASURED', 'Measured concept targets',
     'MEASURED_CONCEPT_SUPPLEMENTATION'),
)
CONDITIONS = {
    'full_original': 'Original model / full inputs',
    'full_original_masked': 'Original model / masked inputs',
    'mixed_full': 'Mixed CBM / full inputs',
    'mixed_masked': 'Mixed CBM / masked inputs',
}


def _label(task):
    return TASK_LABELS.get(task, task)


def _number(value):
    return 'NA' if pd.isna(value) else f'{value:+.6f}'


def _read(path):
    return pd.read_csv(path, dtype={'concept_id': str}, float_precision='round_trip')


def _selection(frame):
    """Use the existing diagnostic decision without another ranking/filter step."""
    values = frame['auroc_selected']
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq('true')


def make_recovery_report(cfg):
    """Call after make_diagnostics_report; never fit models or recompute effects.

    The main exports include every single-ID all-participant row. The existing
    measured diagnostic export retains the separate observed-participant rows.
    """
    root = Path(cfg['output_root'])
    out = root/'tables'
    diagnostics = out/'concept_diagnostics'
    tasks = list(cfg['tasks'])
    bank = ('without Observation' if cfg.get('exclude_observation') else
            'with the four approved Observation surveys')
    for split in ('val', 'test'):
        performance_path = diagnostics/f'DOWNSTREAM_BRIER_{split}.csv'
        if not performance_path.exists():
            continue
        performance = _read(performance_path)
        performance = performance[performance.task.isin(tasks)]
        completed = [task for task in tasks if task in set(performance.task)]
        if not completed:
            continue
        recovery = {}
        for source, prefix, _, legacy in SOURCES:
            frame = _read(diagnostics/f'{legacy}_{split}.csv')
            frame = frame[(frame.task.isin(completed)) &
                          (frame.source == source) &
                          (frame.population == 'all_participants')].copy()
            # Keep configured task order and descending AUROC gain within task.
            frame['_task_order'] = frame.task.map({t: i for i, t in enumerate(tasks)})
            frame = frame.sort_values(
                ['_task_order', 'auroc_rank_within_split', 'concept_id'],
                kind='stable', na_position='last').drop(columns='_task_order')
            selected = frame[_selection(frame)]
            for suffix, data in [('ALL', frame), ('SELECTED', selected)]:
                data.to_csv(out/f'{prefix}_CONCEPT_RECOVERY_{suffix}_{split}.csv', index=False)
            recovery[source] = frame

        lines = [f'# Single-concept recovery — {split}', '',
                 f'Concept bank: **{bank}**. Completed tasks: {len(completed)}/{len(tasks)}. '
                 f'Seed {cfg["seed"]}; {cfg["bootstrap_samples"]:,} paired, outcome-stratified bootstrap resamples.', '',
                 'Each Concept ID is restored separately in the saved Mixed CBM, starting from the same '
                 'masked-input baseline. Masking removes nonbasic clinical inputs; Basic Clinical and '
                 'the other eight modalities remain available. Models, normalization and decision thresholds '
                 'are reused without retraining or threshold optimization.', '',
                 '**Predicted recovery** replaces all channels of one ID with the same participant’s '
                 'full-input predicted concepts. **Measured recovery** replaces observed, encodable, learnable '
                 'items of that ID with their recorded targets using the saved training encoding; '
                 'missing items and unseen categories retain masked predictions.', '',
                 f'Within each task and intervention, select **{split} ΔAUROC > μ + 2σ** among individual IDs '
                 'evaluated on all participants, with population SD (ddof=0). Predicted recovery includes '
                 'constant IDs with zero effect; measured recovery requires usable measured targets. '
                 'Joint/all-ID recovery is excluded. There is no top-10 padding or extra positivity filter. '
                 'This is a descriptive outlier rule, not a significance test. Test selection is exploratory; '
                 'the confidence intervals do not adjust for selection on test or training-seed variability.', '',
                 '## Baseline performance', '',
                 'All entries are point estimates [95% CI]. The original model uses its saved threshold; '
                 'all Mixed CBM conditions use the saved masked-validation threshold.', '']
        performance_rows = []
        for task in completed:
            for condition, title in CONDITIONS.items():
                for row in performance[(performance.task == task) &
                                       (performance.condition == condition)].to_dict('records'):
                    performance_rows.append([_label(task), title, int(row['n']),
                                             *[ci(row, metric) for metric in ('auroc', 'f1', 'ba', 'brier')]])
        lines += [md(['Task', 'Model / input', 'N', 'AUROC', 'F1', 'BA', 'Brier'], performance_rows),
                  '## Selection cutoffs', '',
                  'Statistics and flags below are copied from the diagnostic exports. '
                  'Each intervention has its own distribution; observed-only cohorts do not enter these rankings.', '']
        selection_rows = []
        for task in completed:
            for source, _, title, _ in SOURCES:
                group = recovery[source][recovery[source].task == task]
                if group.empty:
                    continue
                first = group.iloc[0]
                selection_rows.append([_label(task), title, int(first.auroc_selection_n),
                                       _number(first.auroc_selection_mean),
                                       _number(first.auroc_selection_std),
                                       _number(first.auroc_selection_cutoff), int(_selection(group).sum())])
        lines += [md(['Task', 'Intervention', 'Evaluated IDs', 'μ', 'σ', 'μ + 2σ', 'Selected IDs'],
                     selection_rows),
                  '## Selected single IDs', '',
                  'Effects use all participants, including those whose predictions remain unchanged because '
                  'measured targets are unavailable. “Labels any / complete / total” counts participants with '
                  'at least one / all learnable items available, followed by cohort size; this availability '
                  'does not limit predicted recovery. AUROC after recovery is absolute. ΔAUROC, ΔF1 and ΔBA '
                  'are recovery minus masked baseline; Brier improvement is baseline minus recovery. '
                  'Positive changes indicate improvement. All metric entries include 95% CI.', '']
        for task in completed:
            lines += [f'### {_label(task)}', '']
            for source, _, title, _ in SOURCES:
                group = recovery[source][recovery[source].task == task]
                selected = group[_selection(group)]
                lines += [f'**{title}**', '']
                if selected.empty:
                    lines += [f'No single ID exceeded the {split} μ + 2σ cutoff.', '']
                    continue
                rows = []
                for row in selected.to_dict('records'):
                    rows.append([int(row['auroc_rank_within_split']), row['concept_id'],
                                 row['concept_name'],
                                 f'{int(row["n_eligible"])} / {int(row["n_complete"])} / {int(row["n_total"])}',
                                 ci(row, 'restored_auroc'),
                                 *[ci(row, metric+'_gain', True) for metric in ('auroc', 'f1', 'ba', 'brier')]])
                lines += [md(['Rank', 'Concept ID', 'Concept', 'Labels any / complete / total',
                              'AUROC after recovery', 'ΔAUROC', 'ΔF1', 'ΔBA', 'Brier improvement'], rows)]
        lines += ['## Exported results', '',
                  'ALL files retain every single-ID all-participant row, including unavailable IDs, '
                  'absolute metrics, paired effects, confidence intervals, availability, ranks and selection statistics. '
                  'SELECTED files contain exactly the flagged rows. Numerical values are copied from the diagnostic '
                  'exports; the Markdown display is rounded.', '']
        for _, prefix, title, _ in SOURCES:
            filenames = [f'{prefix}_CONCEPT_RECOVERY_{kind}_{split}.csv' for kind in ('ALL', 'SELECTED')]
            lines += [f'- {title}: ' + '; '.join(f'[{name}]({name})' for name in filenames) + '.']
        lines += [f'- Supplementary measured observed-participant effects: '
                  f'[MEASURED_CONCEPT_SUPPLEMENTATION_{split}.csv]'
                  f'(concept_diagnostics/MEASURED_CONCEPT_SUPPLEMENTATION_{split}.csv). '
                  'Observed cohorts may differ between IDs; their effects are not used for main selection.',
                  f'- Full diagnostics, including concept prediction quality and input-removal comparisons: '
                  f'[SUMMARY_CONCEPT_DIAGNOSTICS_{split}.md]'
                  f'(concept_diagnostics/SUMMARY_CONCEPT_DIAGNOSTICS_{split}.md).',
                  '- Source checkpoints and reused runs: [SOURCE_RUNS.json](../SOURCE_RUNS.json).', '']
        (out/f'SUMMARY_CONCEPT_RECOVERY_{split}.md').write_text('\n'.join(lines))
    print(f'[recovery tables] {out}', flush=True)
