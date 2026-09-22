#!/usr/bin/env python3
"""Print validation Concept-ID restoration rankings from completed results."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FOCAL = {'moca_binary': 'retinal_all', 'cvd_broad_binary': 'cardiac_ecg',
         'cvd_strict_binary': 'wearable_activity_monitor', 'osteoporosis_history_binary': 'cardiac_ecg',
         'kidney_history_binary': 'wearable_all', 'lung_history_binary': 'retinal_all'}
LABELS = {'moca_binary': 'MoCA', 'cvd_broad_binary': 'CVD broad', 'cvd_strict_binary': 'CVD strict',
          'osteoporosis_history_binary': 'Osteoporosis history', 'kidney_history_binary': 'Kidney-problem history',
          'lung_history_binary': 'Lung-problem history'}
METRICS = ('auroc', 'balanced_accuracy', 'f1')


def interval(row):
    return f'{float(row["value"]):+.4f} [{float(row["ci_lower"]):+.4f}, {float(row["ci_upper"]):+.4f}]'


def cells(values):
    return '| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in values) + ' |'


def render(root, mode, top_k, all_removals=False):
    lines = [f'# Top {top_k} Concept IDs — {mode}, validation', '',
             'Ranked by single-ID validation AUROC recovery, using the existing saved estimates and confidence intervals. '
             'Each ID includes all of its retained source fields and channels. Only IDs with learnable targets are ranked; '
             'negative recoveries remain eligible. No models or restoration predictions are recomputed.', '',
             'The existing test evaluation includes only the original validation top three IDs. '
             'This report does not evaluate additional test IDs or change the original validation-selected IDs.', '',
             'Intervals are exploratory and unadjusted for multiple comparisons. Restoration uses same-participant '
             'unperturbed predictions, not measured-concept injection.', '']
    count = 0
    for folder in sorted((root / mode).glob('*/by_concept_id')):
        task = folder.parent.name
        definitions = json.loads((folder / 'concept_id_groups.json').read_text())
        rows = defaultdict(dict)
        with (folder / 'restoration.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                cid = row['concept']
                if row['split'] != 'val' or not definitions.get(cid, {}).get('learnable', False):
                    continue
                source = row['removed_input']
                if mode == 'basic' and not all_removals and source != FOCAL[task]:
                    continue
                rows[(source, cid)][row['metric']] = row
        for source in sorted({s for s, _ in rows}):
            ids = sorted((cid for s, cid in rows if s == source),
                         key=lambda cid: (-float(rows[(source, cid)]['auroc']['value']), cid))[:top_k]
            lines += [f'## {LABELS.get(task, task)} — removed {source}', '',
                      cells(['Rank', 'Concept ID', 'Concept', 'Source fields restored together', 'Participants',
                             'AUROC recovery [95% CI]', 'Balanced accuracy recovery [95% CI]', 'F1 recovery [95% CI]']),
                      cells(['---'] * 8)]
            for rank, cid in enumerate(ids, 1):
                d = definitions[cid]; metrics = rows[(source, cid)]
                lines.append(cells([rank, cid, d['concept_name'], ', '.join(d['source_fields']), metrics['auroc']['n'],
                                    *[interval(metrics[m]) for m in METRICS]]))
                count += 1
            lines.append('')
    return '\n'.join(lines), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True, help='Completed individual Concept-ID CBM output directory')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--mode', choices=('both', 'basic', 'mixed'), default='both')
    parser.add_argument('--all-removals', action='store_true', help='Show all Basic removal conditions instead of each task\'s focal removal')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    root = args.result_root.expanduser().resolve()
    found = False
    for mode in ('basic', 'mixed') if args.mode == 'both' else (args.mode,):
        report, count = render(root, mode, args.top_k, args.all_removals)
        if not count:
            print(f'No saved validation restoration rows found for {mode}: {root / mode}')
            continue
        found = True
        tables = root / 'tables'; tables.mkdir(exist_ok=True)
        suffix = '_all_removals' if args.all_removals and mode == 'basic' else ''
        path = tables / f'TOP{args.top_k}_CONCEPT_ID_{mode}_val{suffix}.md'
        path.write_text(report + '\n')
        print(report)
        print(f'\nSaved: {path}\n')
    if not found:
        parser.error('No rankings exported. Point --result-root to the completed CBM experiment output directory.')


if __name__ == '__main__':
    main()
