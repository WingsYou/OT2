#!/usr/bin/env python3
"""SI control: identical modality removals in frozen Full and Basic models."""
import argparse
import fcntl
import os
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
METRICS = {'auroc': 'AUROC', 'f1': 'F1', 'balanced_accuracy': 'BA'}


def source_config(task, cfg):
    from run_concept_recovery import CARDIOVASCULAR_TASKS
    key = 'cardiovascular_tasks' if task in CARDIOVASCULAR_TASKS else 'existing_tasks'
    source = Path(cfg['source_runs'][key]).expanduser().resolve()
    saved = yaml.safe_load((source/'runtime.yaml').read_text())
    backbone = Path(saved.get('backbone_root') or source).expanduser().resolve()
    return {**saved, **cfg, 'backbone_root': str(backbone)}, source


def run_task(task, cfg, device):
    import numpy as np
    import pandas as pd
    import torch
    from common import REMOVALS, read_json, write_json, backbone_dir
    from train import load, cached_removals
    from stats import PairedF1

    cfg, source = source_config(task, cfg)
    backbone = backbone_dir(task, cfg)
    out = Path(cfg['output_root'])/task; out.mkdir(parents=True, exist_ok=True)
    stamp = dict(task=task, seed=cfg['seed'], bootstrap_samples=cfg['bootstrap_samples'],
                 backbone=str(backbone), source_run=str(source), training=False)
    if (out/'DONE.json').exists() and read_json(out/'DONE.json') == stamp:
        print(f'[reuse modality control] {task}', flush=True); return
    (out/'DONE.json').unlink(missing_ok=True)
    removals = {v: cached_removals(task, v, cfg, device, out, source/task/'analysis')
                for v in ('full', 'basic')}
    cuts = {v: read_json(backbone/v/'TRAINING_DONE.json')['threshold'] for v in removals}
    ablations, performance = [], []
    for split in ('val', 'test'):
        features = {v: load(backbone/v/f'features_{split}.pt') for v in removals}
        full, basic = features['full'], features['basic']
        if full['person_ids'] != basic['person_ids'] or not torch.equal(full['y'], basic['y']):
            raise ValueError(f'{task}/{split}: Full and Basic cohorts must match')
        y = full['y'].numpy(); n = len(y)
        boot = PairedF1(y, cfg['bootstrap_samples'], cfg['seed'], device)
        cohort = dict(task=task, split=split, n=n, positive=int(y.sum()), negative=int(n-y.sum()))
        for variant, cache in features.items():
            baseline = cache['original_logits'].sigmoid().numpy()
            removed = removals[variant][split]; cut = cuts[variant]
            probabilities = np.stack([baseline, *[removed[k]['probability'] for k in REMOVALS]])
            # Rank scores once per condition; reuse the same draws for absolute
            # performance and paired drops, without recomputing bootstrap AUROC.
            for metric in METRICS:
                draws = boot.draws(probabilities, cut, metric)
                absolute = boot.intervals(draws)
                drops = boot.intervals(draws[:1]-draws[1:])
                pvalues = boot.pvalues(baseline, probabilities[1:], cut, cut, metric)
                performance.append(dict(**cohort, model=variant, metric=metric, threshold=cut, **absolute[0]))
                for j, (name, drop, p) in enumerate(zip(REMOVALS, drops, pvalues), 1):
                    ablations.append(dict(**cohort, model=variant, removed_input=name, metric=metric,
                        threshold=cut, n_available=int(removed[name]['available'].sum()),
                        baseline=absolute[0]['value'], removed=absolute[j]['value'],
                        removed_ci_lower=absolute[j]['ci_lower'], removed_ci_upper=absolute[j]['ci_upper'],
                        **drop, p_value=float(p)))
        # A separate bridge to the clinical restriction: this keeps Basic23,
        # unlike the all-clinical branch ablation above. No new inference needed.
        for metric in METRICS:
            performance.append(dict(**cohort, model='full_basic_inference', metric=metric,
                threshold=cuts['full'], **boot.performance(full['masked_logits'].sigmoid().numpy(), cuts['full'], metric)))
        print(f'[modality control {task}/{split}] Full + Basic; {len(REMOVALS)} removals each', flush=True)
        del boot
    pd.DataFrame(ablations).to_csv(out/'ablation.csv', index=False)
    pd.DataFrame(performance).to_csv(out/'performance.csv', index=False)
    write_json(out/'DONE.json', stamp)


def make_report(cfg):
    import numpy as np
    import pandas as pd
    from common import REMOVALS, TASK_LABELS, INPUT_LABELS
    from report import md, interval, MODEL_NAMES
    from stats import bh_adjust

    root = Path(cfg['output_root']); out = root/'tables'; out.mkdir(exist_ok=True)
    tasks = [t for t in cfg['tasks'] if (root/t/'DONE.json').exists()]
    if not tasks: return
    ablation = pd.concat([pd.read_csv(root/t/'ablation.csv') for t in tasks], ignore_index=True)
    performance = pd.concat([pd.read_csv(root/t/'performance.csv') for t in tasks], ignore_index=True)
    ablation['q_value'] = np.nan
    for (_, _, metric), idx in ablation.groupby(['split', 'model', 'metric']).groups.items():
        if metric != 'auroc': ablation.loc[idx, 'q_value'] = bh_adjust(ablation.loc[idx, 'p_value'])
    for split in ('val', 'test'):
        a = ablation[ablation.split==split]; p = performance[performance.split==split]
        lines = [f'# Full versus Basic Clinical: modality-ablation control — {split}', '',
            f'Completed tasks: {len(tasks)}/{len(cfg["tasks"])}. Seed {cfg["seed"]}; '
            f'{cfg["bootstrap_samples"]:,} outcome-stratified paired participant bootstrap resamples.', '',
            '## Purpose', '',
            'To assess whether the contribution of nonclinical modalities depends on access to rich structured clinical inputs, '
            'we applied the same input ablations to the separately trained Full Clinical and Basic Clinical models. '
            'The Basic model retains the agreed 23 encoded basic features and the other eight modalities. '
            'This control evaluates model reliance under different clinical-information settings; it does not establish clinical causality. '
            'Small effects in the Full model alone cannot distinguish redundancy from limited use of a modality.', '',
            '## Downstream reference conditions', '',
            'These are the original multimodal models, without CBM. The middle column retains only Basic Clinical at inference '
            'in the Full model; the last column uses the separately trained Basic model.', '']
        conditions = ('full', 'full_basic_inference', 'basic')
        def performance_cell(task, model):
            z = p[(p.task==task)&(p.model==model)].set_index('metric')
            return '<br>'.join(label+': '+interval(z.loc[metric]) for metric, label in METRICS.items())
        lines += [md(['Task', *[MODEL_NAMES[m] for m in conditions]],
                     [[TASK_LABELS[t], *[performance_cell(t, m) for m in conditions]] for t in tasks]),
            '## Matched modality removals', '',
            '**Drop = baseline − removed-input performance; positive values mean deterioration.** '
            'Each cell shows Full and Basic model drops with paired 95% CIs. All 11 removals are displayed; none is selected by its effect.', '',
            'Removing a nonclinical branch leaves the clinical branch intact, including any related clinical measurements. '
            'The clinical row removes the entire clinical branch: all clinical features in Full, or Basic23 in Basic. '
            'It is not the partial nonbasic-clinical removal used in the middle reference column or Mixed-CBM experiment.', '']
        for metric, label in METRICS.items():
            matrix = []
            for removed in REMOVALS:
                row = ['Entire clinical branch (Full / Basic23)' if removed=='clinical_data' else INPUT_LABELS[removed]]
                for task in tasks:
                    z = a[(a.task==task)&(a.removed_input==removed)&(a.metric==metric)].set_index('model')
                    row.append('Full: '+interval(z.loc['full'], True)+'<br>Basic: '+interval(z.loc['basic'], True))
                matrix.append(row)
            lines += [f'### {label} drop', '', md(['Removed input', *[TASK_LABELS[t] for t in tasks]], matrix)]
        lines += ['## Interpretation and exports', '',
            'AUROC assesses discrimination without a classification threshold. F1 is positive-class F1; BA is balanced accuracy. '
            'Each model retains its own original validation-selected threshold for every removal; test is not used to retune thresholds. '
            'Validation and test preserve natural prevalence. All participants are evaluated, including those already missing a removed source; '
            'CSV n_available counts participants with at least one observed source in the removed group.', '',
            'Contrasting drops across models is descriptive; overlapping or non-overlapping individual CIs are not a formal test of the difference between drops. '
            'Paired CIs condition on a single fitted seed and do not include retraining variability. '
            'F1/BA p values use participant prediction swaps; q values use BH correction across completed tasks and all 11 removals, '
            'separately for each model, split and metric. Final q values require all tasks. AUROC has bootstrap CIs without p/q.', '',
            f'`MODALITY_ABLATION_full_{split}.csv` and `MODALITY_ABLATION_basic_{split}.csv` contain all effects, '
            'absolute baseline/removed performance, thresholds, availability and statistics. '
            f'`DOWNSTREAM_REFERENCE_{split}.csv` gives the three reference conditions. Each CSV uses one row per metric. '
            'value/ci_lower/ci_upper mean a performance drop in ablation tables and absolute performance in the reference table.', '']
        for variant in ('full', 'basic'):
            a[a.model==variant].to_csv(out/f'MODALITY_ABLATION_{variant}_{split}.csv', index=False)
        p.to_csv(out/f'DOWNSTREAM_REFERENCE_{split}.csv', index=False)
        (out/f'SUMMARY_MODALITY_CONTROL_{split}.md').write_text('\n'.join(lines))
    print(f'[SI tables] {len(tasks)}/{len(cfg["tasks"])} tasks: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'modality_control.yaml')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--stage', choices=('all', 'report'), default='all')
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    if args.devices: cfg['devices'] = args.devices
    cfg['output_root'] = str((args.output_root or Path(cfg['output_root'])).expanduser().resolve())
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(cfg['cpu_threads'])
    root = Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    if args.stage=='report': make_report(cfg); return
    from common import write_json
    from run import pool
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime = root/'runtime.yaml'
        if runtime.exists():
            old = yaml.safe_load(runtime.read_text()); resources = ('devices', 'cpu_threads', 'num_workers')
            if {k:v for k,v in old.items() if k not in resources} != {k:v for k,v in cfg.items() if k not in resources}:
                raise ValueError('Source/settings changed; use a new --output-root')
        runtime.write_text(yaml.safe_dump(cfg, sort_keys=False))
        status = dict(status='running', training=False, tasks=cfg['tasks'])
        write_json(root/'STATUS.json', status)
        try:
            jobs = [(t+'_modality_control', ['--task', t, '--modality-control']) for t in cfg['tasks']]
            pool(jobs, cfg['devices'], runtime, root, after=lambda: make_report(cfg))
            make_report(cfg)
            status['status'] = 'complete'
        except BaseException as exc:
            status.update(status='failed', error=str(exc)); raise
        finally: write_json(root/'STATUS.json', status)
    print('MODALITY_CONTROL_COMPLETE: '+str(root/'tables'), flush=True)


if __name__=='__main__': main()
