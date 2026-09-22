#!/usr/bin/env python3
"""Task-specific CBMs with Concept-ID restoration; reuse fitted backbones/readouts."""
import argparse
import os
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CLINICAL = '/data/EMED/AIREADI/aireadi300/d55855bd-e2fb-4e53-a07f-e5fd3c41707a/dataset/clinical_data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Completed six-task result folder containing runtime.yaml')
    parser.add_argument('--clinical-dir', type=Path, default=Path(DEFAULT_CLINICAL))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--full-cbm-root', type=Path, help='Existing full/masked feature cache folder, if outside source-root/full_clinical_cbm')
    parser.add_argument('--mode', choices=('both', 'basic', 'mixed'), default='both')
    parser.add_argument('--concept-bank', choices=('all', 'four-surveys'), default='all',
                        help='Task-specific bank; four-surveys limits only the Observation domain')
    parser.add_argument('--devices', nargs='+', default=['0', '1', '2', '3'])
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--bootstrap-samples', type=int)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve()
    folder = 'task_specific_four_survey_cbm_v1' if args.concept_bank == 'four-surveys' else 'task_specific_individual_cbm_v1'
    output = (args.output_root or source / folder).expanduser().resolve()
    if args.report_only:
        cfg = yaml.safe_load((output / 'runtime.yaml').read_text())
    else:
        if (output / 'runtime.yaml').exists():
            previous = yaml.safe_load((output / 'runtime.yaml').read_text())
            if previous.get('concept_bank', 'all') != args.concept_bank:
                raise ValueError('This output contains a different concept bank. Use its default directory or a new --output-root.')
        cfg = yaml.safe_load((source / 'runtime.yaml').read_text())
        cfg.update(output_root=str(source), individual_root=str(output), devices=args.devices,
                   analysis_unit='concept_id', concept_bank=args.concept_bank,
                   individual_modes=['basic', 'mixed'] if args.mode == 'both' else [args.mode],
                   clinical_dir=str(args.clinical_dir.expanduser().resolve()))
        if args.tasks: cfg['tasks'] = args.tasks
        if args.bootstrap_samples is not None: cfg['bootstrap_samples'] = args.bootstrap_samples
        if args.full_cbm_root: cfg['full_cbm_root'] = str(args.full_cbm_root.expanduser().resolve())
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(cfg['cpu_threads'])
    from individual_concept_experiment import make_report
    if args.report_only:
        make_report(cfg); return
    from individual_concept_labels import extract, FOUR_OBSERVATION_FORMS
    from run import pool
    output.mkdir(parents=True, exist_ok=True); (output / 'logs').mkdir(exist_ok=True)
    runtime = output / 'runtime.yaml'; runtime.write_text(yaml.safe_dump(cfg, sort_keys=False))
    label_root = output / 'labels'
    if not (label_root / 'COMPLETE').exists():
        extract(cfg['clinical_dir'], label_root,
                observation_forms=FOUR_OBSERVATION_FORMS if cfg['concept_bank'] == 'four-surveys' else None)
        (label_root / 'COMPLETE').write_text('Labels extracted from supplied clinical tables.\n')
    jobs = [(mode + '_' + task, ['--task', task, '--individual-cbm', mode])
            for task in cfg['tasks'] for mode in cfg['individual_modes']]
    pool(jobs, cfg['devices'], runtime, output, after=lambda: make_report(cfg))
    make_report(cfg)
    print('INDIVIDUAL_CBM_COMPLETE: ' + str(output / 'tables'), flush=True)


if __name__ == '__main__': main()
