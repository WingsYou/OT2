#!/usr/bin/env python3
"""Combine six completed tasks: predicted and measured single-ID recovery."""
import argparse
import fcntl
import os
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
CARDIOVASCULAR_TASKS = {'cardiovascular_conditions_binary', 'major_cardiovascular_events_binary'}


def bank_config(settings, bank):
    sources = settings['source_runs'][bank]
    return dict(output_root=str(Path(settings['output_root']).expanduser().resolve()/bank),
                tasks=settings['tasks'], seed=settings['seed'],
                bootstrap_samples=settings['bootstrap_samples'],
                devices=list(map(str, settings['devices'])),
                cpu_threads=settings['cpu_threads'], num_workers=settings['num_workers'],
                exclude_observation=bank=='without_observation',
                concept_ranking='auroc_mean_plus_2sd',
                diagnostic_sources={task:str(Path(sources['cardiovascular_tasks' if task in
                    CARDIOVASCULAR_TASKS else 'existing_tasks']).expanduser().resolve())
                    for task in settings['tasks']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'concept_recovery.yaml')
    parser.add_argument('--bank', choices=('without_observation','with_observation','both'))
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--stage', choices=('all','report'), default='all')
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    for key in ('devices','output_root'):
        if getattr(args,key) is not None: settings[key] = getattr(args,key)
    bank = args.bank or settings.get('bank','without_observation')
    banks = ('without_observation','with_observation') if bank=='both' else (bank,)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(settings['cpu_threads'])
    from common import write_json
    from run import pool
    from cbm.diagnostics_report import make_diagnostics_report
    from cbm.recovery_report import make_recovery_report

    for bank in banks:
        cfg = bank_config(settings,bank)
        root = Path(cfg['output_root']); root.mkdir(parents=True,exist_ok=True)
        (root/'logs').mkdir(exist_ok=True)
        runtime = root/'runtime.yaml'
        def reports():
            make_diagnostics_report(cfg)
            make_recovery_report(cfg)
        if args.stage=='report':
            reports()
            continue
        with (root/'.run.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'A concept-recovery run is already using {root}') from None
            if runtime.exists():
                previous = yaml.safe_load(runtime.read_text())
                resources = ('devices','cpu_threads','num_workers')
                if {k:v for k,v in previous.items() if k not in resources} != {k:v for k,v in cfg.items() if k not in resources}:
                    raise ValueError('Recovery inputs/settings changed; choose a new --output-root.')
            runtime.write_text(yaml.safe_dump(cfg,sort_keys=False))
            status = dict(status='running',bank=bank,tasks=cfg['tasks'],training=False)
            write_json(root/'STATUS.json',status)
            write_json(root/'SOURCE_RUNS.json',cfg['diagnostic_sources'])
            try:
                print(f'[recovery] {bank}: {len(cfg["tasks"])} tasks; existing checkpoints and thresholds; no training',flush=True)
                jobs = [(task+'_concept_recovery',['--task',task,'--concept-diagnostics']) for task in cfg['tasks']]
                pool(jobs,cfg['devices'],runtime,root,after=reports)
                reports()
                status['status'] = 'complete'; write_json(root/'STATUS.json',status)
            except BaseException as exc:
                status.update(status='failed',error=str(exc)); write_json(root/'STATUS.json',status)
                raise
        print('CONCEPT_RECOVERY_COMPLETE: '+str(root/'tables'),flush=True)


if __name__=='__main__': main()
